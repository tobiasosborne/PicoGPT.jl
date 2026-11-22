# Run the short commands in loop/notes/check-wvl.md from the repository root.
using PicoGPT, LinearAlgebra, Random
include(joinpath(@__DIR__, "../../src/window.jl"))
const WL = WindowLoop
const options = (; position=:boundary, start=:boundary, injection=:anchor_normalized, lag_bias=4)
words(T) = [1 + ((word >> (j-1)) & 1) for j in 1:T, word in 0:2^T-1]

function witness(c, M)
    θ = zeros(nparams(c))
    p = carve(θ, c)
    block = only(p.blocks)
    a = [1, 1, -1, -1]
    b = [1, -1, -1, 1]
    marker = [1, -1, 1, -1]
    p.E .= hcat(a, -a)
    for gain in (block.γ1, block.γ2, p.γf)
        fill!(gain, 1)
    end
    block.Wv .= b * a' / 4
    block.Wo .= Matrix(I, 4, 4)
    block.b2 .= M .* marker
    p.Wout[1,:] .= sqrt(M) .* a ./ 4
    row = 0
    function ramp!(v, lo, hi, sign)
        for (threshold, slope) in ((lo, 1), (hi, -1))
            row += 1
            block.W1[row,:] .= v ./ 4
            block.b1[row] = -threshold
            block.W2[:,row] .= a .* (sign * slope * M / (hi-lo))
        end
    end
    for u in (-1,1), w in (-1,1)
        ramp!(u .* a + w .* b - 4 .* marker, 1.12, 1.25, u*w)
    end
    for u in (-1,1)
        ramp!(u .* a + 3 .* marker, 1.6, 1.8, u)
    end
    @assert row == 12 && length(θ) == 270
    θ
end

function check_witness(c, M)
    θ = witness(c, M)
    # Four actual two-token queries are a complete lookup table for free decoding.
    pairs = words(2)
    logits, _ = WL.windowgpt(θ, pairs, c; K=1, options...)
    table = reshape([argmax(z) for z in eachcol(logits[:,2:2:end])], 2, 2)
    smallest_margin, largest_blind_gap = Inf, 0.0
    for T in 1:8
        x = words(T)
        labels = accumulate(xor, x .- 1; dims=1)
        for i in axes(x,2)
            answer = x[1,i]
            for j in 2:T
                answer = table[answer, x[j,i]]
                @assert answer == 1 + labels[j,i]
            end
        end
        WL.visit_passes(θ, x, c; K=T+1, options...) do k, Z, rest...
            gaps = reshape(Z[1,:] - Z[2,:], size(x))
            for i in axes(x,2), j in axes(x,1)
                if j <= k+1
                    margin = (1 - 2labels[j,i]) * gaps[j,i]
                    @assert margin > 0.68sqrt(M)
                    smallest_margin = min(smallest_margin, margin)
                else
                    @assert abs(gaps[j,i]) <= inv(sqrt(M))
                    largest_blind_gap = max(largest_blind_gap, abs(gaps[j,i]))
                end
            end
        end
    end
    println("M=$M, known margin $smallest_margin, exterior gap $largest_blind_gap")
end

function check_discrete()
    floor(T,k) = max(T-k-1,0) // T
    one = sum(floor(T,1) for T in 4:12)/9
    mixed = sum(sum(sum(floor(T,k) for k in 1:K)/K for K in T-1:2T)/(T+2) for T in 4:12)/9
    @assert one == 89539//124740 && mixed == 384975233676617//1653814216454400
    for t in 0:8, schedule in (fill.( (0//1,1//10,1//2,4//5,1//1), t)...,
                                [i//(t+1) for i in 1:t])
        odd = 0//1
        for mask in 0:2^t-1
            isodd(count_ones(mask)) || continue
            odd += prod((isodd(mask >> (j-1)) ? e : 1-e for (j,e) in enumerate(schedule)); init=1//1)
        end
        @assert odd == (1-prod((1-2e for e in schedule); init=1//1))/2
    end
    for T in 1:8, x in eachcol(words(T))
        bits = vcat(0, x .- 1)
        address = max.(0, (0:T) .- 1)
        for r in 0:4
            @assert all(address[t+1] == max(0,t-2^r) &&
                        bits[t+1] == reduce(xor, x[max(1,t-2^r+1):t] .- 1) for t in 1:T)
            bits, address = xor.(bits, bits[address .+ 1]), address[address .+ 1]
        end
    end
end

# Check the scalar toy gate, including the direct anchor and the embedding-dependent seed.
function toy(θ, g, x, c, K)
    p = carve(θ,c)
    T = size(x,1)
    U, back_embed = WL.window_embed(x, p.E, :boundary)
    S = WL.initial_stream(U, T, :boundary)
    messages, tapes = Matrix{Float64}[], Function[]
    X = S
    for k in 1:K
        m = X-S
        X, back, _, normback = WL.advance(S+g*m, U, only(p.blocks), c.H, T, :anchor_normalized, 4)
        push!(messages,m)
        push!(tapes, Δ -> back(normback(Δ)))
    end
    Z, back_read = WL.readout(X,p)
    function pull(Δ)
        dX, read_grad = back_read(Δ)
        dg, dU, dS = 0.0, zero(U), zero(S)
        block_grad = zero(flatten(p.blocks))
        for k in K:-1:1
            v, dp, anchor = tapes[k](dX)
            dg += dot(v,messages[k])
            block_grad .+= flatten(dp)
            dU .+= anchor
            dS .+= (1-g)*v
            dX = g*v
        end
        dU .+= WL.initial_stream(dS+dX, T, :boundary)
        flatten(back_embed(dU), block_grad, read_grad), dg
    end
    Z, pull
end

function sandwich(θ, g, x, c)
    p = carve(θ,c)
    apply_block(X,i) = block(X,p.blocks[i],c.H,size(x,1))
    X, embed_pull = embed(x,p.E)
    U, prelude = apply_block(X,1)
    M = zero(U)
    tapes = []
    for k in 1:3
        mid, b2 = apply_block(U .+ g .* M,2)
        previous = M
        M, b3 = apply_block(mid,3)
        push!(tapes,(previous,b2,b3))
    end
    out, coda = apply_block(M,4)
    Z, read_pull = WL.readout(out,p)
    function pull(Δ)
        dOut, read_grad = read_pull(Δ)
        dM, d4 = coda(dOut)
        dU, dg = zero(U), zero(g)
        d2, d3 = zero(flatten(p.blocks[2])), zero(flatten(p.blocks[3]))
        for (previous, b2, b3) in reverse(tapes)
            dMid, grads3 = b3(dM)
            v, grads2 = b2(dMid)
            d2 .+= flatten(grads2)
            d3 .+= flatten(grads3)
            dU .+= v
            dg .+= vec(sum(v .* previous; dims=2))
            dM = g .* v
        end
        dX, d1 = prelude(dU)
        flatten(embed_pull(dX),d1,d2,d3,d4,read_grad), dg
    end
    Z,pull
end

function check_gradient(kind, opened)
    rng = Xoshiro(118)
    c = kind == :toy ? Config(V=2,D=4,H=1,L=1,F=16,T=3) : Config(V=2,D=4,H=1,L=4,F=4,T=2)
    x = reshape([2,1,2][1:c.T],:,1)
    y = accumulate(xor, x .- 1; dims=1) .+ 1
    θ = init(c; rng)
    direction = normalize(randn(rng,length(θ)))
    if kind == :toy
        forward = (θ,g) -> toy(θ,g,x,c,3)
        baseline, back_baseline = WL.windowgpt(θ,x,c; K=1, options...)
        @assert first(forward(θ,1.0)) ≈ first(WL.windowgpt(θ,x,c; K=3, options...))
        gains = (0.0,0.17)
        dg = 1.0
    else
        forward = (θ,g) -> sandwich(θ,g,x,c)
        baseline, back_baseline = gpt(θ,x,c)
        gains = (zeros(c.D),[0.18,-0.09,0.23,0.07])
        dg = normalize(randn(rng,c.D))
    end
    gain = gains[opened ? 2 : 1]
    Z, back = forward(θ,gain)
    _, back_loss = crossentropy(Z,y)
    seed = back_loss()
    pgrad, ggrad = back(seed)
    if iszero(gain)
        @assert Z == baseline && pgrad ≈ back_baseline(seed)
        @assert abs(dot(ggrad,dg)) > 1e-6
    end
    score(p,g) = first(crossentropy(first(forward(p,g)),y))
    h = 1e-6
    est_g = (score(θ,gain+h*dg)-score(θ,gain-h*dg))/(2h)
    est_p = (score(θ+h*direction,gain)-score(θ-h*direction,gain))/(2h)
    err_g = abs(est_g-dot(ggrad,dg))
    err_p = abs(est_p-dot(pgrad,direction))
    @assert err_g < 1e-5 && err_p < 1e-6
    println("$kind gain=$gain, directional derivative $(dot(ggrad,dg)), errors $err_g / $err_p")
end

function check_population()
    toy_c = Config(V=2,D=4,H=1,L=1,F=16,T=4)
    # At closure all length-four terminal states of the witness have zero b coordinate.
    # Read b so the forecast is uniform, while individual gate derivatives can be nonzero.
    θ = witness(toy_c,100)
    p = carve(θ,toy_c)
    p.Wout[1,:] .= [1,-1,-1,1] ./ 4
    x = words(4)
    function terminal_gaps(g)
        Z,_ = toy(θ,g,x,toy_c,3)
        vec(Z[1,4:4:end] - Z[2,4:4:end])
    end
    @assert maximum(abs,terminal_gaps(0.0)) < 1e-12
    h = 1e-7
    gap_direction = (terminal_gaps(h)-terminal_gaps(-h))/(2h)
    gold = vec(reduce(xor,x .- 1; dims=1,init=0))
    derivatives = (gold .- 0.5) .* gap_direction
    @assert abs(sum(derivatives)/length(derivatives)) < 1e-12 && maximum(abs,derivatives) > 1e-6
    println("T=4 uniform readout, population gate derivative $(sum(derivatives)/16), max example $(maximum(abs,derivatives))")
end

if ARGS == ["forward"]
    check_discrete()
    toy_config = Config(V=2,D=4,H=1,L=1,F=16,T=8)
    for M in (100,10_000)
        check_witness(toy_config,M)
    end
    @assert nparams(Config(V=65,L=4))+128 == 810177
    @assert nparams(Config(V=65,L=8)) == 1603137
elseif length(ARGS) == 2 && ARGS[1] in ("toy","sandwich") && ARGS[2] in ("closed","open")
    check_gradient(Symbol(ARGS[1]), ARGS[2] == "open")
elseif ARGS == ["population"]
    check_population()
else
    error("select forward, toy, population, or sandwich")
end

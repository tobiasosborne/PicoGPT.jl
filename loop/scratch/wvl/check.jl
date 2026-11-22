# From the repository root: OPENBLAS_NUM_THREADS=1 julia -t1 --project -O0 loop/scratch/wvl/check.jl
using PicoGPT, LinearAlgebra, Random
include("../../src/window.jl")
using .WindowLoop
BLAS.set_num_threads(1)

function witness(c, M)
    θ = zeros(nparams(c))
    p = carve(θ, c)
    b = only(p.blocks)
    a = [1,1,-1,-1]
    v = [1,-1,-1,1]
    marker = [1,-1,1,-1]
    p.E .= hcat(a,-a)
    b.γ1 .= b.γ2 .= p.γf .= 1
    b.Wv[1,:] .= a/4
    b.Wo[:,1] .= v
    b.b2 .= M * marker
    row = 0
    function plateau!(direction, lo, hi, sign)
        for (threshold, slope) in ((lo,1), (hi,-1))
            row += 1
            b.W1[row,:] .= direction/4
            b.b1[row] = -threshold
            b.W2[:,row] .= (sign*slope*M/(hi-lo)) * a
        end
    end
    for u in (-1,1), w in (-1,1)
        plateau!(u*a + w*v - 4marker, 1.12, 1.25, u*w)
    end
    for u in (-1,1)
        plateau!(u*a + 3marker, 1.6, 1.8, u)
    end
    p.Wout[1,:] .= sqrt(M)*a/4
    θ
end

opts = (; position=:boundary, lag_bias=4, injection=:anchor_normalized)
words(t) = [1 + ((word >> (j-1)) & 1) for j in 1:t, word in 0:2^t-1]
label(x) = 1 .+ mod.(cumsum(x .- 1; dims=1), 2)
function check_block(M)
    max_unknown, min_margin = 0.0, Inf
    for t in 1:8
        c = Config(V=2, D=4, H=1, L=1, F=16, T=t)
        θ = witness(c, M)
        x = words(t)
        y = label(x)
        scores, _ = windowgpt(θ, x, c; opts..., K=t+1, start=:boundary, readouts=:all)
        for (k, z) in enumerate(scores)
            gap = reshape(z[1,:]-z[2,:], size(x))
            front = min(k+1,t)
            min_margin = min(min_margin, minimum((gap .* (3 .- 2y))[1:front,:]))
            if front < t
                max_unknown = max(max_unknown, maximum(abs, gap[front+1:end,:]))
            end
            # Independent fair completions for every ambiguous terminal suffix.
            if front < t
                for rest in axes(x,2)
                    @assert y[end,rest] != y[end,1+xor(rest-1,1)]
                end
            end
        end
        # Decode actual predictions and combine them with the next supplied word.
        last_token = x[1:1,:]
        for j in 2:t
            pair = vcat(last_token, x[j:j,:])
            z, _ = windowgpt(θ, pair, c; opts..., K=1, start=:boundary)
            last_token = reshape(1 .+ (z[2,2:2:end] .> z[1,2:2:end]), 1, :)
            @assert last_token == y[j:j,:]
        end
    end
    @assert max_unknown <= 1.01/sqrt(M) && min_margin >= 0.68sqrt(M)
    (; M, min_margin, max_unknown)
end

function channel()
    for t in 0:8, ε in (0//1, 1//10, 1//2, 4//5, 1//1)
        wrong = sum((ε^count_ones(mask) * (1-ε)^(t-count_ones(mask))
                     for mask in 0:2^t-1 if isodd(count_ones(mask))); init=0//1)
        @assert wrong == (1-(1-2ε)^t)/2
    end
end

# A three-pass scalar gate on the actual anchored block, with final cross-entropy.
function gate_loss(c, θ, x, gain, K)
    p = carve(θ, c)
    U, embed_back = WindowLoop.window_embed(x, p.E, :boundary)
    S = WindowLoop.initial_stream(U, size(x,1), :boundary)
    X = S
    tape = []
    for k in 1:K
        message = X-S
        Y, back, _, norm_back = WindowLoop.advance(S+gain*message, U, only(p.blocks),
                                                   c.H, size(x,1), :anchor_normalized, 4, nothing)
        push!(tape, (; message, back, norm_back))
        X = Y
    end
    Z, read_back = WindowLoop.readout(X,p)
    loss, ce_back = crossentropy(Z,label(x))
    dX, dr = read_back(ce_back(1.0))
    dθ = zero(θ)
    dp = carve(dθ,c)
    for (dst, src) in zip((dp.γf,dp.βf,dp.Wout,dp.bout),dr)
        dst .+= src
    end
    dg, dU, dS = 0.0, zero(U), zero(S)
    for step in reverse(tape)
        dY, db, anchor = step.back(step.norm_back(dX))
        dg += dot(dY,step.message)
        dU .+= anchor
        dS .+= (1-gain)*dY
        dX = gain*dY
        for (dst,src) in zip(values(only(dp.blocks)),db)
            dst .+= src
        end
    end
    dU .+= WindowLoop.initial_stream(dS+dX,size(x,1),:boundary)
    dp.E .= embed_back(dU)
    loss, dg, dθ
end

function check_gate()
    rng = Xoshiro(414)
    c = Config(V=2,D=4,H=1,L=1,F=16,T=3)
    θ = init(c; rng)
    x = reshape([2,1,2],:,1)
    direction = normalize(randn(rng,length(θ)))
    h = 1e-6
    for gain in (0.0,0.2)
        loss,dg,dθ = gate_loss(c,θ,x,gain,3)
        if iszero(gain)
            once,_,dθ1 = gate_loss(c,θ,x,gain,1)
            @assert loss == once && dθ ≈ dθ1 && abs(dg)>1e-7
        end
        fdg = (first(gate_loss(c,θ,x,gain+h,3))-first(gate_loss(c,θ,x,gain-h,3)))/(2h)
        fdθ = (first(gate_loss(c,θ+h*direction,x,gain,3))-
               first(gate_loss(c,θ-h*direction,x,gain,3)))/(2h)
        err = max(abs(fdg-dg), abs(fdθ-dot(dθ,direction)))
        @assert err < 2e-7
        println((;gain, dg, err))
    end
end

for M in (100,10000)
    println(check_block(M))
end
channel()
check_gate()
@assert nparams(Config(V=65,D=128,F=512,T=256,H=4,L=4)) + 128 == 810177
println("Binary-channel sums and Shakespeare parameter count agree.")

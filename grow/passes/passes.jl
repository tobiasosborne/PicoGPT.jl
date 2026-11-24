# Growing in passes while the world lengthens: a looped block on prefix parity.
# Standalone (standard library only). Run: julia passes.jl staged|scratch nseeds cap
#
# Model (270 parameters): V=2, D=4, one head, F=16, no position code.
#   X0 = E[:, token];  X_{j+1} = block(X_j), k times with the same weights;
#   logits = Wu * LN(X_k) + bu.
# Attention sees the position itself and its predecessor only (the first position sees itself).
# Target at position t: parity of the number of `not` (token 2) among positions 1..t.
using LinearAlgebra, Random, Printf
BLAS.set_num_threads(1)

const D, F, V = 4, 16, 2
const NAMES = (:E, :g1, :b1, :Wq, :bq, :Wk, :bk, :Wv, :bv, :Wo, :bo, :g2, :b2, :W1, :c1, :W2, :c2, :gf, :bf, :Wu, :bu)

function init(rng)
    w(m, n) = randn(rng, m, n) .* sqrt(1 / n)
    (E = randn(rng, D, V), g1 = ones(D), b1 = zeros(D),
     Wq = w(D, D), bq = zeros(D), Wk = w(D, D), bk = zeros(D), Wv = w(D, D), bv = zeros(D),
     Wo = w(D, D), bo = zeros(D), g2 = ones(D), b2 = zeros(D),
     W1 = w(F, D), c1 = zeros(F), W2 = w(D, F), c2 = zeros(D),
     gf = ones(D), bf = zeros(D), Wu = w(V, D), bu = zeros(V))
end
nparams(p) = sum(length, values(p))
flat(p) = reduce(vcat, (vec(getfield(p, n)) for n in NAMES))
function unflat(θ, like)
    i = 0
    NamedTuple{NAMES}(map(NAMES) do n
        a = getfield(like, n); r = reshape(θ[i+1:i+length(a)], size(a)); i += length(a); copy(r)
    end)
end

# ---- operations, each returning value and pullback ----
function layernorm(X, g, b)
    μ = sum(X, dims = 1) ./ size(X, 1)
    C = X .- μ
    s = sqrt.(sum(abs2, C, dims = 1) ./ size(X, 1) .+ 1e-5)
    Z = C ./ s
    function back(Ȳ)
        Z̄ = Ȳ .* g
        X̄ = (Z̄ .- sum(Z̄, dims = 1) ./ size(X, 1) .- Z .* (sum(Z̄ .* Z, dims = 1) ./ size(X, 1))) ./ s
        X̄, vec(sum(Ȳ .* Z, dims = 2)), vec(sum(Ȳ, dims = 2))
    end
    Z .* g .+ b, back
end
affine(X, W, b) = (W * X .+ b, Ȳ -> (W' * Ȳ, Ȳ * X', vec(sum(Ȳ, dims = 2))))

# shift(M)[:, n] = M[:, n-1] inside a sequence, zero at each first position; unshift is its adjoint
function shift(M, T)
    S = zero(M)
    @inbounds for n in 2:size(M, 2)
        (n - 1) % T == 0 && continue
        @views S[:, n] .= M[:, n-1]
    end
    S
end
function unshift(M, T)
    S = zero(M)
    @inbounds for n in 2:size(M, 2)
        (n - 1) % T == 0 && continue
        @views S[:, n-1] .= M[:, n]
    end
    S
end

function window_mix(Q, K, Vv, T)
    N = size(Q, 2)
    α = 1 / sqrt(D)
    nonfirst = reshape([(n - 1) % T != 0 for n in 1:N], 1, N)
    dK = (K .- shift(K, T)) .* nonfirst
    Vp = shift(Vv, T)
    gap = α .* sum(Q .* dK, dims = 1)
    a = ifelse.(nonfirst, 1 ./ (1 .+ exp.(-gap)), 1.0)     # weight on the position itself
    O = a .* Vv .+ (1 .- a) .* Vp
    function back(Ō)
        ḡ = sum(Ō .* (Vv .- Vp), dims = 1) .* a .* (1 .- a) .* α .* nonfirst
        V̄ = a .* Ō .+ unshift((1 .- a) .* Ō, T)
        Q̄ = ḡ .* dK
        K̄ = ḡ .* Q
        K̄ = K̄ .- unshift(K̄, T)
        Q̄, K̄, V̄
    end
    O, back
end

function block(X, p, T)
    A, bl1 = layernorm(X, p.g1, p.b1)
    Q, bq = affine(A, p.Wq, p.bq)
    K, bk = affine(A, p.Wk, p.bk)
    Vv, bv = affine(A, p.Wv, p.bv)
    O, bm = window_mix(Q, K, Vv, T)
    Y1, bo = affine(O, p.Wo, p.bo)
    X1 = X + Y1
    Fh, bl2 = layernorm(X1, p.g2, p.b2)
    U, b1 = affine(Fh, p.W1, p.c1)
    R = max.(U, 0)
    Y2, b2 = affine(R, p.W2, p.c2)
    function back(X̄2)
        R̄, W̄2, c̄2 = b2(X̄2)
        F̄, W̄1, c̄1 = b1(R̄ .* (U .> 0))
        X̄1, ḡ2, b̄2 = bl2(F̄)
        X̄1 = X̄1 + X̄2
        Ō, W̄o, b̄o = bo(X̄1)
        Q̄, K̄, V̄ = bm(Ō)
        Aq, W̄q, b̄q = bq(Q̄); Ak, W̄k, b̄k = bk(K̄); Av, W̄v, b̄v = bv(V̄)
        X̄, ḡ1, b̄1 = bl1(Aq + Ak + Av)
        X̄ + X̄1, (g1 = ḡ1, b1 = b̄1, Wq = W̄q, bq = b̄q, Wk = W̄k, bk = b̄k, Wv = W̄v, bv = b̄v,
                  Wo = W̄o, bo = b̄o, g2 = ḡ2, b2 = b̄2, W1 = W̄1, c1 = c̄1, W2 = W̄2, c2 = c̄2)
    end
    X1 + Y2, back
end

# all 2^T sequences, concatenated; tokens and targets in 1:2
function world(T)
    toks = Int[]; tgt = Int[]
    for s in 0:2^T-1
        par = 0
        for t in 1:T
            b = (s >> (t - 1)) & 1
            par ⊻= b
            push!(toks, b + 1); push!(tgt, par + 1)
        end
    end
    toks, tgt
end

"Mean cross-entropy (nats per position), accuracy, and (if grad) the gradient as a NamedTuple."
function loss(p, toks, tgt, T, k; grad = false)
    N = length(toks)
    X = p.E[:, toks]
    backs = Any[]
    for _ in 1:k
        X, b = block(X, p, T)
        push!(backs, b)
    end
    Z, bf = layernorm(X, p.gf, p.bf)
    L, bu = affine(Z, p.Wu, p.bu)
    m = maximum(L, dims = 1)
    lse = m .+ log.(sum(exp.(L .- m), dims = 1))
    ℓ = 0.0; correct = 0
    for n in 1:N
        ℓ += lse[n] - L[tgt[n], n]
        correct += (L[tgt[n], n] > L[3-tgt[n], n])
    end
    ℓ /= N
    grad || return ℓ, correct / N, nothing
    P = exp.(L .- lse)
    for n in 1:N; P[tgt[n], n] -= 1; end
    P ./= N
    Z̄, W̄u, b̄u = bu(P)
    X̄, ḡf, b̄f = bf(Z̄)
    acc = nothing
    for b in reverse(backs)
        X̄, g = b(X̄)
        acc = acc === nothing ? g : map(+, acc, g)
    end
    Ē = zero(p.E)
    for n in 1:N; @views Ē[:, toks[n]] .+= X̄[:, n]; end
    ℓ, correct / N, merge((E = Ē,), acc, (gf = ḡf, bf = b̄f, Wu = W̄u, bu = b̄u))
end

mutable struct Adam; m; v; t; end
Adam(n) = Adam(zeros(n), zeros(n), 0)
function step!(o::Adam, θ, g; lr = 1e-2)
    o.t += 1
    @. o.m = 0.9 * o.m + 0.1 * g
    @. o.v = 0.999 * o.v + 0.001 * g^2
    @. θ -= lr * (o.m / (1 - 0.9^o.t)) / (sqrt(o.v / (1 - 0.999^o.t)) + 1e-8)
end

"""
The exact floor for k passes on length T. With no position code, a first position is the same
as a position whose predecessor is in the same state, so some histories cannot be told apart
by any weights. Refine a colouring k times, colour ← (own colour, predecessor's colour, or own
at a first position); the floor is the entropy of the target given the final colour.
"""
function floor_exact(T, k)
    toks, tgt = world(T)
    c = copy(toks)
    for _ in 1:k
        ids = Dict{Tuple{Int,Int},Int}()
        c = [get!(ids, (c[n], (n - 1) % T == 0 ? c[n] : c[n-1]), length(ids) + 1) for n in eachindex(c)]
    end
    cnt = Dict{Int,Vector{Int}}()
    for n in eachindex(c); get!(cnt, c[n], [0, 0])[tgt[n]] += 1; end
    h = 0.0
    for (a, b) in values(cnt), x in (a, b)
        x > 0 && (h -= x * log(x / (a + b)))
    end
    h / length(c)
end

const TOL = 0.01   # "reached" = loss ≤ 1.01 × exact floor

"Train with k passes on length T until reached or cap. Returns (p, steps, reached, loss at start, acc at start, final loss)."
function train(p, T, k; cap, lr = 1e-2)
    toks, tgt = world(T)
    fl = floor_exact(T, k)
    θ = flat(p); o = Adam(length(θ))
    ℓ0, a0, _ = loss(p, toks, tgt, T, k)
    ℓ = ℓ0
    for s in 0:cap
        ℓ, a, g = loss(p, toks, tgt, T, k; grad = true)
        ℓ ≤ (1 + TOL) * fl && return p, s, true, ℓ0, a0, ℓ
        s == cap && break
        step!(o, θ, flat(g); lr)
        p = unflat(θ, p)
    end
    p, cap, false, ℓ0, a0, ℓ
end

function gradcheck()
    rng = MersenneTwister(1)
    p = init(rng); p = unflat(flat(p) .+ 0.1 .* randn(rng, nparams(p)), p)
    T, k = 4, 3
    toks, tgt = world(T)
    _, _, g = loss(p, toks, tgt, T, k; grad = true)
    θ = flat(p); gv = flat(g); worst = 0.0
    for i in eachindex(θ)
        h = 1e-5
        θp = copy(θ); θp[i] += h; θm = copy(θ); θm[i] -= h
        fd = (loss(unflat(θp, p), toks, tgt, T, k)[1] - loss(unflat(θm, p), toks, tgt, T, k)[1]) / 2h
        worst = max(worst, abs(fd - gv[i]) / max(1e-8, abs(fd) + abs(gv[i])))
    end
    worst, norm(gv)
end

function main()
    mode = ARGS[1]
    nseeds = parse(Int, ARGS[2]); cap = parse(Int, ARGS[3])
    KMAX = 7
    w, gn = gradcheck()
    @printf("parameters %d; gradient check (T=4, 3 passes, all parameters): worst relative error %.2e (|g| = %.3f)\n",
            nparams(init(MersenneTwister(0))), w, gn)
    println("exact floors (nats per position), k passes on length k+1: ",
            join([@sprintf("k=%d: %.5f", k, floor_exact(k + 1, k)) for k in 1:KMAX], "  "))
    work(k, T, steps) = k * T * steps              # block applications × positions × steps, per sequence
    workseq(k, T, steps) = work(k, T, steps) * 2^T # the same times the 2^T sequences of the full batch
    rows = []; finals = Dict()
    if mode == "staged"
        println("\n== staged: k passes on length k+1, k = 1..$KMAX (cap $cap steps per stage) ==")
        println("seed  k  T  floor  loss_at_birth  ratio_to_floor  acc_at_birth  steps  reached  final_loss")
        for seed in 1:nseeds
            p = init(MersenneTwister(seed)); tot = 0; totseq = 0; ok = true; steps_all = 0
            for k in 1:KMAX
                T = k + 1
                p, s, r, ℓ0, a0, ℓ = train(p, T, k; cap)
                fl = floor_exact(T, k)
                @printf("%4d %2d %2d  %.5f  %.5f  %7.3f  %.4f  %5d  %5s  %.5f\n", seed, k, T, fl, ℓ0, ℓ0 / fl, a0, s, r, ℓ)
                tot += work(k, T, s); totseq += workseq(k, T, s); steps_all += s; ok &= r
                flush(stdout)
            end
            push!(rows, (seed, ok, tot, totseq, steps_all)); finals[seed] = p
        end
    else
        println("\n== scratch: $KMAX passes on length $(KMAX+1) from random weights (cap $cap steps) ==")
        println("seed  steps  reached  final_loss  floor")
        for seed in 1:nseeds
            p, s, r, ℓ0, a0, ℓ = train(init(MersenneTwister(seed)), KMAX + 1, KMAX; cap)
            @printf("%4d  %5d  %5s  %.5f  %.5f\n", seed, s, r, ℓ, floor_exact(KMAX + 1, KMAX))
            push!(rows, (seed, r, work(KMAX, KMAX + 1, s), workseq(KMAX, KMAX + 1, s), s)); finals[seed] = p
            flush(stdout)
        end
    end
    println("\n== total work to reach 1.01 × floor ==")
    println("regime  seed  reached  steps  work=passes×positions×steps  work×sequences")
    for (seed, ok, w1, w2, s) in rows
        @printf("%-7s %4d  %5s  %6d  %12d  %14d\n", mode, seed, ok, s, w1, w2)
    end
    med(v) = isempty(v) ? NaN : sort(v)[(length(v)+1)÷2]
    good = [r for r in rows if r[2]]
    @printf("%-7s failed %d of %d; median over successes: work %.0f, work×sequences %.0f\n", mode,
            length(rows) - length(good), length(rows), med([r[3] for r in good]), med([r[4] for r in good]))
    println("\n== the final model run with k passes on length k+1: loss/floor (k ≤ 6 reverse map; k ≥ 8 never trained) ==")
    for seed in 1:nseeds
        p = finals[seed]
        print(rpad(mode, 8), seed, " ")
        for k in 1:9
            toks, tgt = world(k + 1)
            ℓ, a, _ = loss(p, toks, tgt, k + 1, k)
            @printf(" k=%d: %.3f", k, ℓ / floor_exact(k + 1, k))
        end
        println()
    end
end
(abspath(PROGRAM_FILE) == @__FILE__) && main()

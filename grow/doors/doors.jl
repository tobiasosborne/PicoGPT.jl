# grow/doors: a world refined by adding a hop; a looped block grown by one pass behind a gate.
# Run: OPENBLAS_NUM_THREADS=1 julia --project -t2 grow/doors/doors.jl [seeds]
using PicoGPT, LinearAlgebra, Random, Printf
BLAS.set_num_threads(1)

# Symbols: names 1:4, keys 5:8, doors 9:12, blank 13. A fact is ONE position whose column is
# Ek[:, a] + Ev[:, b] ("a → b"); the query is Ek[:, name] + Ev[:, blank]. No position code:
# the facts are a set. The answer is read at the query (last) position only.
const VS, BLANK = 13, 13
const NAMES, KEYS, DOORS = 1:4, 5:8, 9:12

# Each world returns a (T×B source ids), b (T×B target ids), y (B answers).
function coarse(rng, B)                       # name → door, query name: one lookup
    a = zeros(Int, 5, B); b = zeros(Int, 5, B); y = zeros(Int, B)
    for n in 1:B
        d = shuffle(rng, DOORS); o = randperm(rng, 4)
        a[1:4, n] = NAMES[o]; b[1:4, n] = d[o]
        q = rand(rng, 1:4); a[5, n] = NAMES[q]; b[5, n] = BLANK; y[n] = d[q]
    end
    a, b, y
end
function finefacts!(rng, a, b, n)
    k = shuffle(rng, KEYS); d = shuffle(rng, DOORS)       # name i → k[i], key k[i] → d[i]
    src = vcat(NAMES, k); dst = vcat(k, d); o = randperm(rng, 8)
    a[1:8, n] = src[o]; b[1:8, n] = dst[o]
    k, d
end
function fine(rng, B)                         # name → key, key → door, query name: two lookups
    a = zeros(Int, 9, B); b = zeros(Int, 9, B); y = zeros(Int, B)
    for n in 1:B
        k, d = finefacts!(rng, a, b, n)
        q = rand(rng, 1:4); a[9, n] = NAMES[q]; b[9, n] = BLANK; y[n] = d[q]
    end
    a, b, y
end
function hops(rng, B)                         # the fine context, but ONE lookup: name ↦ key or key ↦ door
    a = zeros(Int, 9, B); b = zeros(Int, 9, B); y = zeros(Int, B)
    for n in 1:B
        k, d = finefacts!(rng, a, b, n)
        q = rand(rng, 1:4); b[9, n] = BLANK
        if rand(rng, Bool); a[9, n] = NAMES[q]; y[n] = k[q] else a[9, n] = k[q]; y[n] = d[q] end
    end
    a, b, y
end

config(D, H, F) = Config(V = 2VS, D = D, H = H, L = 1, T = 16, F = F)   # E holds [Ek Ev]

"""
    ℓ, Z, back = looped(θ, a, b, y, c, k)

θ = [PicoGPT parameters (L = 1); g]. X₁ = block(X₀); with k = 2, X₂ = X₁ + g (block(X₁) − X₁),
the same weights. Logits over the 13 symbols at the last position. back() is ∇ℓ(θ), gate last.
"""
function looped(θ, a, b, y, c, k)
    g = θ[end]; p = carve(view(θ, 1:length(θ)-1), c)
    T, B = size(a); blk = p.blocks[1]
    X0 = p.E[:, vec(a)] + p.E[:, VS .+ vec(b)]
    X1, back1 = block(X0, blk, c.H, T)
    if k == 2
        Y, back2 = block(X1, blk, c.H, T)
        Δ = Y - X1
        X = X1 + g * Δ
    else
        X = X1
    end
    last = T:T:T*B
    Xf, back_Xf = layernorm(X[:, last], p.γf, p.βf)
    Zfull, back_Z = affine(Xf, p.Wout, p.bout)
    Z = Zfull[1:VS, :]
    ℓ, back_ℓ = crossentropy(Z, y)
    function back()
        Z̄ = zeros(size(Zfull)); Z̄[1:VS, :] = back_ℓ(1)
        X̄f, W̄out, b̄out = back_Z(Z̄)
        X̄l, γ̄f, β̄f = back_Xf(X̄f)
        X̄ = zeros(size(X)); X̄[:, last] = X̄l
        ḡ = 0.0
        if k == 2
            ḡ = dot(X̄, Δ)
            X̄1y, p̄2 = back2(g * X̄)
            X̄1 = (1 - g) * X̄ + X̄1y
            X̄0, p̄1 = back1(X̄1)
            p̄ = map(+, p̄1, p̄2)
        else
            X̄0, p̄ = back1(X̄)
        end
        Ē = zero(p.E)
        for (n, (s, t)) in enumerate(zip(a, b))
            @views Ē[:, s] .+= X̄0[:, n]; @views Ē[:, VS + t] .+= X̄0[:, n]
        end
        vcat(flatten(Ē, [p̄], γ̄f, β̄f, W̄out, b̄out), ḡ)
    end
    ℓ, Z, back
end

accuracy(Z, y) = sum(argmax(Z[:, n]) == y[n] for n in eachindex(y)) / length(y)
function evaluate(θ, data, c, k)
    ℓ, Z, _ = looped(θ, data..., c, k)
    ℓ, accuracy(Z, data[3])
end

"Train; return θ and the curve [(step, loss, acc)] on fixed test data, stopping at `stop` accuracy."
function train!(θ, world, c, k; rng, test, steps, B = 64, η = 3e-3, every = 25, stop = 0.995, trainmask = trues(length(θ)))
    opt = AdamW(θ); curve = Tuple{Int,Float64,Float64}[]
    for s in 1:steps
        ℓ, _, back = looped(θ, world(rng, B)..., c, k)
        θ̄ = back(); θ̄[.!trainmask] .= 0
        clip!(θ̄, 1.0)
        update!(opt, θ, θ̄, η)
        if s % every == 0
            lt, at = evaluate(θ, test, c, k); push!(curve, (s, lt, at))
            at ≥ stop && break
        end
    end
    θ, curve
end
reach(curve, level) = (i = findfirst(x -> x[3] ≥ level, curve); i === nothing ? missing : curve[i][1])
newθ(c, rng; g) = vcat(init(c; rng), g)

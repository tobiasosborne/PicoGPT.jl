# The synthetic two-level world, with exact entropies.
# Fine symbol v = 2(c-1)+b, class c ∈ 1:3, child b ∈ 1:2.
#   p(v'|v) = Pc(c'|v) · q(b'|c', v)
#   Pc(c'|v) ∝ P[c,c'] · exp(±δ·U[c,c'])      (sign + for b=1, − for b=2)   dial δ: the defect
#   q(b'=1|c',v) = σ(κ·(S[c',c] ± ε·R[c',c]))                               dials κ (peak), ε (child-dependence)
# With δ = 0 the class transition is P(c'|c): exactly lumpable for every ε, κ.
const P = [0.70 0.20 0.10; 0.10 0.60 0.30; 0.45 0.05 0.50]
const U = [1.0 -1.0 0.0; 0.0 1.0 -1.0; -1.0 0.0 1.0]
const S = [1.0 -1.0 0.5; -0.5 1.0 -1.0; 1.0 0.5 -1.0]
const R = [1.0 -1.0 1.0; 1.0 1.0 -1.0; -1.0 1.0 1.0]
cls(v) = (v + 1) ÷ 2
sgn(v) = isodd(v) ? 1.0 : -1.0
σ(x) = 1 / (1 + exp(-x))
xlogx(p) = p > 0 ? p * log(p) : 0.0
H(p) = -sum(xlogx, p)

"World(ε, κ, δ): M[v,v'] fine transition, Pc[v,c'] class transition, q1[v,c'] = q(b'=1|c',v), π stationary."
function world(; ε, κ, δ)
    Pc = zeros(6, 3); q1 = zeros(6, 3); M = zeros(6, 6)
    for v in 1:6
        c = cls(v)
        w = [P[c, c2] * exp(sgn(v) * δ * U[c, c2]) for c2 in 1:3]
        Pc[v, :] = w ./ sum(w)
        for c2 in 1:3
            q1[v, c2] = σ(κ * (S[c2, c] + sgn(v) * ε * R[c2, c]))
            M[v, 2c2-1] = Pc[v, c2] * q1[v, c2]
            M[v, 2c2]   = Pc[v, c2] * (1 - q1[v, c2])
        end
    end
    π = ones(6) / 6
    for _ in 1:10_000; π = M' * π; end
    (; ε, κ, δ, M, Pc, q1, π)
end

"Exact quantities (nats). T = window length used for the coarse floor of a model that sees t = 1..T classes."
function entropies(w; T = 8)
    (; M, Pc, q1, π) = w
    fine  = sum(π[v] * H(M[v, :]) for v in 1:6)                    # H(v'|v): fine floor (Markov)
    cfine = sum(π[v] * H(Pc[v, :]) for v in 1:6)                   # H(c'|v): class floor given fine history
    split = sum(π[v] * Pc[v, c2] * H([q1[v, c2], 1 - q1[v, c2]]) for v in 1:6, c2 in 1:3)  # H(v'|c',v)
    # H(c'|c): class floor given the last class only
    πc = [π[2c-1] + π[2c] for c in 1:3]
    Pcc = [(π[2c-1] * Pc[2c-1, c2] + π[2c] * Pc[2c, c2]) / πc[c] for c in 1:3, c2 in 1:3]
    c1 = sum(πc[c] * H(Pcc[c, :]) for c in 1:3)
    defect = c1 - cfine                                            # I(c'; v | c)
    # H(c_{t+1} | c_1..c_t) for t = 1..T, exact, by enumerating coarse histories with a forward filter
    Ht = zeros(T)
    function rec(α, t)                 # α[v] = P(c_1..c_t, v_t = v), unnormalised
        pa = sum(α); pa == 0 && return
        pred = Pc' * α                  # joint with next class
        Ht[t] += -sum(xlogx, pred) + xlogx(pa)
        t == T && return
        for c2 in 1:3
            β = zeros(6)
            for v2 in (2c2-1, 2c2); β[v2] = sum(α[v] * M[v, v2] for v in 1:6); end
            rec(β, t + 1)
        end
    end
    for c in 1:3
        α = zeros(6); α[2c-1] = π[2c-1]; α[2c] = π[2c]; rec(α, 1)
    end
    (; fine, cfine, split, c1, defect, Ht, coarse = sum(Ht) / T, chain = fine - (cfine + split))
end

function sample(w, n, rng)
    cum = cumsum(w.M; dims = 2); cπ = cumsum(w.π)
    x = Vector{Int}(undef, n)
    x[1] = min(searchsortedfirst(cπ, rand(rng)), 6)
    for i in 2:n
        x[i] = min(searchsortedfirst(view(cum, x[i-1], :), rand(rng)), 6)
    end
    x
end

"Windows: X[t,b], Y[t,b] fine; each window is an independent stationary sample of length T+1."
function batch(w, T, B, rng)
    X = zeros(Int, T, B); Y = zeros(Int, T, B)
    for b in 1:B
        s = sample(w, T + 1, rng); X[:, b] = s[1:T]; Y[:, b] = s[2:end]
    end
    X, Y
end

"Mean surprise of the true fine process on (X,Y), and of the true coarse process (forward filter) on the classes."
function truefloors(w, X, Y)
    T, B = size(X); f = 0.0; c = 0.0
    for b in 1:B
        α = zeros(6)
        for t in 1:T
            v = X[t, b]; f -= log(w.M[v, Y[t, b]])
            if t == 1
                α .= 0; cc = cls(v); α[2cc-1] = w.π[2cc-1]; α[2cc] = w.π[2cc]
            else
                cc = cls(v); β = zeros(6)
                for v2 in (2cc-1, 2cc); β[v2] = sum(α[u] * w.M[u, v2] for u in 1:6); end
                α .= β
            end
            α ./= sum(α)
            pred = w.Pc' * α
            c -= log(pred[cls(Y[t, b])])
        end
    end
    f / (T * B), c / (T * B)
end

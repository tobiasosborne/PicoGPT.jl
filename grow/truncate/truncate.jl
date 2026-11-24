# Coarse-graining back: truncate the residual stream of a trained PicoGPT.
#
# The stream of a model with L blocks is visited at K = 2L + 1 *sites*:
#
#   site 1        after the embedding
#   site 2ℓ       after the attention write of block ℓ
#   site 2ℓ + 1   after the MLP write of block ℓ
#
# A *cut* replaces the stream at every site k by
#
#   x ↦ Q (x − μₖ) + μₖ,        Q = T S,   T ∈ ℝ^{D×r},  S ∈ ℝ^{r×D},
#
# so the stream lives in the r-dimensional affine space μₖ + range(T), and the directions that
# Q discards are frozen at μₖ. With S T = 1 the map is a projector (orthogonal if S = Tᵀ).
# Everything else in the model is untouched: this file composes PicoGPT's own operations and
# adds only `site`. Its pullback is X̄ ↦ Qᵀ X̄.

using PicoGPT, LinearAlgebra, Random, Printf

"""A cut of the stream: x ↦ T S (x − μₖ) + μₖ at site k. `μ` has one column per site."""
struct Cut{R}
    T::Matrix{R}
    S::Matrix{R}
    μ::Matrix{R}
end
Base.size(q::Cut) = size(q.T)                       # (D, r)
nocut(c::Config, R = Float64) = nothing

site(X, ::Nothing, k) = (X, identity)
function site(X, q::Cut, k)
    μ = view(q.μ, :, k)
    Y = q.T * (q.S * (X .- μ)) .+ μ
    back(Ȳ) = q.S' * (q.T' * Ȳ)
    Y, back
end

# ── the two halves of a block, each a residual branch ───────────────────────────────

"""Y, back = attend(X, p, H, T): the attention write Wo mix(A, V) + bo, read through LN₁."""
function attend(X, p, H, T)
    Â, back_Â = layernorm(X, p.γ1, p.β1)
    Q, back_Q = affine(Â, p.Wq, p.bq)
    K, back_K = affine(Â, p.Wk, p.bk)
    V, back_V = affine(Â, p.Wv, p.bv)
    A, back_A = attention(Q, K, H, T)
    O, back_O = mix(A, V)
    Y, back_Y = affine(O, p.Wo, p.bo)
    function back(Ȳ)
        Ō, W̄o, b̄o = back_Y(Ȳ)
        Ā, V̄ = back_O(Ō)
        Q̄, K̄ = back_A(Ā)
        Âq, W̄q, b̄q = back_Q(Q̄)
        Âk, W̄k, b̄k = back_K(K̄)
        Âv, W̄v, b̄v = back_V(V̄)
        X̄, γ̄1, β̄1 = back_Â(Âq + Âk + Âv)
        X̄, (γ̄1, β̄1, W̄q, b̄q, W̄k, b̄k, W̄v, b̄v, W̄o, b̄o)
    end
    Y, back
end

"""Y, back = think(X, p): the MLP write W2 relu(W1 LN₂(X) + b1) + b2."""
function think(X, p)
    F̂, back_F̂ = layernorm(X, p.γ2, p.β2)
    U, back_U = affine(F̂, p.W1, p.b1)
    R, back_R = relu(U)
    Y, back_Y = affine(R, p.W2, p.b2)
    function back(Ȳ)
        R̄, W̄2, b̄2 = back_Y(Ȳ)
        F̄, W̄1, b̄1 = back_U(back_R(R̄))
        X̄, γ̄2, β̄2 = back_F̂(F̄)
        X̄, (γ̄2, β̄2, W̄1, b̄1, W̄2, b̄2)
    end
    Y, back
end

"""Z, back = readout(X, p): logits Wout LN_f(X) + bout."""
function readout(X, p)
    Xf, back_Xf = layernorm(X, p.γf, p.βf)
    Z, back_Z = affine(Xf, p.Wout, p.bout)
    function back(Z̄)
        X̄f, W̄out, b̄out = back_Z(Z̄)
        X̄, γ̄f, β̄f = back_Xf(X̄f)
        X̄, (γ̄f, β̄f, W̄out, b̄out)
    end
    Z, back
end

# ── the model with a cut stream ──────────────────────────────────────────────────────

"""
    Z, back, sites, pre = tgpt(θ, x, c, cut = nothing)

PicoGPT's `gpt` with the stream cut at every site. `sites[k]` is the stream at site k (after
the cut), `pre[k]` the value before the cut. `back(Z̄)` returns (θ̄, adj): the gradient laid
out like θ, and adj[k], the adjoint of sites[k], which is Jᵀ Z̄ for the Jacobian J of the
logits with respect to the stream at site k (all later paths included).
With `cut = nothing` this is `gpt` itself.
"""
function tgpt(θ, x, c::Config, cut = nothing)
    T = size(x, 1)
    p = carve(θ, c)
    K = 2c.L + 1
    sites, pre = Vector{Any}(undef, K), Vector{Any}(undef, K)
    cuts, branches = Vector{Any}(undef, K), Vector{Any}(undef, K)
    pre[1], back_embed = embed(x, p.E)
    sites[1], cuts[1] = site(pre[1], cut, 1)
    for k in 2:K
        b = p.blocks[k ÷ 2]
        Y, branches[k] = iseven(k) ? attend(sites[k-1], b, c.H, T) : think(sites[k-1], b)
        pre[k] = sites[k-1] + Y
        sites[k], cuts[k] = site(pre[k], cut, k)
    end
    Z, back_Z = readout(sites[K], p)

    function back(Z̄)
        adj = Vector{Any}(undef, K)
        halves = Vector{Any}(undef, K)
        X̄, p̄out = back_Z(Z̄)
        for k in K:-1:2
            adj[k] = X̄
            Ȳ = cuts[k](X̄)                        # adjoint of pre[k] = sites[k-1] + Y
            X̄, halves[k] = branches[k](Ȳ)
            X̄ = X̄ + Ȳ                             # the residual path
        end
        adj[1] = X̄
        Ē = back_embed(cuts[1](X̄))
        p̄blocks = [(halves[2ℓ]..., halves[2ℓ+1]...) for ℓ in 1:c.L]
        flatten(Ē, p̄blocks, p̄out...), adj
    end
    Z, back, sites, pre
end

"""ℓ, back = tloss(θ, x, y, c, cut): mean surprise of the cut model; back() = (∇θ ℓ, adj)."""
function tloss(θ, x, y, c::Config, cut = nothing)
    Z, back_Z, = tgpt(θ, x, c, cut)
    ℓ, back_ℓ = crossentropy(Z, y)
    back(ℓ̄ = 1) = back_Z(back_ℓ(ℓ̄))
    ℓ, back
end

"""Forward passes in parallel over groups of sequences; f(x_group) -> value, weights by size."""
function overgroups(f, x; group = 2)
    B = size(x, 2)
    PicoGPT.without_blas_threads() do
        tasks = @sync [Threads.@spawn (length(J), f(J)) for J in Iterators.partition(1:B, group)]
        fetch.(tasks)
    end
end

"""Mean loss of the cut model, in parallel over sequences."""
cutloss(θ, x, y, c, cut = nothing) =
    sum(n * v for (n, v) in overgroups(J -> first(tloss(θ, x[:, J], y[:, J], c, cut)), x)) / size(x, 2)

"""ℓ, θ̄ of the cut model, in parallel over sequences."""
function ∇tloss(θ, x, y, c, cut = nothing)
    B = size(x, 2)
    parts = overgroups(x) do J
        ℓ, back = tloss(θ, x[:, J], y[:, J], c, cut)
        w = eltype(θ)(length(J)) / B
        w * ℓ, first(back(w))
    end
    sum(first(v) for (_, v) in parts), sum(last(v) for (_, v) in parts)
end

"""Mean KL(p ‖ q) in nats between the columns of softmax(Zp) and softmax(Zq)."""
function meankl(Zp, Zq)
    lp = Zp .- mapslices(logsumexp, Zp; dims = 1)
    lq = Zq .- mapslices(logsumexp, Zq; dims = 1)
    sum(exp.(lp) .* (lp .- lq)) / size(Zp, 2)
end

# ── what the stream holds, and what the rest can see ────────────────────────────────

"""
    stats = streamstats(θ, x, c; metric, probes, rng)

Per site k, over all N token columns of the batch:

    μ[:, k]   the mean of the stream,
    C[k]      its covariance,                          E (x − μ)(x − μ)ᵀ,
    M[k]      the sensitivity Gram,                    E Σ_{n′} J_{n′n}ᵀ G_{n′} J_{n′n},

where J_{n′n} = ∂ z_{n′} / ∂ x_n is the Jacobian of the logits at token n′ with respect to the
stream at token n, site k, and G is a metric on logits:

    :logit    G = 1 − 11ᵀ/V                 centred Euclidean (a shift of all logits is no change)
    :fisher   G = diag(p) − ppᵀ             the Fisher metric: ½ δᵀGδ is the KL to second order

G = BᵀB, and one backward pass from Z̄ = Bᵀv per column gives Jᵀ Bᵀ v for every token at once.
`probes = :exact` runs v over a basis (position t′, symbol a), which is exact because
sequences do not interact; an integer runs that many Gaussian probes (Hutchinson's estimate).
"""
function streamstats(θ, x, c::Config; metric = :logit, probes = :exact, rng = Xoshiro(0), group = 2)
    T, B = size(x)
    R = eltype(θ)
    D, K, N = c.D, 2c.L + 1, length(x)
    function onegroup(J, vs)                         # vs: probe arrays V×T×|J|
        Z, back, sites, = tgpt(θ, x[:, J], c)
        P = softmax(Z; dims = 1)
        Ms = [zeros(R, D, D) for _ in 1:K]      # (not `M`: that name belongs to the enclosing scope)
        for v in vs
            v = reshape(v, c.V, :)
            Z̄ = metric == :logit ? v .- sum(v, dims = 1) ./ c.V :
                (w = sqrt.(P) .* v; w .- P .* sum(w, dims = 1))
            _, adj = back(Z̄)
            for k in 1:K
                mul!(Ms[k], adj[k], adj[k]', true, true)
            end
        end
        s1 = [vec(sum(X, dims = 2)) for X in sites]
        s2 = [X * X' for X in sites]
        (; s1, s2, M = Ms)
    end
    groups = collect(Iterators.partition(1:B, group))
    function probesfor(J, g)
        if probes === :exact
            [(v = zeros(R, c.V, T, length(J)); v[a, t, :] .= 1; v) for a in 1:c.V, t in 1:T]
        else
            r = Xoshiro(hash((g, rand(copy(rng), UInt))))
            [randn(r, R, c.V, T, length(J)) for _ in 1:probes]
        end
    end
    parts = PicoGPT.without_blas_threads() do
        tasks = @sync [Threads.@spawn onegroup(J, probesfor(J, g)) for (g, J) in enumerate(groups)]
        fetch.(tasks)
    end
    nprobe = probes === :exact ? 1 : probes
    μ = reduce(hcat, [sum(p.s1[k] for p in parts) ./ N for k in 1:K])
    C = [Symmetric(Float64.(sum(p.s2[k] for p in parts) ./ N .- μ[:, k] * μ[:, k]')) for k in 1:K]
    M = [Symmetric(Float64.(sum(p.M[k] for p in parts) ./ (N * nprobe))) for k in 1:K]
    (; μ = Float64.(μ), C, M)
end

pooled(As) = Symmetric(sum(Matrix, As) ./ length(As))

"""Symmetric square root and pseudo-inverse square root of a positive semidefinite matrix."""
function sqrts(C; rtol = 1e-10)
    λ, U = eigen(Symmetric(Matrix(C)))
    keep = λ .> rtol * maximum(λ)
    U[:, keep] * Diagonal(sqrt.(λ[keep])) * U[:, keep]', U[:, keep] * Diagonal(1 ./ sqrt.(λ[keep])) * U[:, keep]'
end

topvecs(A, r) = (e = eigen(Symmetric(Matrix(A))); e.vectors[:, sortperm(e.values, rev = true)[1:r]])
spectrum(A) = sort(eigvals(Symmetric(Matrix(A))), rev = true)

"""The Hankel values √eig(C M): the singular values of G^{1/2} J C^{1/2}, pooled."""
hankel(C, M) = (L = first(sqrts(C)); sqrt.(max.(spectrum(L * M * L), 0)))

"""
    cut = choose(rule, r, stats; rng)

A rank-r cut of the stream, one projector shared by all sites, μₖ the per-site means.

    :pca       top eigenvectors of the pooled covariance C                    (orthogonal)
    :pcaperp      the same after removing the direction 1, which no LayerNorm reads
    :jac       top eigenvectors of the pooled sensitivity Gram M              (orthogonal)
    :balanced  top eigenvectors of C^{1/2} M C^{1/2}, T = C^{1/2} V, S = Vᵀ C^{-1/2}   (oblique)
    :random    a Haar-random r-dimensional subspace                           (orthogonal)

`:balanced` minimises E |G^{1/2} J (1 − Q)(x − μ)|² over rank-r Q when J is one fixed matrix and
x has covariance C (reduced-rank regression; "balanced truncation" in control theory).
"""
function choose(rule, r, stats; rng = Xoshiro(0), R = Float64)
    D = size(stats.μ, 1)
    C, M = pooled(stats.C), pooled(stats.M)
    if rule == :pca
        U = topvecs(C, r); T, S = U, Matrix(U')
    elseif rule == :pcaperp
        Π = I - fill(1 / D, D, D)
        U = topvecs(Π * C * Π, r); T, S = U, Matrix(U')
    elseif rule == :jac
        U = topvecs(M, r); T, S = U, Matrix(U')
    elseif rule == :balanced
        L, Linv = sqrts(C)
        V = topvecs(L * M * L, r)
        T, S = L * V, V' * Linv
    elseif rule == :random
        U = Matrix(qr(randn(rng, D, r)).Q)[:, 1:r]; T, S = U, Matrix(U')
    else
        error("unknown rule $rule")
    end
    Cut{R}(T, S, stats.μ)
end

# ── training the cut model ───────────────────────────────────────────────────────────

"""
    θ′, losses = retrain(θ, x, y, c, cut; steps, η)

A short retraining of the cut model: AdamW on all of θ with the cut held fixed, full batch,
warm-up and cosine. (θ over-parametrises the cut model: only the products of the weights with
T and S matter.) losses[k+1] is the loss after k updates.
"""
function retrain(θ, x, y, c, cut; steps = 50, η = 1e-2, ηmin = η / 10, warmup = 5, λ = 0.01, clip = 1.0,
                 batch = nothing, rng = Xoshiro(0))
    θ = copy(θ)
    opt = AdamW(θ; λ = decay(c, eltype(θ)(λ)))
    losses = Float64[]
    for t in 1:steps
        J = batch === nothing ? (1:size(x, 2)) : rand(rng, 1:size(x, 2), batch)
        ℓ, θ̄ = ∇tloss(θ, x[:, J], y[:, J], c, cut)
        push!(losses, ℓ)
        update!(opt, θ, clip!(θ̄, clip), warmup_cosine(t; η, warmup, steps, ηmin))
    end
    batch === nothing && push!(losses, cutloss(θ, x, y, c, cut))
    θ, losses
end

# ── the cut model, written in its own r coordinates ─────────────────────────────────
#
# With z = S (x − μₖ) and x = T z + μₖ, a LayerNorm that reads the lifted stream is
#
#   LN(T z + μ) = γ ⊙ (ΠT z + Πμ)/σ(z) + β,     σ(z)² = (zᵀG z + 2hᵀz + c₀)/D + ε,
#   G = TᵀΠT,  h = TᵀΠμ,  c₀ = μᵀΠμ,            Π = 1 − 11ᵀ/D,
#
# so a reader W LN(x) + b becomes  (W Γ ΠT) z/σ + (W Γ Πμ)/σ + (W β + b):  a linear map of
# (z, 1)/σ(z) plus a bias, where σ is the square root of a general quadratic form in z.
# The model below uses only r-column weights. It is the same function as `tgpt` with the cut.

"""The reader W LN(·; γ, β) + b in cut coordinates at a site with mean μ."""
function smallreader(W, b, γ, β, q::Cut, μ)
    D = size(q.T, 1)
    Π = I - fill(1 / D, D, D)
    (A = W * Diagonal(γ) * Π * q.T, a = W * (γ .* (Π * μ)), b = W * β + b,
     G = q.T' * Π * q.T, h = q.T' * (Π * μ), c0 = dot(μ, Π * μ), D)
end
function reads(read, Z)
    σ = sqrt.((vec(sum(Z .* (read.G * Z), dims = 1)) .+ 2 .* (Z' * read.h) .+ read.c0) ./ read.D .+ 1e-5)'
    (read.A * Z .+ read.a) ./ σ .+ read.b
end

"""
    Z = smallgpt(θ, x, c, cut)

The cut model computed entirely in r coordinates (stream z ∈ ℝ^{r×N}); returns the logits and
the streams zₖ. Forward only: it exists to test the formulas above against `tgpt`.
"""
function smallgpt(θ, x, c::Config, q::Cut)
    T = size(x, 1)
    p = carve(θ, c)
    μ(k) = q.μ[:, k]
    E′ = q.S * p.E                                        # r × V
    P′ = q.S * (positions(c.D, T) .- μ(1))                # r × T: the position code, minus the mean
    z = E′[:, vec(x)] .+ repeat(P′, 1, size(x, 2))
    zs = [z]
    for k in 2:2c.L+1
        b = p.blocks[k ÷ 2]
        shift = q.S * (μ(k-1) - μ(k))                      # the frozen directions move with the mean
        if iseven(k)
            Q = reads(smallreader(b.Wq, b.bq, b.γ1, b.β1, q, μ(k-1)), z)
            K = reads(smallreader(b.Wk, b.bk, b.γ1, b.β1, q, μ(k-1)), z)
            V = reads(smallreader(b.Wv, b.bv, b.γ1, b.β1, q, μ(k-1)), z)
            O = first(mix(first(attention(Q, K, c.H, T)), V))
            z = z .+ (q.S * b.Wo) * O .+ q.S * b.bo .+ shift
        else
            U = reads(smallreader(b.W1, b.b1, b.γ2, b.β2, q, μ(k-1)), z)
            z = z .+ (q.S * b.W2) * max.(U, 0) .+ q.S * b.b2 .+ shift
        end
        push!(zs, z)
    end
    reads(smallreader(p.Wout, p.bout, p.γf, p.βf, q, μ(2c.L+1)), z), zs
end

"""Parameters of the cut model in r coordinates (weights only; the fixed tables listed apart)."""
function smallcount(c::Config, r)
    (; V, D, F, L) = c
    learned = r * V + L * (3 * (D * r + D + D) + (r * D + r) + (F * r + F + F) + (r * F + r)) + (V * r + V + V)
    fixed = r * c.T + (2L + 1) * (r * (r + 1) ÷ 2 + r + 1)      # position table; the LayerNorm forms G, h, c₀
    (; learned, fixed)
end

# ── the intertwiner defect ───────────────────────────────────────────────────────────

"""
    defects(θ₂, θ₁, x, c, cut)

Is the cut model (weights θ₁, cut `cut`) a coarse-graining of the full model θ₂, R∘F₂ = F₁∘R,
with R the restriction x ↦ S (x − μₖ)? Step by step along the stream of the full model, on its
own activations Xₖ:

    local[k]   |Rₖ F₂ₖ(X_{k−1}) − F₁ₖ(R_{k−1} X_{k−1})| / |Rₖ Xₖ|     one step from the true state
    drift[k]   |Rₖ Xₖ − zₖ| / |Rₖ Xₖ|                                 the cut model run from the start

with Frobenius norms; F₁ₖ(z) = Rₖ(step of θ₁ from the lift T z + μ_{k−1}). The readout is
compared by the mean KL between the two predictions (`klocal`, `kdrift`).
"""
function defects(θ₂, θ₁, x, c::Config, q::Cut)
    T = size(x, 1)
    K = 2c.L + 1
    Z₂, _, X, = tgpt(θ₂, x, c)                             # the full model, untouched
    Z₁, _, Y, = tgpt(θ₁, x, c, q)                          # the cut model, from the start
    p₁ = carve(θ₁, c)
    R(X, k) = q.S * (X .- q.μ[:, k])
    lift(z, k) = q.T * z .+ q.μ[:, k]
    loc, drift = zeros(K), zeros(K)
    for k in 1:K
        scale = norm(R(X[k], k))
        drift[k] = norm(R(X[k], k) - R(Y[k], k)) / scale
        if k == 1
            loc[k] = norm(R(X[1], 1) - R(first(embed(x, p₁.E)), 1)) / scale
        else
            b = p₁.blocks[k ÷ 2]
            X₀ = lift(R(X[k-1], k - 1), k - 1)               # R then lift: the coarse state
            step = iseven(k) ? first(attend(X₀, b, c.H, T)) : first(think(X₀, b))
            loc[k] = norm(R(X[k], k) - R(X₀ + step, k)) / scale
        end
    end
    klocal = meankl(Z₂, first(readout(lift(R(X[K], K), K), p₁)))
    (; loc, drift, klocal, kdrift = meankl(Z₂, Z₁))
end

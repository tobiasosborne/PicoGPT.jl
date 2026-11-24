# Shared by the scripts of grow/tower/. Run everything from the repository root, e.g.
#
#     julia --project=. -t 3 grow/tower/floors.jl
#
# PicoGPT (src/) is used as it stands; nothing in it is changed.

using PicoGPT, LinearAlgebra, Random, Printf

# ── the two-level world of the specimen ──────────────────────────────────────────────
const CORPUS = "to be or not to be"
const ABC = alphabet(CORPUS)                       # ' ', b, e, n, o, r, t
const CLASSNAMES = ["vowel", "consonant", "space"]
classof(ch::Char) = ch in "eo" ? 1 : ch == ' ' ? 3 : 2
const CLS = classof.(ABC)                          # class of each fine symbol: π on symbols

# ── the split loss ───────────────────────────────────────────────────────────────────
"""
    ℓc, ℓr, back = splitloss(Z, y, cls)

The surprise −log softmax(z)[y] of a fine target y, split by the chain rule into

    class term        ℓc = mean −log P(v | ·),   P(v) = Σ_{a ∈ v} softmax(z)_a,   v = cls[y],
    refinement term   ℓr = mean −log q(y | v),   q(a | v) = exp z_a / Σ_{a′ ∈ v} exp z_a′,

so that ℓc + ℓr is the ordinary cross-entropy of Z against y. Z ∈ ℝ^{V×N}.

Pullback (c̄, r̄) ↦ Z̄ of c̄ ℓc + r̄ ℓr. With p = softmax(z) and q̃ = q(· | v) (zero outside v):
∂ℓc/∂z = p − q̃ and ∂ℓr/∂z = q̃ − e_y, per column, over N. Their sum is p − e_y.
"""
function splitloss(Z, y, cls)
    V, N = size(Z)
    length(y) == N || throw(DimensionMismatch("one target per column"))
    P = softmax(Z; dims = 1)
    Q = zero(Z)                                    # q̃: the within-class Gibbs distribution
    ℓc = ℓr = zero(eltype(Z))
    for n in 1:N
        v = cls[y[n]]
        z = view(Z, :, n)
        inclass = [a for a in 1:V if cls[a] == v]
        lse_all, lse_v = logsumexp(z), logsumexp(z[inclass])
        ℓc += lse_all - lse_v
        ℓr += lse_v - z[y[n]]
        Q[inclass, n] .= exp.(z[inclass] .- lse_v)
    end
    function back(c̄ = 1, r̄ = 1)
        Z̄ = c̄ .* (P .- Q) .+ r̄ .* Q
        for n in 1:N
            Z̄[y[n], n] -= r̄
        end
        Z̄ ./= N
    end
    ℓc / N, ℓr / N, back
end

"""Class probabilities P[v, n] = Σ_{a ∈ v} softmax(Z[:, n])_a: the push-forward of the prediction through π."""
function classprobs(Z, cls)
    P = softmax(Z; dims = 1)
    [sum(P[a, n] for a in axes(P, 1) if cls[a] == v) for v in 1:maximum(cls), n in axes(P, 2)]
end

# ── the vocabulary lift ──────────────────────────────────────────────────────────────
"""
    θ2, c2 = lift(θ1, c1, cls)

Lift a model on K classes to one on V = length(cls) fine symbols that changes nothing at
birth: every fine symbol a inherits the embedding column and the readout row of its class,
and the readout bias of a is the class bias minus log |class|. Then the fine model reads a
fine history as its coarse image, and softmax(z2)_a = softmax(z1)_{cls a} / |cls a|:
p(v, b | h) = p(v | πh) · 1/|v|. Summing over the class returns the parent.
"""
function lift(θ1, c1::Config, cls)
    c1.V == maximum(cls) || throw(ArgumentError("the parent must have one symbol per class"))
    c2 = Config(V = length(cls), D = c1.D, H = c1.H, L = c1.L, T = c1.T, F = c1.F)
    θ2 = zeros(eltype(θ1), nparams(c2))
    p1, p2 = carve(θ1, c1), carve(θ2, c2)
    size_of = [count(==(v), cls) for v in 1:c1.V]
    p2.E .= p1.E[:, cls]
    for (b2, b1) in zip(p2.blocks, p1.blocks), k in keys(b1)
        b2[k] .= b1[k]
    end
    p2.γf .= p1.γf
    p2.βf .= p1.βf
    p2.Wout .= p1.Wout[cls, :]
    p2.bout .= p1.bout[cls] .- log.(size_of[cls])
    θ2, c2
end

# ── training ─────────────────────────────────────────────────────────────────────────
const RECIPE = (η = 2e-2, ηmin = 2e-3, warmup = 20, steps = 200,
                β1 = 0.9, β2 = 0.95, ε = 1e-8, λ = 0.01, clip = 1.0)    # the specimen's recipe

"""
    losses, θ = train(θ₀, c, lossgrad; recipe = RECIPE, mask = nothing, record = ℓ -> ℓ)

AdamW with warmup and cosine, as for the specimen. `lossgrad(θ)` returns (value, θ̄), where
value may be any record of the loss (a number or a tuple). `mask` (a Bool vector, true =
trainable) freezes the other coordinates: their gradient and weight decay are set to zero,
so they do not move. losses[k+1] is the value after k updates.
"""
function train(θ₀, c::Config, lossgrad; recipe = RECIPE, mask = nothing)
    θ = copy(θ₀)
    λ = decay(c, recipe.λ)
    mask === nothing || (λ .*= mask)
    opt = AdamW(θ; recipe.β1, recipe.β2, recipe.ε, λ)
    values = []
    for t in 1:recipe.steps
        ℓ, θ̄ = lossgrad(θ)
        push!(values, ℓ)
        mask === nothing || (θ̄ .*= mask)
        η = warmup_cosine(t; recipe.η, recipe.warmup, recipe.steps, recipe.ηmin)
        update!(opt, θ, clip!(θ̄, recipe.clip), η)
    end
    push!(values, first(lossgrad(θ)))
    values, θ
end

"""The first number of updates after which the loss is ≤ target (nothing if never)."""
steps_to(losses, target) = (k = findfirst(≤(target), losses); k === nothing ? nothing : k - 1)

show_steps(k) = k === nothing ? "never" : string(k)

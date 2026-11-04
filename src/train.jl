# Chapter 8 · Learning
#
# Training is gradient descent on the mean surprise ℓ(θ), with refinements: a minibatch
# gives a noisy but cheap estimate of the gradient; AdamW rescales the step coordinate by
# coordinate; clipping caps the gradient norm; and the step size follows a schedule.
# All of it acts on plain vectors in ℝᵖ.

"""
    x, y = windows(data, T, starts = 1:length(data) - T)

A batch of windows of the token sequence `data`, one per start i: column b of x is
data[i : i+T-1] and column b of y is data[i+1 : i+T], the same window shifted by one.
At every position, the target is the next token.
"""
windows(data, T, starts = 1:length(data) - T) =
    ([data[i + t - 1] for t in 1:T, i in starts], [data[i + t] for t in 1:T, i in starts])

"""
    clip!(θ̄, r)

Global-norm clipping: if |θ̄| > r ≥ 0, scale θ̄ back onto the sphere of radius r.
"""
function clip!(θ̄, r)
    r ≥ 0 || throw(ArgumentError("the clipping radius must be nonnegative"))
    r = eltype(θ̄)(r)
    n = norm(θ̄)
    n > r && (θ̄ .*= r / n)
    θ̄
end

"""
    Descent()

Plain gradient descent, θ ← θ − η θ̄: an Euler step for the gradient flow dθ/dt = −∇ℓ(θ).
"""
struct Descent end
update!(::Descent, θ, θ̄, η) = (θ .-= eltype(θ)(η) .* θ̄)

"""
    AdamW(θ; β1 = 0.9, β2 = 0.95, ε = 1e-8, λ = 0)

The AdamW optimiser (Kingma & Ba 2015; Loshchilov & Hutter 2019). It keeps running
averages of the gradient and of its square, coordinate by coordinate,

    m ← β1 m + (1 − β1) θ̄,        v ← β2 v + (1 − β2) θ̄²,

and at step t moves  θ ← θ − η (m̂ / (√v̂ + ε) + λ θ),  with m̂ = m/(1 − β1ᵗ) and
v̂ = v/(1 − β2ᵗ) undoing the bias of starting the averages at zero. Dividing by √v̂ is a
diagonal preconditioner. For a constant gradient and negligible ε, each nonzero
coordinate's adaptive step has magnitude about η; changing gradients need not obey
this sign-step limit. λθ is weight decay, applied directly (not through θ̄); λ may be
a vector. Both terms use the old θ. State and arithmetic use θ's element type.
"""
mutable struct AdamW{V<:AbstractVector, R<:AbstractFloat, Λ}
    m::V
    v::V
    t::Int
    β1::R
    β2::R
    ε::R
    λ::Λ
end

function AdamW(θ; β1 = 0.9, β2 = 0.95, ε = 1e-8, λ = 0)
    R = eltype(θ)
    β1, β2, ε = R(β1), R(β2), R(ε)
    0 ≤ β1 < 1 && 0 ≤ β2 < 1 && ε > 0 || throw(ArgumentError("AdamW needs 0 ≤ β1, β2 < 1 and ε > 0"))
    AdamW(zero(θ), zero(θ), 0, β1, β2, ε, R.(λ))
end

function update!(o::AdamW, θ, θ̄, η)
    (; m, v, β1, β2, ε, λ) = o
    η = eltype(θ)(η)
    t = o.t += 1
    c1, c2 = 1 - β1^t, 1 - β2^t                           # bias corrections
    @. m = β1 * m + (1 - β1) * θ̄
    @. v = β2 * v + (1 - β2) * θ̄^2
    @. θ -= η * ((m / c1) / (√(v / c2) + ε) + λ * θ)
end

"""
    λ = decay(c, λ₀)

Weight decay for each coordinate of θ: λ₀ on the entries of the matrices, and 0 on the
gains γ, offsets β and biases b, which have no reason to be small.
"""
function decay(c::Config, λ₀)
    λ = zeros(typeof(float(λ₀)), nparams(c))
    for x in leaves(carve(λ, c))
        ndims(x) == 2 && fill!(x, λ₀)
    end
    λ
end

"""
    warmup_cosine(t; η, warmup, steps, ηmin = η/10)

The step size at step t: a linear ramp from 0 up to η over the first `warmup` steps
(while Adam's averages are still unreliable), then half a cosine down to ηmin at `steps`.
Require 0 ≤ warmup < steps and 0 ≤ t ≤ steps. With warmup = 0 the cosine starts
at η at t = 0; otherwise the step size at t = 0 is zero. Arithmetic uses η's float type.
"""
function warmup_cosine(t; η, warmup, steps, ηmin = η / 10)
    0 ≤ warmup < steps || throw(ArgumentError("the schedule needs 0 ≤ warmup < steps"))
    0 ≤ t ≤ steps || throw(ArgumentError("the schedule step must lie in 0:steps"))
    η = float(η)
    ηmin = oftype(η, ηmin)
    t == 0 && return warmup == 0 ? η : zero(η)
    t ≤ warmup && return η * t / warmup
    u = oftype(η, t - warmup) / (steps - warmup)
    ηmin + (η - ηmin) * (1 + cos(π * u)) / 2
end

"""
    ℓ, θ̄ = ∇loss(θ, x, y, c)

The loss on the batch and its gradient, as `loss` gives them, but computed in parallel.
The batch loss is the mean of the per-sequence losses, so its gradient is the mean of
theirs: the sequences are dealt into groups, and each group makes its own forward and
backward pass, on a thread of its own.
"""
function ∇loss(θ, x, y, c::Config)
    size(x) == size(y) || throw(DimensionMismatch("inputs and targets must have the same shape"))
    isempty(x) && throw(ArgumentError("the mean loss needs a nonempty batch"))
    B = size(x, 2)
    parts = without_blas_threads() do
        tasks = @sync [Threads.@spawn begin
                    ℓ, back = loss(θ, x[:, J], y[:, J], c)
                    w = eltype(θ)(length(J)) / B       # this group's share of the batch
                    w * ℓ, back(w)                    # ∇(wℓ) = back(w): pullbacks are linear
                end
                for J in Iterators.partition(1:B, 2)]
        fetch.(tasks)                                 # @sync joins every task even on failure
    end
    θ̄ = last(first(parts))
    for (_, g) in Iterators.drop(parts, 1)
        θ̄ .+= g                                       # fixed order; no full-vector sum temporaries
    end
    sum(first, parts), θ̄
end

"""
    ℓ = step!(θ, opt, x, y, c; η, clip = 1)

One step of training on the batch (x, y): the loss and its gradient, the gradient clipped
to norm ≤ clip, and one update of θ by the optimiser. Returns the loss before the update.
"""
function step!(θ, opt, x, y, c::Config; η, clip = 1)
    ℓ, θ̄ = ∇loss(θ, x, y, c)
    update!(opt, θ, clip!(θ̄, clip), η)
    ℓ
end

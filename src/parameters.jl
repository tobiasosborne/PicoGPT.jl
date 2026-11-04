# Chapter 6 · The model is a point in ℝᵖ
#
# All the parameters live in one vector θ ∈ ℝᵖ, in a fixed order. The named parameters
# (E, Wq, γ1, …) are windows onto θ: reshaped views, not copies, so that writing to
# p.blocks[1].Wq writes to θ. The gradient θ̄ is a vector of the same length, in the same
# order. Then optimisers, clipping and gradient checks are statements about vectors.

"""
    Config(; V, D = 128, H = 4, L = 4, T = 256, F = 4D)

The shape of a model: V symbols, a residual stream of width D, H heads of width d = D/H,
L blocks, context length T, and MLP hidden width F.
"""
struct Config
    V::Int
    D::Int
    H::Int
    L::Int
    T::Int
    F::Int
end

function Config(; V, D = 128, H = 4, L = 4, T = 256, F = 4D)
    iseven(D) && D % H == 0 || throw(ArgumentError("D = $D must be even and a multiple of H = $H"))
    Config(V, D, H, L, T, F)
end

"""
    nparams(c) = 2VD + V + 2D + L(4D² + 2DF + 9D + F)

The number of parameters p. Embedding and readout give 2VD + V, the final LayerNorm 2D;
each block has four D×D attention matrices, two D×F MLP matrices, and 9D + F gains and biases.
"""
function nparams(c::Config)
    (; V, D, F, L) = c
    2V*D + V + 2D + L * (4D^2 + 2D*F + 9D + F)
end

"""
    p = carve(θ, c)

Cut θ into the named parameters, in their fixed order:

    E;   per block: γ1 β1 Wq bq Wk bk Wv bv Wo bo γ2 β2 W1 b1 W2 b2;   γf βf Wout bout.

Each piece is a view onto θ; matrices are stored column-major. `p.E` is θ[1:DV] seen as a
D×V matrix, and so on. `flatten` is the inverse.
"""
function carve(θ, c::Config)
    (; V, D, F, L) = c
    length(θ) == nparams(c) || throw(DimensionMismatch("θ has length $(length(θ)); $c needs $(nparams(c))"))
    used = 0
    function next(dims...)                                # the next prod(dims) entries of θ
        piece = reshape(view(θ, used+1:used+prod(dims)), dims)
        used += prod(dims)
        piece
    end
    E = next(D, V)
    blocks = [(γ1 = next(D),    β1 = next(D),
               Wq = next(D, D), bq = next(D),
               Wk = next(D, D), bk = next(D),
               Wv = next(D, D), bv = next(D),
               Wo = next(D, D), bo = next(D),
               γ2 = next(D),    β2 = next(D),
               W1 = next(F, D), b1 = next(F),
               W2 = next(D, F), b2 = next(D)) for ℓ in 1:L]
    (; E, blocks, γf = next(D), βf = next(D), Wout = next(V, D), bout = next(V))
end

"""
    leaves(p) -> Vector

The arrays inside a nest of named tuples, tuples and vectors, in order.
"""
leaves(x::AbstractArray{<:Number}) = [x]
leaves(xs) = mapreduce(leaves, vcat, values(xs))

"""
    flatten(pieces...) -> Vector

All the entries of all the arrays in `pieces` (nested as you like), concatenated in order:
the inverse of `carve`, flatten(carve(θ, c)) == θ. This is how the pullback assembles θ̄.
"""
flatten(pieces...) = reduce(vcat, vec.(leaves(pieces)))

"""
    μ, σ = initial(name, c)

How the parameter `name` starts: every entry is drawn from N(μ, σ²); σ = 0 means the
constant μ. The variances are chosen so that, at the start, every column of the stream
and every logit has a sensible size:

    E               N(0, 1)               as loud as the positional code
    Wq, Wk, Wv, W1  N(0, 1/D)             each output entry has variance ≈ 1 (fan-in D)
    Wo, W2          N(0, 1/(2L⋅fan-in))   the 2L updates of the stream add up to ≈ 1
    Wout            N(0, 1/D²)            logits ≈ 0: the first guess is nearly uniform
    γ  1            β, b  0
"""
function initial(name::Symbol, c::Config)
    (; D, F, L) = c
    name == :E                   ? (0.0, 1.0) :
    name in (:Wq, :Wk, :Wv, :W1) ? (0.0, 1 / √D) :
    name == :Wo                  ? (0.0, 1 / √(2L * D)) :
    name == :W2                  ? (0.0, 1 / √(2L * F)) :
    name == :Wout                ? (0.0, 1 / D) :
    name in (:γ1, :γ2, :γf)      ? (1.0, 0.0) :
                                   (0.0, 0.0)              # β1 β2 βf and the biases
end

"""
    θ = init(c, R = Float64; rng = Random.default_rng())

A random starting point θ ∈ Rᵖ, with entries of type R, each parameter drawn as
`initial` says.
"""
function init(c::Config, R::Type{<:AbstractFloat} = Float64; rng = Random.default_rng())
    θ = zeros(R, nparams(c))
    p = carve(θ, c)
    draw!(name, x) = ((μ, σ) = initial(name, c); x .= μ .+ σ .* randn(rng, R, size(x)))
    draw!(:E, p.E)
    for b in p.blocks, (name, x) in pairs(b)
        draw!(name, x)
    end
    foreach(name -> draw!(name, p[name]), (:γf, :βf, :Wout, :bout))
    θ
end

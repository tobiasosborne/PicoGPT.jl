"""
One PicoGPT block, applied repeatedly. As in PicoGPT, every model operation returns
its value and its pullback, with one token per column and parameters in one vector.
"""
module Loop

using PicoGPT
using PicoGPT: record!
using LinearAlgebra: mul!
using Random: randn!

export loopgpt, looploss, predictions, registergpt, registerloss, register_init, register_nparams

include("adjacent.jl")
include("relative.jl")
include("register.jl")

"""Token id of the largest logit in each column, in sequence order."""
predictions(Z, T) = reshape(map(argmax, eachcol(Z)), T, :)

function readout(X, p)
    Xf, back_norm = layernorm(X, p.γf, p.βf)
    Z, back_affine = affine(Xf, p.Wout, p.bout)
    function back(Z̄)
        X̄f, W̄out, b̄out = back_affine(Z̄)
        X̄, γ̄f, β̄f = back_norm(X̄f)
        X̄, (γ̄f, β̄f, W̄out, b̄out)
    end
    Z, back
end

"""
    Z, back = loopgpt(θ, x, c; K = c.L, tied = true, injection = :none,
                      readouts = :last, trace = nothing, attention_span = :full,
                      position_code = true, stabilize = false, residual_scale = 1,
                      previous_bias = 0)

With tied weights, c.L = 1 and Xₖ = Bθ(Xₖ₋₁). Injection :after adds X₀ to the
block output, while :before adds X₀ to the block input (including on pass one).
With tied = false, the first K of c.L independent blocks form the untied control.

Apply the same LayerNorm and affine readout at the last pass or at :all passes.
Z (and its adjoint) is a matrix for :last, and a vector of matrices for :all.
An optional trace receives the embedding, then one record per pass. Its
attention axes are [query, key, head, sequence]. Test lengths may exceed c.T.

The locality experiment sets attention_span = :adjacent, with a separate trace
field for attention to the left boundary. position_code = false drops the
fixed sinusoidal code. stabilize = true inserts a parameter-free LayerNorm
after each transition (including its injection) to bound the stream.
The :refresh update keeps X₀ + s(B(X) − X), where s = residual_scale.
Other updates scale the block residual by s before adding it to the input.
A positive previous_bias adds a fixed logit to the immediately preceding key.
With :previous attention, each head reads exactly the predecessor, with a
fixed zero stream column to the left of each sequence.
"""
function loopgpt(θ, x, c::Config; K::Int = c.L, tied::Bool = true, injection = :none,
                 readouts = :last, trace = nothing, attention_span = :full,
                 position_code::Bool = true, stabilize::Bool = false,
                 residual_scale::Real = 1, previous_bias::Real = 0)
    K ≥ 1 && (tied ? c.L == 1 : c.L ≥ K) ||
        throw(ArgumentError("use L=1 for tied passes or L≥K for untied passes, with K ≥ 1"))
    injection in (:none, :before, :after, :refresh) || throw(ArgumentError("unknown injection mode"))
    readouts in (:last, :all) || throw(ArgumentError("unknown readout mode"))
    attention_span in (:full, :adjacent, :previous) || throw(ArgumentError("unknown attention span"))
    s = eltype(θ)(residual_scale)
    T = size(x, 1)
    p = carve(θ, c)
    X₀, back_embed = embed(x, p.E)
    if !position_code
        reshape(X₀, c.D, T, size(x, 2)) .-= positions(c.D, T, eltype(θ))
    end
    record!(trace, (; pass = 0, X = X₀))

    X = X₀
    Zs = typeof(X₀)[]
    back_blocks = Vector{Function}(undef, K)
    back_readouts = Vector{Function}(undef, readouts == :all ? K : 1)
    back_stabilize = Vector{Function}(undef, stabilize ? K : 0)
    γ = fill(one(eltype(θ)), c.D)
    β = zero(γ)
    block_trace = isnothing(trace) ? nothing : NamedTuple[]
    for k in 1:K
        X_in = injection == :before ? X + X₀ : X
        pk = p.blocks[tied ? 1 : k]
        if attention_span == :full && iszero(previous_bias)
            Y, back_blocks[k] = block(X_in, pk, c.H, T; trace = block_trace)
        elseif attention_span == :adjacent && iszero(previous_bias)
            Y, back_blocks[k] = adjacent_block(X_in, pk, c.H, T; trace = block_trace)
        else
            Y, back_blocks[k] = relative_block(X_in, pk, c.H, T, attention_span,
                                               previous_bias; trace = block_trace)
        end
        X = if injection == :refresh
            X₀ + s * (Y - X_in)
        elseif isone(s)
            Y
        else
            X_in + s * (Y - X_in)
        end
        injection == :after && (X = X + X₀)
        stabilize && ((X, back_stabilize[k]) = layernorm(X, γ, β))
        keep = readouts == :all || k == K
        if keep || !isnothing(trace)
            Z, back_Z = readout(X, p)
            if keep
                push!(Zs, Z)
                back_readouts[length(Zs)] = back_Z
            end
            if !isnothing(trace)
                frame = only(block_trace)
                boundary = hasproperty(frame, :boundary_attention) ? (; frame.boundary_attention) : (;)
                record!(trace, (; pass = k, X, frame.A, Z, predictions = predictions(Z, T),
                                  boundary...))
                empty!(block_trace)
            end
        end
    end

    function back(Z̄)
        X̄ = zero(X₀)
        X̄₀ = zero(X₀)
        θ̄ = zero(θ)
        q = carve(θ̄, c)
        for k in K:-1:1
            if readouts == :all || k == K
                dX, d_readout = readouts == :all ? back_readouts[k](Z̄[k]) : back_readouts[1](Z̄)
                X̄ .+= dX
                for (dst, src) in zip((q.γf, q.βf, q.Wout, q.bout), d_readout)
                    dst .+= src
                end
            end
            stabilize && (X̄ = first(back_stabilize[k](X̄)))
            # An injection sends the same adjoint to both operands of the sum.
            injection in (:after, :refresh) && (X̄₀ .+= X̄)
            ℓ = tied ? 1 : k
            if injection == :refresh || !isone(s)
                through_block, d_block = back_blocks[k](s * X̄)
                X̄ = through_block + ((injection == :refresh ? zero(s) : one(s)) - s) * X̄
            else
                X̄, d_block = back_blocks[k](X̄)
            end
            injection == :before && (X̄₀ .+= X̄)
            # Every use of the tied block contributes to the same slice of θ̄.
            for (dst, src) in zip(values(q.blocks[ℓ]), d_block)
                dst .+= src
            end
        end
        q.E .= back_embed(X̄ + X̄₀)
        θ̄
    end
    (readouts == :all ? Zs : only(Zs)), back
end

"""
    ℓ, back = looploss(θ, x, y, c; supervision = :last, opts...)

Mean cross-entropy over tokens. With :deep supervision, average it over passes
as well. :cone averages over passes, using the first k positions at pass k.
Give one target matrix y for all selected readouts, or a matrix per pass for
:deep or :cone. The pullback returns θ̄.
"""
function looploss(θ, x, y, c::Config; supervision = :last, opts...)
    supervised_loss(readouts -> loopgpt(θ, x, c; readouts, opts...), x, y, supervision)
end

function supervised_loss(forward, x, y, supervision)
    supervision in (:last, :deep, :cone) || throw(ArgumentError("unknown supervision mode"))
    isempty(x) && throw(ArgumentError("the mean loss needs a nonempty batch"))
    if supervision == :last
        size(y) == size(x) || throw(DimensionMismatch("inputs and targets must have the same shape"))
        Z, back_Z = forward(:last)
        ℓ, back_ℓ = crossentropy(Z, y)
        return ℓ, (ℓ̄ = 1) -> back_Z(back_ℓ(ℓ̄))
    end
    Zs, back_Zs = forward(:all)
    ys = y isa AbstractMatrix ? fill(y, length(Zs)) : y
    length(ys) == length(Zs) && all(yk -> size(yk) == size(x), ys) ||
        throw(DimensionMismatch("deep supervision needs a target matrix at every pass"))
    terms = if supervision == :cone
        # With zero initial state the one-site-per-pass front reaches position k.
        T, B = size(x)
        map(eachindex(Zs)) do k
            t = min(k, T)
            selected = vec([(b - 1) * T + j for j in 1:t, b in 1:B])
            ℓ, pull = crossentropy(view(Zs[k], :, selected), view(ys[k], 1:t, :))
            function back_term(ℓ̄)
                Z̄ = zero(Zs[k])
                Z̄[:, selected] = pull(ℓ̄)
                Z̄
            end
            ℓ, back_term
        end
    else
        map(crossentropy, Zs, ys)
    end
    K = length(terms)
    back(ℓ̄ = 1) = back_Zs([pull(ℓ̄ / K) for (_, pull) in terms])
    sum(first, terms) / K, back
end

end

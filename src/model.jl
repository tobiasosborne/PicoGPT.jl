# Chapter 6 · The model is a composition of maps
#
#   tokens ─embed→ X₀ ─block→ X₁ ─block→ ⋯ ─block→ X_L ─LN→ X_f ─affine→ Z ─crossentropy→ ℓ
#
# Running it pushes values to the right. Training pulls one number, ℓ̄ = 1, back to the left,
# through the adjoint of each map in turn, and collects θ̄ ∈ ℝᵖ on the way.

"""
    Z, back = gpt(θ, x, c)

The model. Token ids x ∈ {1…V}^{T×B} (one sequence per column, T ≤ c.T) go to logits
Z ∈ ℝ^{V×N}, N = T·B: column n of Z scores the candidates for the token after token n.

The pullback maps Z̄ to θ̄ ∈ ℝᵖ, laid out like θ.
"""
function gpt(θ, x, c::Config; trace = nothing)
    T = size(x, 1)
    T ≤ c.T || throw(ArgumentError("sequences of length $T exceed the context length $(c.T)"))
    p = carve(θ, c)
    X, back_embed = embed(x, p.E)
    record!(trace, (; X0 = X))
    back_blocks = Vector{Any}(undef, c.L)
    for ℓ in 1:c.L
        X, back_blocks[ℓ] = block(X, p.blocks[ℓ], c.H, T; trace)
    end
    Xf, back_Xf = layernorm(X, p.γf, p.βf)
    Z,  back_Z  = affine(Xf, p.Wout, p.bout)
    record!(trace, (; Xf, Z))

    function back(Z̄)
        X̄f, W̄out, b̄out = back_Z(Z̄)
        X̄, γ̄f, β̄f = back_Xf(X̄f)
        p̄blocks = Vector{Any}(undef, c.L)
        for ℓ in c.L:-1:1
            X̄, p̄blocks[ℓ] = back_blocks[ℓ](X̄)
        end
        Ē = back_embed(X̄)
        flatten(Ē, p̄blocks, γ̄f, β̄f, W̄out, b̄out)
    end
    Z, back
end

"""
    ℓ, back = loss(θ, x, y, c)

The training objective: the mean surprise of the targets y ∈ {1…V}^{T×B} given the inputs x.
`back()` is the gradient ∇ℓ(θ) ∈ ℝᵖ (and back(ℓ̄) = ℓ̄ ∇ℓ(θ)).
"""
function loss(θ, x, y, c::Config)
    Z, back_Z = gpt(θ, x, c)
    ℓ, back_ℓ = crossentropy(Z, y)
    back(ℓ̄ = 1) = back_Z(back_ℓ(ℓ̄))
    ℓ, back
end

# For the book's figures: record named intermediate values, if asked to.
record!(::Nothing, values) = nothing
record!(trace, values) = push!(trace, values)

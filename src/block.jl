# Chapter 6 · The block
#
# A block makes two residual updates of the stream, X ← X + f(X): small corrections, like
# the steps of Euler's method. In the first, tokens read from each other (attention); in
# the second, each token thinks alone (a two-layer MLP, column by column). Each update
# reads a normalised copy of X, made by LayerNorm.

"""
    Y, back = affine(X, W, b)

The same affine map on every column: Y = W X + b 1ᵀ.

Pullback (Ȳ ↦ X̄, W̄, b̄). dY = W dX + dW X + db 1ᵀ; moving each factor across the
inner product ⟨Ȳ, ⋅⟩ gives X̄ = Wᵀ Ȳ, W̄ = Ȳ Xᵀ and b̄ = Ȳ 1 (the sum over columns).
"""
function affine(X, W, b)
    Y = muladd(W, X, b)
    back(Ȳ) = (W' * Ȳ, Ȳ * X', vec(sum(Ȳ, dims = 2)))
    Y, back
end

"""
    R, back = relu(U)

The rectifier max(u, 0), entry by entry. Its derivative is 1 where u > 0 and 0 where
u < 0; at zero we select the backward value 0 (there is no derivative there).
The pullback is a mask: R̄ ↦ R̄ ⊙ [U > 0], and [U > 0] = [R > 0].
"""
function relu(U)
    R = max.(U, 0)
    back(R̄) = R̄ .* (R .> 0)
    R, back
end

"""
    Y, back = layernorm(X, γ, β)

LayerNorm, column by column. Geometrically, each column x ∈ ℝᴰ is

    centred     x̃ = x − (1ᵀx/D) 1,             the orthogonal projection onto 1⟂;
    scaled      x̂ = x̃/σ,  σ² = |x̃|²/D + ε,     inside the sphere |x̂| < √D, with ε = 1e-5;
    mapped      y = γ ⊙ x̂ + β,                  a diagonal affine map.

Pullback (Ȳ ↦ X̄, γ̄, β̄). γ̄ = Σₙ ȳₙ ⊙ x̂ₙ and β̄ = Σₙ ȳₙ. Let g = γ ⊙ ȳ, the adjoint of x̂.
From σ dσ = x̃⋅dx̃/D, dx̂ = (dx̃ − x̂ (x̂⋅dx̃)/D)/σ; and dx̃ = Π dx with Π = I − 11ᵀ/D. So

    x̄ = Π (g − x̂ (x̂⋅g)/D)/σ = (g − mean(g) 1 − x̂ (x̂⋅g)/D)/σ      (as x̂ ⟂ 1):

For ε = 0 and x̃ ≠ 0 these are tangent and centring projections. With ε > 0 the
formula is still exact, but the radial response is ε x̃/σ³ rather than zero.
"""
function layernorm(X, γ, β)
    D = size(X, 1)
    ε = eltype(X)(1e-5)
    X̃ = X .- sum(X, dims = 1) ./ D                         # centre
    σ = sqrt.(sum(abs2, X̃, dims = 1) ./ D .+ ε)
    X̂ = X̃ ./ σ                                             # inside the sphere
    function back(Ȳ)
        G = γ .* Ȳ
        X̄ = (G .- sum(G, dims = 1) ./ D .- X̂ .* sum(X̂ .* G, dims = 1) ./ D) ./ σ
        X̄, vec(sum(Ȳ .* X̂, dims = 2)), vec(sum(Ȳ, dims = 2))
    end
    γ .* X̂ .+ β, back
end

"""
    X₂, back = block(X₀, p, H, T)

One transformer block with parameters p (a named piece of θ) and H heads:

    Â = LN₁(X₀),   Q, K, V = Wq Â + bq, Wk Â + bk, Wv Â + bv,
    X₁ = X₀ + Wo mix(attention(Q, K), V) + bo                        tokens talk
    F̂ = LN₂(X₁),   X₂ = X₁ + W2 relu(W1 F̂ + b1) + b2                each token thinks

The pullback reads these lines backwards. At a residual sum X₁ = X₀ + Y₁ the adjoint
goes both ways: Ȳ₁ = X̄₁ into the branch, and X̄₁ straight on into X̄₀. It returns X̄₀
and the adjoint of p, as a tuple in the order of θ.
"""
function block(X₀, p, H, T; trace = nothing)
    Â,  back_Â  = layernorm(X₀, p.γ1, p.β1)
    Q,  back_Q  = affine(Â, p.Wq, p.bq)
    K,  back_K  = affine(Â, p.Wk, p.bk)
    V,  back_V  = affine(Â, p.Wv, p.bv)
    A,  back_A  = attention(Q, K, H, T)
    O,  back_O  = mix(A, V)
    Y₁, back_Y₁ = affine(O, p.Wo, p.bo)
    X₁ = X₀ + Y₁
    F̂,  back_F̂  = layernorm(X₁, p.γ2, p.β2)
    U,  back_U  = affine(F̂, p.W1, p.b1)
    R,  back_R  = relu(U)
    Y₂, back_Y₂ = affine(R, p.W2, p.b2)
    X₂ = X₁ + Y₂
    record!(trace, (; ln1 = Â, Q, K, V, A, O, Xmid = X₁, ln2 = F̂, U, R, Y = Y₂, Xout = X₂))

    function back(X̄₂)
        R̄, W̄2, b̄2   = back_Y₂(X̄₂)
        Ū            = back_R(R̄)
        F̂̄, W̄1, b̄1   = back_U(Ū)
        X̄₁, γ̄2, β̄2  = back_F̂(F̂̄)
        X̄₁ .+= X̄₂                                         # the residual path
        Ō, W̄o, b̄o   = back_Y₁(X̄₁)
        Ā, V̄         = back_O(Ō)
        Q̄, K̄         = back_A(Ā)
        Â̄q, W̄q, b̄q  = back_Q(Q̄)
        Â̄k, W̄k, b̄k  = back_K(K̄)
        Â̄v, W̄v, b̄v  = back_V(V̄)
        X̄₀, γ̄1, β̄1  = back_Â(Â̄q + Â̄k + Â̄v)
        X̄₀ .+= X̄₁                                         # the residual path
        X̄₀, (γ̄1, β̄1, W̄q, b̄q, W̄k, b̄k, W̄v, b̄v, W̄o, b̄o, γ̄2, β̄2, W̄1, b̄1, W̄2, b̄2)
    end
    X₂, back
end

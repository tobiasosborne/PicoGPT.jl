# Chapter 2 · Symbols become vectors
#
# Symbol a becomes the column E eₐ of an embedding matrix E ∈ ℝ^{D×V}. (Here eₐ is the
# one-hot vector; we never form it, since multiplying by it just picks a column.)
# A symbol does not know where it stands, so the column P[:, t] of a fixed positional
# code is added: the same letter at two places becomes two different vectors.

"""
    P = positions(D, T)

The positional code P ∈ ℝ^{D×T}: Fourier features of the position t. Rows 2i−1 and 2i
hold sin and cos of ωᵢ(t − 1), at the geometrically spaced frequencies

    ωᵢ = 10000^(−(i−1)/(D/2)),   i = 1 … D/2,

from one radian per step down towards 1/10000, like the hands of a clock. D must be
positive and even, and T nonnegative. The optional third argument sets the float type.
"""
function positions(D, T, R::Type{<:AbstractFloat} = Float64)
    D > 0 && iseven(D) && T ≥ 0 || throw(ArgumentError("positions needs positive even D and nonnegative T"))
    ω(k) = R(10000)^(-R((k - 1) ÷ 2) / (D ÷ 2))    # rows 2i−1 and 2i share ωᵢ
    [isodd(k) ? sin(ω(k) * (t - 1)) : cos(ω(k) * (t - 1)) for k in 1:D, t in 1:T]
end

"""
    X, back = embed(x, E)

The residual stream at the start. The token ids x ∈ {1…V}^{T×B} hold one sequence per
column; X ∈ ℝ^{D×N}, N = T·B, holds one column per token n, at place t of its sequence:

    X[:, n] = E[:, xₙ] + P[:, t].

Pullback (X̄ ↦ Ē). dX[:, n] = dE[:, xₙ], so ⟨X̄, dX⟩ = Σₙ ⟨X̄[:, n], dE[:, xₙ]⟩, and
column a of Ē collects the adjoints of all the tokens that are symbol a:
Ē[:, a] = Σ_{n : xₙ = a} X̄[:, n]. The adjoint of a lookup is a scatter-add.
"""
function embed(x, E)
    D, T = size(E, 1), size(x, 1)
    X = E[:, vec(x)]                               # look up one column per token
    T > 0 || throw(ArgumentError("sequences must contain at least one token"))
    reshape(X, D, T, size(x, 2)) .+= positions(D, T, eltype(E))  # the same P for every sequence
    function back(X̄)
        Ē = zero(E)
        for (n, a) in enumerate(x)
            @views Ē[:, a] .+= X̄[:, n]
        end
        Ē
    end
    X, back
end

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

from one radian per step (fast) down to nearly 1/10000 (slow), like the hands of a clock.
"""
function positions(D, T)
    ω(k) = 10000.0^(-((k - 1) ÷ 2) / (D ÷ 2))       # rows 2i−1 and 2i share ωᵢ
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
    reshape(X, D, T, :) .+= positions(D, T)        # the same P for every sequence
    function back(X̄)
        Ē = zero(E)
        for (n, a) in enumerate(x)
            @views Ē[:, a] .+= X̄[:, n]
        end
        Ē
    end
    X, back
end

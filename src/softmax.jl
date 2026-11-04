# Chapters 3–4 · Energies, probabilities, surprise
#
# The model's output for one token is a vector of logits z ∈ ℝ^V: negative energies,
# one per symbol. Its prediction is the Gibbs distribution p = softmax(z). It is
# scored by its surprise at what actually came next, −log p_y, averaged over tokens.

"""
    logsumexp(z) = log Σₐ exp(zₐ)

The logarithm of the partition function. With m = max z, every exponent in
log Σₐ exp zₐ = m + log Σₐ exp(zₐ − m) is ≤ 0, so exponentiation cannot overflow
for finite logits.
"""
logsumexp(z) = (m = maximum(z); m + log(sum(zₐ -> exp(zₐ - m), z)))

"""
    p = softmax(z; dims = :)

The Gibbs distribution with energies −z at temperature 1: pₐ = exp(zₐ) / Σ_b exp(z_b).
Adding a constant to every zₐ changes nothing (only energy differences matter), so
the maximum is subtracted first, for safety. With `dims`, each slice along `dims` is
normalised (dims = 2: each row). `softmax!` overwrites z with p.
"""
softmax(z; dims = :) = softmax!(float.(z); dims)
softmax!(z; dims = :) = (z .= exp.(z .- maximum(z; dims)); z ./= sum(z; dims))

"""
    ℓ, back = crossentropy(Z, y)

The mean surprise of the targets y ∈ {1…V}^N under the predictions softmax(zₙ):

    ℓ = (1/N) Σₙ −log softmax(zₙ)[yₙ] = (1/N) Σₙ [ logsumexp(zₙ) − Z[yₙ, n] ].

Pullback (ℓ̄ ↦ Z̄). ∂ logsumexp(z)/∂zₐ = softmax(z)ₐ and ∂z_y/∂zₐ = δₐᵧ, so

    Z̄ = ℓ̄ (softmax(Z) − onehot(y)) / N,

column by column: the prediction minus the truth.
The batch must contain at least one token. Subtract the column maximum from both
terms of the loss before subtracting them, so a large common logit shift cannot
erase the surprise or spoil the probabilities used by the pullback.
"""
function crossentropy(Z, y)
    N = size(Z, 2)
    N > 0 || throw(ArgumentError("cross-entropy needs at least one token"))
    length(y) == N || throw(DimensionMismatch("one target is required per logit column"))
    m = maximum(Z; dims = 1)
    s = map(eachcol(Z), vec(m)) do z, mₙ
        sum(zₐ -> exp(zₐ - mₙ), z)
    end
    ℓ = sum(log(s[n]) - (Z[y[n], n] - m[n]) for n in 1:N) / N
    function back(ℓ̄ = 1)
        Z̄ = exp.(Z .- m) ./ s'                    # softmax(zₙ), for every n
        for n in 1:N
            Z̄[y[n], n] -= 1                         # − onehot(yₙ)
        end
        Z̄ .*= eltype(Z)(ℓ̄) / N
    end
    ℓ, back
end

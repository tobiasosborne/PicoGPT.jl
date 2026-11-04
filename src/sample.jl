# Chapter 9 · Speaking
#
# To write, run the model on the text so far, turn the logits of the last position into a
# distribution, draw the next symbol from it, append it, and repeat. Dividing the logits by
# τ sets the temperature of the Gibbs system with the same energies: as τ → 0 the Gibbs distribution freezes onto
# its ground states (uniform among exactly tied maxima); as τ → ∞ it melts to uniform.

"""
    a = draw([rng,] p)

A sample from the distribution p on 1:V, by inverting its distribution function: the first
a with p₁ + ⋯ + pₐ > u, for u uniform on [0, Σp).
"""
function draw(rng, p)
    !isempty(p) && all(x -> isfinite(x) && x ≥ 0, p) ||
        throw(ArgumentError("weights must be nonempty, finite and nonnegative"))
    F = cumsum(p)
    isfinite(F[end]) && F[end] > 0 || throw(ArgumentError("weights need a finite positive sum"))
    something(findfirst(>(rand(rng) * F[end]), F), length(p))
end
draw(p) = draw(Random.default_rng(), p)

"""
    x = generate(θ, c, prompt, n; τ = 1, rng = Random.default_rng())

The token sequence `prompt` extended by n tokens, each drawn from softmax(z/τ), where z are
the logits after the last T tokens. τ = 0 always takes the likeliest token.
"""
function generate(θ, c::Config, prompt, n; τ = 1, rng = Random.default_rng())
    n ≥ 0 && isfinite(τ) && τ ≥ 0 || throw(ArgumentError("generation needs n ≥ 0 and finite τ ≥ 0"))
    n == 0 || !isempty(prompt) || throw(ArgumentError("generation needs a nonempty prompt"))
    τ = eltype(θ)(τ)
    x = collect(prompt)
    for _ in 1:n
        context = x[max(end - c.T + 1, 1):end]
        Z, _ = gpt(θ, reshape(context, :, 1), c)
        z = Z[:, end]
        push!(x, τ == 0 ? argmax(z) : draw(rng, softmax(z ./ τ)))
    end
    x
end

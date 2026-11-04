# Chapter 9 · Speaking
#
# To write, run the model on the text so far, turn the logits of the last position into a
# distribution, draw the next symbol from it, append it, and repeat. Dividing the logits by
# a temperature τ is literally a temperature: as τ → 0 the Gibbs distribution freezes onto
# its ground state, the likeliest symbol; as τ → ∞ it melts to uniform.

"""
    a = draw([rng,] p)

A sample from the distribution p on 1:V, by inverting its distribution function: the first
a with p₁ + ⋯ + pₐ > u, for u uniform on [0, Σp).
"""
function draw(rng, p)
    F = cumsum(p)
    something(findfirst(>(rand(rng) * F[end]), F), length(p))
end
draw(p) = draw(Random.default_rng(), p)

"""
    x = generate(θ, c, prompt, n; τ = 1, rng = Random.default_rng())

The token sequence `prompt` extended by n tokens, each drawn from softmax(z/τ), where z are
the logits after the last T tokens. τ = 0 always takes the likeliest token.
"""
function generate(θ, c::Config, prompt, n; τ = 1, rng = Random.default_rng())
    x = collect(prompt)
    for _ in 1:n
        context = x[max(end - c.T + 1, 1):end]
        Z, _ = gpt(θ, reshape(context, :, 1), c)
        z = Z[:, end]
        push!(x, τ == 0 ? argmax(z) : draw(rng, softmax(z ./ τ)))
    end
    x
end

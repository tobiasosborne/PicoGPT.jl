"""
    PicoGPT

A small GPT, written to be read alongside the book.

Every operation returns its value and its *pullback*,

    y, back = op(x, w…)            back(ȳ) = (x̄, w̄…),

the adjoint of the derivative: ⟨ȳ, dy⟩ = ⟨x̄, dx⟩ + ⟨w̄, dw⟩ + ⋯. The backward pass is
the forward pass read in reverse. No automatic differentiation is used anywhere.

Conventions. The residual stream X ∈ ℝ^{D×N} has one column per token; a batch of B
sequences of length T is just N = T·B columns. All parameters live in one vector
θ ∈ ℝᵖ, and the gradient θ̄ is a vector of the same length.

The files follow the chapters of the book.
"""
module PicoGPT

using LinearAlgebra, Random

include("text.jl")         # 1    text as a sequence of symbols
include("embedding.jl")    # 2    symbols become vectors
include("softmax.jl")      # 3–4  energies, probabilities, surprise
include("attention.jl")    # 5    attention: the only place tokens meet
include("block.jl")        # 6    the block
include("parameters.jl")   # 6    the model is a point in ℝᵖ
include("model.jl")        # 6    the model is a composition of maps
include("gradcheck.jl")    # 7    pulling back, checked
include("train.jl")        # 8    learning
include("sample.jl")       # 9    speaking

export alphabet, encode, decode,
       positions, embed,
       logsumexp, softmax, softmax!, crossentropy,
       attention, mix,
       affine, relu, layernorm, block,
       Config, nparams, carve, leaves, flatten, initial, init,
       gpt, loss,
       gradcheck, checkpullback,
       windows, ∇loss, clip!, Descent, AdamW, update!, decay, warmup_cosine, step!,
       draw, generate

end

# config.jl -- GPTConfig and TrainConfig structs with defaults
#
# All hyperparameters for the model and training loop live here.
# No dependencies.

using LinearAlgebra

"""
Model architecture hyperparameters (GPT-2 style).
"""
struct GPTConfig
    vocab_size::Int   # V -- number of unique tokens (data-determined)
    d_model::Int      # D -- embedding / hidden dimension
    n_heads::Int      # H -- number of attention heads
    n_layers::Int     # number of transformer blocks
    seq_len::Int      # T -- context window length
    d_ff::Int         # feedforward hidden dimension (typically 4 * d_model)
end

"""
    GPTConfig(; vocab_size=65, d_model=128, n_heads=4, n_layers=4, seq_len=256)

Construct with defaults.  `d_ff` is computed as `4 * d_model` unless overridden.
"""
function GPTConfig(; vocab_size::Int=65,
                     d_model::Int=128,
                     n_heads::Int=4,
                     n_layers::Int=4,
                     seq_len::Int=256,
                     d_ff::Int=4*d_model)
    @assert d_model % n_heads == 0 "d_model ($d_model) must be divisible by n_heads ($n_heads)"
    GPTConfig(vocab_size, d_model, n_heads, n_layers, seq_len, d_ff)
end

"""
Training hyperparameters.
"""
struct TrainConfig
    batch_size::Int
    learning_rate::Float64   # eta_max
    lr_min::Float64          # eta_min for cosine decay
    warmup_steps::Int
    n_epochs::Int
    max_grad_norm::Float64
    eval_interval::Int       # evaluate every N epochs
end

function TrainConfig(; batch_size::Int=32,
                       learning_rate::Float64=1e-3,
                       lr_min::Float64=1e-5,
                       warmup_steps::Int=100,
                       n_epochs::Int=20,
                       max_grad_norm::Float64=1.0,
                       eval_interval::Int=1)
    TrainConfig(batch_size, learning_rate, lr_min, warmup_steps, n_epochs,
                max_grad_norm, eval_interval)
end

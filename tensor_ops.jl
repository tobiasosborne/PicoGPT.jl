# tensor_ops.jl -- Hand-written tensor operations with forward + backward
#
# Every operation has:
#   forward(args...) -> (result, cache)
#   backward(cache, grad_output) -> (grad_input, grad_params...)
#
# Array layout (Julia is column-major):
#   Hidden states: (D, T, B) -- contiguous along embedding dimension
#   Logits:        (V, T, B)
#   Embeddings:    (D, V)    -- each column is an embedding vector

using Statistics: mean

# ===================================================================
# Weight initialization
# ===================================================================

# DERIVATION: Xavier / Glorot uniform
# For y = Wx where x ∈ R^{fan_in}, we want Var[y_j] ≈ 1.
# Var[y_j] = fan_in · Var[w] · Var[x].  If Var[x]=1 we need Var[w] = 1/fan_in.
# Xavier balances forward and backward: Var[w] = 2/(fan_in + fan_out).

"""
    xavier_init(fan_in, fan_out) -> Matrix{Float64}

Gaussian Xavier: W ~ N(0, √(2/(fan_in + fan_out))).
"""
function xavier_init(fan_in::Int, fan_out::Int)
    std = sqrt(2.0 / (fan_in + fan_out))
    randn(fan_out, fan_in) .* std
end

"""
    gpt2_init(fan_in, fan_out; scale=1.0) -> Matrix{Float64}

GPT-2 style: W ~ N(0, 0.02 * scale).
Residual projections use scale = 1/√(2*n_layers) to keep variance stable.
"""
function gpt2_init(fan_in::Int, fan_out::Int; scale::Float64=1.0)
    randn(fan_out, fan_in) .* (0.02 * scale)
end

"""
    zeros_init(dims...) -> Array{Float64}
"""
zeros_init(dims...) = zeros(Float64, dims...)

# ===================================================================
# Embedding lookup
# ===================================================================

"""
    embed_forward(E, token_ids) -> (h, cache)

Look up token embeddings.  E is (D, V), token_ids is (T, B) of integers in [1, V].
Returns h of shape (D, T, B).
"""
function embed_forward(E::Matrix{Float64}, token_ids::Matrix{Int})
    D, V = size(E)
    T, B = size(token_ids)
    # Bulk column gather: flatten token_ids, index, reshape
    h = reshape(E[:, vec(token_ids)], D, T, B)
    (h, (E, token_ids))
end

"""
    embed_backward(cache, grad_h) -> grad_E

Scatter-add: accumulate gradients into the rows that were looked up.
"""
function embed_backward(cache, grad_h::Array{Float64,3})
    E, token_ids = cache
    D, V = size(E)
    T, B = size(token_ids)
    grad_E = zeros(D, V)
    for b in 1:B, t in 1:T
        @views grad_E[:, token_ids[t, b]] .+= grad_h[:, t, b]
    end
    grad_E
end

# ===================================================================
# Sinusoidal positional encoding  (Vaswani et al.)
# ===================================================================

# PE(pos, 2i)   = sin(pos / 10000^(2i/d))
# PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
#
# No learnable parameters -> no backward needed.

"""
    sinusoidal_pe(seq_len, d_model) -> PE matrix (D, T)

Precomputed, added once to embeddings.
"""
function sinusoidal_pe(seq_len::Int, d_model::Int)
    pe = zeros(d_model, seq_len)
    for pos in 0:(seq_len-1)
        for i in 0:2:(d_model-1)
            denom = 10000.0^(i / d_model)
            pe[i+1, pos+1] = sin(pos / denom)
            if i + 2 <= d_model
                pe[i+2, pos+1] = cos(pos / denom)
            end
        end
    end
    pe
end

# ===================================================================
# Softmax
# ===================================================================

# FORWARD: softmax(x)_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
# Subtracting max for numerical stability does not change the result because
# exp(x_i - c) / Σ exp(x_j - c) = exp(x_i) / Σ exp(x_j).
#
# BACKWARD: dL/dx_i = s_i (dL/ds_i - Σ_j s_j dL/ds_j)
#
# Derivation (index notation):
#   ∂s_i/∂x_k = s_i (δ_ik - s_k)
#   dL/dx_k = Σ_i dL/ds_i · s_i (δ_ik - s_k)
#           = dL/ds_k · s_k  -  s_k Σ_i dL/ds_i · s_i
#           = s_k (dL/ds_k  -  Σ_i s_i dL/ds_i)

"""
    softmax_forward(x) -> (s, cache)

Numerically stable softmax along dimension 1 (feature/vocab dim).
"""
function softmax_forward(x::AbstractArray{Float64})
    x_max = maximum(x, dims=1)
    ex = exp.(x .- x_max)
    s = ex ./ sum(ex, dims=1)
    (s, (s,))
end

"""
    softmax_backward(cache, grad_s) -> grad_x
"""
function softmax_backward(cache, grad_s::AbstractArray{Float64})
    s, = cache
    # dot = Σ_j s_j · dL/ds_j  (per column)
    dot = sum(s .* grad_s, dims=1)
    s .* (grad_s .- dot)
end

# ===================================================================
# Layer Norm
# ===================================================================

# FORWARD: y = γ ⊙ x̂ + β,   x̂ = (x - μ) / σ
# where μ = (1/D) Σ_i x_i,  σ = √(Var + ε),  Var = (1/D) Σ_i (x_i - μ)²
#
# BACKWARD:
#   dL/dγ = Σ_{t,b} dL/dy ⊙ x̂          (sum over batch and time)
#   dL/dβ = Σ_{t,b} dL/dy
#   dL/dx̂ = dL/dy ⊙ γ
#   dL/dσ² = -(1/2) Σ_i dL/dx̂_i (x_i - μ) σ^{-3}
#   dL/dμ  = -σ^{-1} Σ_i dL/dx̂_i  +  dL/dσ² · (-2/D) Σ_i (x_i - μ)
#   dL/dx_i = dL/dx̂_i · σ^{-1}  +  dL/dσ² · (2/D)(x_i - μ)  +  dL/dμ · (1/D)

"""
    layer_norm_forward(x, gamma, beta; eps=1e-5) -> (y, cache)

x: (D, T, B),  gamma, beta: (D,).  Normalizes along dim 1.
"""
function layer_norm_forward(x::AbstractArray{Float64}, gamma::Vector{Float64},
                            beta::Vector{Float64}; eps::Float64=1e-5)
    D = size(x, 1)
    mu = mean(x, dims=1)                          # (1, T, B)
    diff = x .- mu
    var_ = mean(diff .^ 2, dims=1)                # (1, T, B)
    sigma_inv = 1.0 ./ sqrt.(var_ .+ eps)         # (1, T, B)
    x_hat = diff .* sigma_inv                      # (D, T, B)
    y = gamma .* x_hat .+ beta                     # broadcast gamma (D,) over (D,T,B)
    cache = (x, x_hat, gamma, mu, sigma_inv, D)
    (y, cache)
end

"""
    layer_norm_backward(cache, grad_y) -> (grad_x, grad_gamma, grad_beta)
"""
function layer_norm_backward(cache, grad_y::AbstractArray{Float64})
    x, x_hat, gamma, mu, sigma_inv, D = cache

    # Parameter gradients (sum over T and B)
    grad_gamma = vec(sum(grad_y .* x_hat, dims=(2,3)))  # keep only along 2,3 -> length D
    # handle 2D input (no batch dim 3)
    if ndims(grad_y) == 2
        grad_gamma = vec(sum(grad_y .* x_hat, dims=2))
    end
    grad_beta = ndims(grad_y) == 2 ? vec(sum(grad_y, dims=2)) : vec(sum(grad_y, dims=(2,3)))

    # Input gradient
    dx_hat = grad_y .* gamma                            # (D, T, B)
    diff = x .- mu
    dvar = sum(dx_hat .* diff, dims=1) .* (-0.5) .* sigma_inv .^ 3
    dmu  = sum(dx_hat .* (-sigma_inv), dims=1) .+ dvar .* (-2.0 / D) .* sum(diff, dims=1)
    grad_x = dx_hat .* sigma_inv .+ dvar .* (2.0 / D) .* diff .+ dmu ./ D

    (grad_x, grad_gamma, grad_beta)
end

# ===================================================================
# ReLU
# ===================================================================

# FORWARD: y_i = max(0, x_i)
# BACKWARD: dL/dx_i = dL/dy_i · 𝟙[x_i > 0]

"""
    relu_forward(x) -> (y, cache)
"""
function relu_forward(x::AbstractArray{Float64})
    mask = x .> 0
    (x .* mask, (mask,))
end

"""
    relu_backward(cache, grad_y) -> grad_x
"""
function relu_backward(cache, grad_y::AbstractArray{Float64})
    mask, = cache
    grad_y .* mask
end

# ===================================================================
# Cross-Entropy Loss  (fused with softmax)
# ===================================================================

# FORWARD: L = -(1/N) Σ_{t,b} log p(target_{t,b})
# where p = softmax(logits) and N = T*B.
#
# BACKWARD (fused softmax + CE):
#   dL/dlogits_{v,t,b} = (1/N) · (p_{v,t,b} - 𝟙[v = target_{t,b}])
#
# This is the standard result: the gradient of cross-entropy w.r.t. logits
# is simply (softmax output - one-hot target), averaged over the batch.

"""
    cross_entropy_forward(logits, targets) -> (loss, cache)

logits: (V, T, B), targets: (T, B) of integers in [1, V].
Returns scalar mean loss.
"""
function cross_entropy_forward(logits::AbstractArray{Float64}, targets::AbstractArray{Int})
    V, T, B = size(logits)
    # Stable softmax along dim 1
    probs, _ = softmax_forward(logits)

    # Gather log-probs at target positions
    loss = 0.0
    for b in 1:B, t in 1:T
        loss -= log(probs[targets[t, b], t, b] + 1e-10)
    end
    loss /= (T * B)

    (loss, (probs, targets, T, B))
end

"""
    cross_entropy_backward(cache) -> grad_logits

No `grad_output` arg needed: this is the final loss, so dL/dL = 1.
"""
function cross_entropy_backward(cache)
    probs, targets, T, B = cache
    grad = copy(probs)
    N = T * B
    for b in 1:B, t in 1:T
        grad[targets[t, b], t, b] -= 1.0
    end
    grad ./= N
end

# attention.jl -- Multi-head causal self-attention (forward + backward)
#
# Array layout (Julia column-major):
#   x, h:          (D, T, B)
#   Q, K, V per head: (Dh, T, B)   where Dh = D / H
#   Attention weights: (T, T, B)    per head -- attn[i,j] = how much position i attends to j
#   Causal mask:       (T, T)       -- 0 on/below diagonal, -Inf above

# ===================================================================
# Mutable params struct (so we can .= into it for gradcheck)
# ===================================================================

mutable struct AttentionParams
    Wq::Matrix{Float64}   # (D, D)
    bq::Vector{Float64}   # (D,)
    Wk::Matrix{Float64}   # (D, D)
    bk::Vector{Float64}   # (D,)
    Wv::Matrix{Float64}   # (D, D)
    bv::Vector{Float64}   # (D,)
    Wo::Matrix{Float64}   # (D, D)
    bo::Vector{Float64}   # (D,)
    n_heads::Int
end

struct AttentionGrads
    Wq::Matrix{Float64}
    bq::Vector{Float64}
    Wk::Matrix{Float64}
    bk::Vector{Float64}
    Wv::Matrix{Float64}
    bv::Vector{Float64}
    Wo::Matrix{Float64}
    bo::Vector{Float64}
end

function init_attention_params(cfg::GPTConfig)
    D = cfg.d_model
    scale = 1.0 / sqrt(2.0 * cfg.n_layers)
    AttentionParams(
        xavier_init(D, D), zeros(D),   # Wq, bq
        xavier_init(D, D), zeros(D),   # Wk, bk
        xavier_init(D, D), zeros(D),   # Wv, bv
        gpt2_init(D, D; scale=scale), zeros(D),  # Wo, bo  (residual projection)
        cfg.n_heads
    )
end

function copy_attention_params(p::AttentionParams)
    AttentionParams(copy(p.Wq), copy(p.bq), copy(p.Wk), copy(p.bk),
                    copy(p.Wv), copy(p.bv), copy(p.Wo), copy(p.bo), p.n_heads)
end

# ===================================================================
# Linear projection  (y = W*x + b, applied per (t, b) slice)
# ===================================================================

# MATMUL GRADIENT DERIVATION:
#   Forward: y = W x          (for a single vector x ∈ R^{D_in}, y ∈ R^{D_out})
#   dL/dW = dL/dy · x^T       (outer product, summed over T, B)
#   dL/dx = W^T · dL/dy
#   dL/db = sum_{t,b} dL/dy

function linear_forward(W::Matrix{Float64}, b::Vector{Float64},
                        x::Array{Float64,3})
    D_out, D_in = size(W)
    D, T, B = size(x)
    @assert D == D_in
    y = zeros(D_out, T, B)
    for bb in 1:B, t in 1:T
        @views y[:, t, bb] .= W * x[:, t, bb] .+ b
    end
    (y, (W, b, x))
end

function linear_backward(cache, grad_y::Array{Float64,3})
    W, b, x = cache
    D_out, D_in = size(W)
    _, T, B = size(grad_y)

    grad_x = zeros(D_in, T, B)
    grad_W = zeros(D_out, D_in)
    grad_b = zeros(D_out)

    for bb in 1:B, t in 1:T
        @views grad_x[:, t, bb] .= W' * grad_y[:, t, bb]
        @views grad_W .+= grad_y[:, t, bb] * x[:, t, bb]'
        @views grad_b .+= grad_y[:, t, bb]
    end
    (grad_x, grad_W, grad_b)
end

# ===================================================================
# Causal mask
# ===================================================================

"""
    causal_mask(T) -> Matrix{Float64} of shape (T, T)

mask[i,j] = 0 if j ≤ i (attend), -Inf if j > i (block future).
"""
function causal_mask(T::Int)
    mask = zeros(T, T)
    for i in 1:T, j in 1:T
        if j > i
            mask[i, j] = -Inf
        end
    end
    mask
end

# ===================================================================
# Single-head scaled dot-product attention
# ===================================================================

# FORWARD:
#   scores = Q^T K / √Dh          (T, T, B)  -- scores[i,j] = query_i · key_j
#   attn   = softmax(scores + mask)  along dim 2 (key dimension)
#   out    = V * attn^T            (Dh, T, B)  -- weighted sum of values
#
# Here we compute per-batch:
#   S[i,j,b] = (1/√Dh) Σ_d Q[d,i,b] K[d,j,b]
#   A[i,:,b] = softmax(S[i,:,b] + mask[i,:])
#   out[d,i,b] = Σ_j V[d,j,b] A[i,j,b]
#
# BACKWARD:
#   dL/dV[d,j,b] = Σ_i dL/dout[d,i,b] · A[i,j,b]
#   dL/dA[i,j,b] = Σ_d dL/dout[d,i,b] · V[d,j,b]
#   dL/dS = softmax_backward(dL/dA)      (per row i)
#   dL/dQ[d,i,b] = (1/√Dh) Σ_j dL/dS[i,j,b] · K[d,j,b]
#   dL/dK[d,j,b] = (1/√Dh) Σ_i dL/dS[i,j,b] · Q[d,i,b]

function single_head_attention_forward(Q::Array{Float64,3}, K::Array{Float64,3},
                                       V::Array{Float64,3}, mask::Matrix{Float64})
    Dh, T, B = size(Q)
    scale = 1.0 / sqrt(Float64(Dh))

    scores = zeros(T, T, B)
    for bb in 1:B
        @views scores[:, :, bb] .= (Q[:, :, bb]' * K[:, :, bb]) .* scale .+ mask
    end

    # softmax per row (along dim 2 = key dimension)
    attn = zeros(T, T, B)
    attn_caches = []
    for bb in 1:B, i in 1:T
        row = scores[i, :, bb]
        s, c = softmax_forward(row)
        attn[i, :, bb] = s
        push!(attn_caches, c)
    end

    out = zeros(Dh, T, B)
    for bb in 1:B
        @views out[:, :, bb] .= V[:, :, bb] * attn[:, :, bb]'
    end

    cache = (Q, K, V, attn, attn_caches, mask, scale, Dh, T, B)
    (out, cache)
end

function single_head_attention_backward(cache, grad_out::Array{Float64,3})
    Q, K, V, attn, attn_caches, mask, scale, Dh, T, B = cache

    grad_Q = zeros(Dh, T, B)
    grad_K = zeros(Dh, T, B)
    grad_V = zeros(Dh, T, B)

    # dL/dA[i,j,b] = Σ_d dL/dout[d,i,b] · V[d,j,b]
    grad_attn = zeros(T, T, B)
    for bb in 1:B
        @views grad_attn[:, :, bb] .= grad_out[:, :, bb]' * V[:, :, bb]
        # dL/dV[d,j,b] = Σ_i dL/dout[d,i,b] · A[i,j,b]
        @views grad_V[:, :, bb] .= grad_out[:, :, bb] * attn[:, :, bb]
    end

    # attn -> scores via softmax backward (per row)
    grad_scores = zeros(T, T, B)
    idx = 1
    for bb in 1:B, i in 1:T
        c = attn_caches[idx]
        row_grad = softmax_backward(c, grad_attn[i, :, bb])
        grad_scores[i, :, bb] = row_grad
        idx += 1
    end

    # scores -> Q, K
    for bb in 1:B
        @views grad_Q[:, :, bb] .= K[:, :, bb] * grad_scores[:, :, bb]' .* scale
        @views grad_K[:, :, bb] .= Q[:, :, bb] * grad_scores[:, :, bb] .* scale
    end

    (grad_Q, grad_K, grad_V)
end

# ===================================================================
# Multi-head causal self-attention
# ===================================================================

# Split x into H heads, project Q/K/V, run attention, concat, project Wo.
#
# Forward:
#   Q_full = Wq * x + bq   (D, T, B)  then split into H heads of (Dh, T, B)
#   K_full = Wk * x + bk
#   V_full = Wv * x + bv
#   per head h:  out_h = attention(Q_h, K_h, V_h, mask)
#   concat -> (D, T, B)
#   output = Wo * concat + bo

function multi_head_attention_forward(params::AttentionParams, x::Array{Float64,3},
                                      mask::Matrix{Float64})
    D, T, B = size(x)
    H = params.n_heads
    Dh = div(D, H)

    Q_full, cache_q = linear_forward(params.Wq, params.bq, x)
    K_full, cache_k = linear_forward(params.Wk, params.bk, x)
    V_full, cache_v = linear_forward(params.Wv, params.bv, x)

    # Split into heads and run attention
    head_outs = Array{Float64,3}[]
    head_caches = []
    for h in 1:H
        rows = (h-1)*Dh+1 : h*Dh
        Qh = Q_full[rows, :, :]
        Kh = K_full[rows, :, :]
        Vh = V_full[rows, :, :]
        out_h, cache_h = single_head_attention_forward(Qh, Kh, Vh, mask)
        push!(head_outs, out_h)
        push!(head_caches, cache_h)
    end

    # Concat heads -> (D, T, B)
    concat = zeros(D, T, B)
    for h in 1:H
        rows = (h-1)*Dh+1 : h*Dh
        concat[rows, :, :] .= head_outs[h]
    end

    # Output projection
    output, cache_o = linear_forward(params.Wo, params.bo, concat)

    cache = (params, cache_q, cache_k, cache_v, head_caches, cache_o, H, Dh, D, T, B)
    (output, cache)
end

function multi_head_attention_backward(cache, grad_output::Array{Float64,3})
    params, cache_q, cache_k, cache_v, head_caches, cache_o, H, Dh, D, T, B = cache

    # Backward through output projection
    grad_concat, grad_Wo, grad_bo = linear_backward(cache_o, grad_output)

    # Split grad_concat into heads and backward through attention
    grad_Q_full = zeros(D, T, B)
    grad_K_full = zeros(D, T, B)
    grad_V_full = zeros(D, T, B)

    for h in 1:H
        rows = (h-1)*Dh+1 : h*Dh
        grad_h = grad_concat[rows, :, :]
        gQ, gK, gV = single_head_attention_backward(head_caches[h], grad_h)
        grad_Q_full[rows, :, :] .= gQ
        grad_K_full[rows, :, :] .= gK
        grad_V_full[rows, :, :] .= gV
    end

    # Backward through Q, K, V projections
    grad_x_q, grad_Wq, grad_bq = linear_backward(cache_q, grad_Q_full)
    grad_x_k, grad_Wk, grad_bk = linear_backward(cache_k, grad_K_full)
    grad_x_v, grad_Wv, grad_bv = linear_backward(cache_v, grad_V_full)

    grad_x = grad_x_q .+ grad_x_k .+ grad_x_v

    grads = AttentionGrads(grad_Wq, grad_bq, grad_Wk, grad_bk,
                           grad_Wv, grad_bv, grad_Wo, grad_bo)
    (grad_x, grads)
end

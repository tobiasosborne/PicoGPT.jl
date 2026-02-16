# model.jl -- GPT model: transformer blocks + full forward/backward wiring
#
# Architecture (pre-norm residual):
#   h = h + MHA(LayerNorm(h))
#   h = h + FFN(ReLU(LayerNorm(h)))
#
# Depends on: config.jl, tensor_ops.jl, attention.jl

# ===================================================================
# Transformer Block
# ===================================================================

mutable struct TransformerBlock
    attn::AttentionParams
    ln1_gamma::Vector{Float64}
    ln1_beta::Vector{Float64}
    W1::Matrix{Float64}    # (d_ff, D)  -- FFN first layer
    b1::Vector{Float64}    # (d_ff,)
    W2::Matrix{Float64}    # (D, d_ff)  -- FFN second layer
    b2::Vector{Float64}    # (D,)
    ln2_gamma::Vector{Float64}
    ln2_beta::Vector{Float64}
end

function init_transformer_block(cfg::GPTConfig)
    D = cfg.d_model
    d_ff = cfg.d_ff
    scale = 1.0 / sqrt(2.0 * cfg.n_layers)
    TransformerBlock(
        init_attention_params(cfg),
        ones(D), zeros(D),                         # ln1
        xavier_init(D, d_ff), zeros(d_ff),          # W1, b1
        gpt2_init(d_ff, D; scale=scale), zeros(D),  # W2, b2 (residual proj)
        ones(D), zeros(D)                           # ln2
    )
end

function transformer_block_forward(block::TransformerBlock, h::Array{Float64,3},
                                   mask::Matrix{Float64})
    # --- Sub-block 1: h = h + MHA(LN(h)) ---
    ln1_out, cache_ln1 = layer_norm_forward(h, block.ln1_gamma, block.ln1_beta)
    attn_out, cache_attn = multi_head_attention_forward(block.attn, ln1_out, mask)
    h2 = h .+ attn_out   # residual

    # --- Sub-block 2: h = h + FFN(ReLU(LN(h))) ---
    ln2_out, cache_ln2 = layer_norm_forward(h2, block.ln2_gamma, block.ln2_beta)
    ff1, cache_ff1 = linear_forward(block.W1, block.b1, ln2_out)
    ff1_relu, cache_relu = relu_forward(ff1)
    ff2, cache_ff2 = linear_forward(block.W2, block.b2, ff1_relu)
    h3 = h2 .+ ff2   # residual

    cache = (block, h, h2, cache_ln1, cache_attn, cache_ln2, cache_ff1, cache_relu, cache_ff2)
    (h3, cache)
end

function transformer_block_backward(cache, grad_h3::Array{Float64,3})
    block, h, h2, cache_ln1, cache_attn, cache_ln2, cache_ff1, cache_relu, cache_ff2 = cache

    grads = Dict{Symbol, Any}()

    # --- backward sub-block 2 ---
    grad_h2_res = grad_h3                  # residual gradient
    grad_ff2 = grad_h3                     # same, split at the add

    grad_ff1_relu, grad_W2, grad_b2 = linear_backward(cache_ff2, grad_ff2)
    grads[:W2] = grad_W2; grads[:b2] = grad_b2

    grad_ff1 = relu_backward(cache_relu, grad_ff1_relu)
    grad_ln2_out, grad_W1, grad_b1 = linear_backward(cache_ff1, grad_ff1)
    grads[:W1] = grad_W1; grads[:b1] = grad_b1

    grad_h2_ln, grad_ln2_gamma, grad_ln2_beta = layer_norm_backward(cache_ln2, grad_ln2_out)
    grads[:ln2_gamma] = grad_ln2_gamma; grads[:ln2_beta] = grad_ln2_beta

    grad_h2 = grad_h2_res .+ grad_h2_ln

    # --- backward sub-block 1 ---
    grad_h_res = grad_h2
    grad_attn_out = grad_h2

    grad_ln1_out, grads_attn = multi_head_attention_backward(cache_attn, grad_attn_out)
    grads[:attn] = grads_attn

    grad_h_ln, grad_ln1_gamma, grad_ln1_beta = layer_norm_backward(cache_ln1, grad_ln1_out)
    grads[:ln1_gamma] = grad_ln1_gamma; grads[:ln1_beta] = grad_ln1_beta

    grad_h = grad_h_res .+ grad_h_ln

    (grad_h, grads)
end

# ===================================================================
# Full GPT Model
# ===================================================================

mutable struct GPT
    cfg::GPTConfig
    E_tok::Matrix{Float64}     # (D, V) token embeddings
    pe::Matrix{Float64}        # (D, T) sinusoidal positional encoding (not learnable)
    blocks::Vector{TransformerBlock}
    ln_f_gamma::Vector{Float64}  # final layer norm
    ln_f_beta::Vector{Float64}
    W_out::Matrix{Float64}     # (V, D) output projection
    b_out::Vector{Float64}     # (V,)
end

function init_gpt(cfg::GPTConfig)
    D = cfg.d_model; V = cfg.vocab_size; T = cfg.seq_len

    E_tok = randn(D, V) .* 0.02
    pe = sinusoidal_pe(T, D)
    blocks = [init_transformer_block(cfg) for _ in 1:cfg.n_layers]
    ln_f_gamma = ones(D)
    ln_f_beta = zeros(D)
    W_out = xavier_init(D, V)
    b_out = zeros(V)

    GPT(cfg, E_tok, pe, blocks, ln_f_gamma, ln_f_beta, W_out, b_out)
end

function gpt_forward(model::GPT, token_ids::Matrix{Int})
    D = model.cfg.d_model
    T, B = size(token_ids)

    # Token embedding + positional encoding
    h, cache_emb = embed_forward(model.E_tok, token_ids)
    h .+= model.pe[:, 1:T]  # broadcast PE over batch

    # Transformer blocks
    mask = causal_mask(T)
    block_caches = []
    for (i, block) in enumerate(model.blocks)
        h, cache_blk = transformer_block_forward(block, h, mask)
        push!(block_caches, cache_blk)
    end

    # Final layer norm
    h, cache_ln_f = layer_norm_forward(h, model.ln_f_gamma, model.ln_f_beta)

    # Output projection to logits
    logits, cache_out = linear_forward(model.W_out, model.b_out, h)

    cache = (model, cache_emb, block_caches, cache_ln_f, cache_out, token_ids)
    (logits, cache)
end

function gpt_backward(model::GPT, cache, grad_logits::Array{Float64,3})
    _, cache_emb, block_caches, cache_ln_f, cache_out, token_ids = cache

    grads = Dict{Symbol, Any}()

    # Output projection
    grad_h, grad_W_out, grad_b_out = linear_backward(cache_out, grad_logits)
    grads[:W_out] = grad_W_out
    grads[:b_out] = grad_b_out

    # Final layer norm
    grad_h, grad_ln_f_gamma, grad_ln_f_beta = layer_norm_backward(cache_ln_f, grad_h)
    grads[:ln_f_gamma] = grad_ln_f_gamma
    grads[:ln_f_beta] = grad_ln_f_beta

    # Transformer blocks (reverse order)
    block_grads = []
    for i in length(model.blocks):-1:1
        grad_h, blk_grads = transformer_block_backward(block_caches[i], grad_h)
        pushfirst!(block_grads, blk_grads)
    end
    grads[:blocks] = block_grads

    # Embedding
    grad_E_tok = embed_backward(cache_emb, grad_h)
    grads[:E_tok] = grad_E_tok

    grads
end

# ===================================================================
# SGD update (in-place)
# ===================================================================

function sgd_update!(model::GPT, grads::Dict{Symbol,Any}, lr::Float64)
    # Token embedding
    model.E_tok .-= lr .* grads[:E_tok]

    # Final layer norm
    model.ln_f_gamma .-= lr .* grads[:ln_f_gamma]
    model.ln_f_beta  .-= lr .* grads[:ln_f_beta]

    # Output projection
    model.W_out .-= lr .* grads[:W_out]
    model.b_out .-= lr .* grads[:b_out]

    # Blocks
    for (i, blk_grads) in enumerate(grads[:blocks])
        block = model.blocks[i]
        attn_g = blk_grads[:attn]

        # Attention params
        block.attn.Wq .-= lr .* attn_g.Wq
        block.attn.bq .-= lr .* attn_g.bq
        block.attn.Wk .-= lr .* attn_g.Wk
        block.attn.bk .-= lr .* attn_g.bk
        block.attn.Wv .-= lr .* attn_g.Wv
        block.attn.bv .-= lr .* attn_g.bv
        block.attn.Wo .-= lr .* attn_g.Wo
        block.attn.bo .-= lr .* attn_g.bo

        # LN1
        block.ln1_gamma .-= lr .* blk_grads[:ln1_gamma]
        block.ln1_beta  .-= lr .* blk_grads[:ln1_beta]

        # FFN
        block.W1 .-= lr .* blk_grads[:W1]
        block.b1 .-= lr .* blk_grads[:b1]
        block.W2 .-= lr .* blk_grads[:W2]
        block.b2 .-= lr .* blk_grads[:b2]

        # LN2
        block.ln2_gamma .-= lr .* blk_grads[:ln2_gamma]
        block.ln2_beta  .-= lr .* blk_grads[:ln2_beta]
    end
end

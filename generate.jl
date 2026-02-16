# generate.jl -- Autoregressive text generation (forward pass only)
#
# Three modes: greedy (argmax), temperature sampling, top-k sampling.
# Depends on: config.jl, tensor_ops.jl, model.jl

"""
    generate(model, context, max_new_tokens; temperature=1.0, top_k=0) -> Vector{Int}

Autoregressive generation. Returns the full sequence (context + generated).
"""
function generate(model::GPT, context::Vector{Int}, max_new_tokens::Int;
                  temperature::Float64=1.0, top_k::Int=0)
    seq = copy(context)
    T_max = model.cfg.seq_len

    for _ in 1:max_new_tokens
        # Truncate to last seq_len tokens if needed
        ctx = seq[max(1, end - T_max + 1):end]
        T_ctx = length(ctx)

        # Forward pass (batch size 1)
        tokens = reshape(ctx, T_ctx, 1)
        logits, _ = gpt_forward(model, tokens)

        # Take logits at last position
        logits_last = logits[:, end, 1]  # (V,)

        next_token = sample_token(logits_last; temperature=temperature, top_k=top_k)
        push!(seq, next_token)
    end

    seq
end

"""
    sample_token(logits; temperature=1.0, top_k=0) -> Int

Sample a single token from logits.
- temperature=0.0 -> greedy (argmax)
- top_k > 0 -> zero out all but top k logits before softmax
"""
function sample_token(logits::Vector{Float64}; temperature::Float64=1.0, top_k::Int=0)
    if temperature == 0.0 || temperature < 1e-10
        return argmax(logits)
    end

    scaled = logits ./ temperature

    # Top-k filtering
    if top_k > 0 && top_k < length(scaled)
        threshold = sort(scaled, rev=true)[top_k]
        scaled[scaled .< threshold] .= -Inf
    end

    # Softmax
    probs, _ = softmax_forward(scaled)

    # Sample from categorical distribution
    r = rand()
    cumsum_p = 0.0
    for i in eachindex(probs)
        cumsum_p += probs[i]
        if r <= cumsum_p
            return i
        end
    end
    return length(probs)  # fallback
end

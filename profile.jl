# profile.jl -- Benchmark key operations for PicoGPT.jl
#
# Usage: julia profile.jl
# No external deps -- uses stdlib @elapsed / @allocated only.

using Printf

include("config.jl")
include("gradcheck.jl")
include("tensor_ops.jl")
include("attention.jl")
include("model.jl")
include("data.jl")
include("train.jl")
include("generate.jl")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function bench(name::String, f; warmup::Int=2, trials::Int=5)
    # warmup
    for _ in 1:warmup
        f()
    end
    times = Float64[]
    allocs = Int[]
    for _ in 1:trials
        a = @allocated t = @elapsed f()
        push!(times, t)
        push!(allocs, a)
    end
    med_t = sort(times)[div(trials, 2) + 1]
    med_a = sort(allocs)[div(trials, 2) + 1]
    @printf("  %-45s  %8.3f ms  %8.2f MB\n", name, med_t * 1000, med_a / 1e6)
    (med_t, med_a)
end

# ---------------------------------------------------------------------------
# Config (matches run.jl defaults)
# ---------------------------------------------------------------------------

D = 128; T = 256; B = 32; H = 4; V = 65; d_ff = 4 * D
cfg = GPTConfig(vocab_size=V, d_model=D, n_heads=H, n_layers=4, seq_len=T, d_ff=d_ff)

println("="^72)
printstyled("PicoGPT.jl Profiling  (D=$D, T=$T, B=$B, H=$H, V=$V)\n", bold=true)
println("="^72)

# ---------------------------------------------------------------------------
# 1. linear_forward / linear_backward at various sizes
# ---------------------------------------------------------------------------

printstyled("\n── Linear Forward/Backward ──\n", bold=true)

for (D_in, D_out, label) in [(D, D, "128→128 (QKV proj)"),
                               (D, d_ff, "128→512 (FFN up)"),
                               (d_ff, D, "512→128 (FFN down)"),
                               (D, V, "128→65 (output proj)")]
    W = randn(D_out, D_in)
    b = zeros(D_out)
    x = randn(D_in, T, B)
    bench("linear_fwd  $label", () -> linear_forward(W, b, x))
    y, cache = linear_forward(W, b, x)
    gy = randn(size(y)...)
    bench("linear_bwd  $label", () -> linear_backward(cache, gy))
end

# ---------------------------------------------------------------------------
# 2. Softmax (single row vs full attention matrix)
# ---------------------------------------------------------------------------

printstyled("\n── Softmax ──\n", bold=true)

x_sm = randn(T)
bench("softmax_forward  single row (T=$T)", () -> softmax_forward(x_sm))

x_sm2 = randn(T, T)
bench("softmax_forward  full matrix (T×T)", () -> begin
    for i in 1:T
        softmax_forward(x_sm2[i, :])
    end
end)

# ---------------------------------------------------------------------------
# 3. Single-head and multi-head attention
# ---------------------------------------------------------------------------

printstyled("\n── Attention ──\n", bold=true)

Dh = div(D, H)
Q = randn(Dh, T, B); K = randn(Dh, T, B); Vmat = randn(Dh, T, B)
mask = causal_mask(T)

bench("single_head_attn fwd  (Dh=$Dh, T=$T, B=$B)", () -> single_head_attention_forward(Q, K, Vmat, mask))

out_sh, cache_sh = single_head_attention_forward(Q, K, Vmat, mask)
g_sh = randn(size(out_sh)...)
bench("single_head_attn bwd  (Dh=$Dh, T=$T, B=$B)", () -> single_head_attention_backward(cache_sh, g_sh))

params = init_attention_params(cfg)
x_mha = randn(D, T, B)
bench("multi_head_attn fwd   (D=$D, T=$T, B=$B, H=$H)", () -> multi_head_attention_forward(params, x_mha, mask))

out_mha, cache_mha = multi_head_attention_forward(params, x_mha, mask)
g_mha = randn(size(out_mha)...)
bench("multi_head_attn bwd   (D=$D, T=$T, B=$B, H=$H)", () -> multi_head_attention_backward(cache_mha, g_mha))

# ---------------------------------------------------------------------------
# 4. Embed
# ---------------------------------------------------------------------------

printstyled("\n── Embedding ──\n", bold=true)

E = randn(D, V)
ids = rand(1:V, T, B)
bench("embed_forward    (D=$D, T=$T, B=$B, V=$V)", () -> embed_forward(E, ids))
h_e, cache_e = embed_forward(E, ids)
g_e = randn(size(h_e)...)
bench("embed_backward   (D=$D, T=$T, B=$B, V=$V)", () -> embed_backward(cache_e, g_e))

# ---------------------------------------------------------------------------
# 5. Full training step (1 batch)
# ---------------------------------------------------------------------------

printstyled("\n── Full Training Step ──\n", bold=true)

model = init_gpt(cfg)
tokens = rand(1:V, T, B)
targets = rand(1:V, T, B)
cfg_t = TrainConfig()

bench("full forward pass", () -> begin
    gpt_forward(model, tokens)
end)

bench("full fwd + bwd + clip + update", () -> begin
    logits, cache = gpt_forward(model, tokens)
    loss, cache_ce = cross_entropy_forward(logits, targets)
    grad_logits = cross_entropy_backward(cache_ce)
    grads = gpt_backward(model, cache, grad_logits)
    clip_gradients!(grads, cfg_t.max_grad_norm)
    sgd_update!(model, grads, cfg_t.learning_rate)
end)

println("\n", "="^72)
println("Done.")

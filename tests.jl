# tests.jl -- TDD test suite for PicoGPT.jl (Tiers 1-6)
#
# Minimal harness: no Test stdlib. Just check(name, condition).
# Run: julia tests.jl

using Printf
using Statistics: mean

include("config.jl")
include("gradcheck.jl")
include("tensor_ops.jl")
include("attention.jl")

# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

const TEST_RESULTS = Pair{String,Bool}[]

function check(name::String, condition::Bool)
    push!(TEST_RESULTS, name => condition)
    if condition
        printstyled("  PASS  ", color=:green, bold=true)
    else
        printstyled("  FAIL  ", color=:red, bold=true)
    end
    println(name)
    condition
end

function test_summary()
    n = length(TEST_RESULTS)
    passed = count(last, TEST_RESULTS)
    failed = n - passed
    println("\n", "="^60)
    printstyled("Results: $passed/$n passed", color = failed == 0 ? :green : :red, bold=true)
    println()
    if failed > 0
        println("Failures:")
        for (name, ok) in TEST_RESULTS
            ok || println("  - $name")
        end
    end
    println("="^60)
    failed == 0
end

# ===================================================================
# Tier 1 -- Tensor ops
# ===================================================================

function tier1_tests()
    printstyled("\n── Tier 1: Tensor Ops ──\n", bold=true)

    # --- softmax forward ---
    x = randn(5, 3)  # (D, B)
    s, _ = softmax_forward(x)
    check("softmax: all positive",        all(s .> 0))
    check("softmax: cols sum to 1",       all(abs.(sum(s, dims=1) .- 1.0) .< 1e-6))
    # shift invariance
    s2, _ = softmax_forward(x .+ 42.0)
    check("softmax: shift invariant",     maximum(abs.(s .- s2)) < 1e-6)

    # --- softmax backward gradcheck ---
    # Use a weighted sum as loss so gradients are non-trivial
    # (sum(softmax(x)) = 1 always, so its gradient is ~0 and gradcheck is unstable)
    x_small = randn(4, 2)
    w_sm = randn(4, 2)  # random weights for a non-trivial loss
    function sm_loss(z)
        s, _ = softmax_forward(z)
        sum(s .* w_sm)
    end
    s, cache_sm = softmax_forward(copy(x_small))
    grad_s = w_sm  # dL/ds = w
    grad_x = softmax_backward(cache_sm, grad_s)
    ok, err = check_gradient(sm_loss, grad_x, copy(x_small))
    check("softmax backward: gradcheck (err=$( @sprintf("%.2e", err) ))", ok)

    # --- layer norm forward ---
    x_ln = randn(8, 4, 2)  # (D, T, B)
    gamma = ones(8)
    beta  = zeros(8)
    y_ln, _ = layer_norm_forward(x_ln, gamma, beta)
    mu  = mean(y_ln, dims=1)
    var_ = mean((y_ln .- mu).^2, dims=1)
    check("layer_norm: mean ≈ 0",   maximum(abs.(mu))  < 1e-5)
    check("layer_norm: var ≈ 1",    maximum(abs.(var_ .- 1.0)) < 1e-4)

    # --- layer norm backward gradcheck ---
    D_ln = 4; T_ln = 3; B_ln = 2
    x_ln2 = randn(D_ln, T_ln, B_ln)
    g_ln  = randn(D_ln) .* 0.5 .+ 1.0
    b_ln  = randn(D_ln) .* 0.1
    function ln_loss(z)
        y, _ = layer_norm_forward(z, g_ln, b_ln)
        sum(y)
    end
    y_ln2, cache_ln = layer_norm_forward(copy(x_ln2), g_ln, b_ln)
    grad_y_ln = ones(size(y_ln2))
    grad_x_ln, _, _ = layer_norm_backward(cache_ln, grad_y_ln)
    ok, err = check_gradient(ln_loss, grad_x_ln, copy(x_ln2))
    check("layer_norm backward: gradcheck x (err=$( @sprintf("%.2e", err) ))", ok)

    # --- relu forward ---
    x_r = randn(5, 3)
    y_r, _ = relu_forward(x_r)
    check("relu: matches max(0,x)", maximum(abs.(y_r .- max.(0, x_r))) < 1e-10)

    # --- relu backward gradcheck ---
    x_r2 = randn(4, 3) .* 2.0
    # avoid exact zeros for finite diff stability
    x_r2[abs.(x_r2) .< 0.1] .= 0.5
    function relu_loss(z)
        y, _ = relu_forward(z)
        sum(y)
    end
    _, cache_r = relu_forward(copy(x_r2))
    grad_r = relu_backward(cache_r, ones(size(x_r2)))
    ok, err = check_gradient(relu_loss, grad_r, copy(x_r2))
    check("relu backward: gradcheck (err=$( @sprintf("%.2e", err) ))", ok)

    # --- cross-entropy forward ---
    V = 5; T_ce = 3; B_ce = 2
    # uniform logits -> loss = -log(1/V) = log(V)
    logits_uni = zeros(V, T_ce, B_ce)
    targets_ce = rand(1:V, T_ce, B_ce)
    loss_uni, _ = cross_entropy_forward(logits_uni, targets_ce)
    check("cross_entropy: uniform ≈ log(V)",  abs(loss_uni - log(V)) < 1e-5)

    # perfect prediction -> loss ≈ 0
    logits_perf = fill(-100.0, V, T_ce, B_ce)
    for b in 1:B_ce, t in 1:T_ce
        logits_perf[targets_ce[t,b], t, b] = 100.0
    end
    loss_perf, _ = cross_entropy_forward(logits_perf, targets_ce)
    check("cross_entropy: perfect ≈ 0",  loss_perf < 1e-5)

    # --- cross-entropy backward gradcheck ---
    logits_gc = randn(V, T_ce, B_ce) * 0.5
    targets_gc = rand(1:V, T_ce, B_ce)
    function ce_loss(z)
        l, _ = cross_entropy_forward(z, targets_gc)
        l
    end
    _, cache_ce = cross_entropy_forward(copy(logits_gc), targets_gc)
    grad_ce = cross_entropy_backward(cache_ce)
    ok, err = check_gradient(ce_loss, grad_ce, copy(logits_gc))
    check("cross_entropy backward: gradcheck (err=$( @sprintf("%.2e", err) ))", ok)
end

# ===================================================================
# Tier 2 -- Attention  (added later)
# ===================================================================

function tier2_tests()
    printstyled("\n── Tier 2: Attention ──\n", bold=true)

    D = 8; T = 4; H = 2; B = 2; Dh = div(D, H)

    # --- causal mask: zero above diagonal ---
    mask = causal_mask(T)
    check("causal_mask: upper triangle is -Inf",
          all(mask[i, j] == -Inf for i in 1:T for j in 1:T if j > i))
    check("causal_mask: diagonal and below is 0",
          all(mask[i, j] == 0.0 for i in 1:T for j in 1:T if j <= i))

    # --- single-head attention: Q=K => uniform-ish weights ---
    Q = randn(Dh, T, B)
    V_mat = randn(Dh, T, B)
    out_sh, cache_sh = single_head_attention_forward(Q, Q, V_mat, mask)
    check("single_head_attention: output shape", size(out_sh) == (Dh, T, B))
    # with Q=K the attention pattern is symmetric / close to uniform in each causal row

    # --- multi-head attention: output shape ---
    cfg = GPTConfig(vocab_size=10, d_model=D, n_heads=H, n_layers=1, seq_len=T)
    params = init_attention_params(cfg)
    x_mha = randn(D, T, B)
    out_mha, cache_mha = multi_head_attention_forward(params, x_mha, mask)
    check("multi_head_attention: output shape (D,T,B)",
          size(out_mha) == (D, T, B))

    # --- multi-head with H=1 matches single-head pipeline ---
    cfg1 = GPTConfig(vocab_size=10, d_model=D, n_heads=1, n_layers=1, seq_len=T)
    params1 = init_attention_params(cfg1)
    out1, _ = multi_head_attention_forward(params1, x_mha, mask)
    check("multi_head n_heads=1: output shape",  size(out1) == (D, T, B))

    # --- forward-backward round trip gradcheck on Wq ---
    x_gc = randn(D, T, B)
    params_gc = init_attention_params(cfg)
    function mha_loss_wq(Wq_flat)
        p = copy_attention_params(params_gc)
        p.Wq .= reshape(Wq_flat, size(p.Wq))
        out, _ = multi_head_attention_forward(p, x_gc, mask)
        sum(out)
    end
    out_gc, cache_gc = multi_head_attention_forward(params_gc, x_gc, mask)
    grad_out = ones(size(out_gc))
    grad_x_gc, grads_gc = multi_head_attention_backward(cache_gc, grad_out)
    ok, err = check_gradient(mha_loss_wq, vec(grads_gc.Wq), vec(copy(params_gc.Wq)))
    check("multi_head backward: gradcheck Wq (err=$( @sprintf("%.2e", err) ))", ok)

    # --- gradcheck on Wv ---
    function mha_loss_wv(Wv_flat)
        p = copy_attention_params(params_gc)
        p.Wv .= reshape(Wv_flat, size(p.Wv))
        out, _ = multi_head_attention_forward(p, x_gc, mask)
        sum(out)
    end
    ok, err = check_gradient(mha_loss_wv, vec(grads_gc.Wv), vec(copy(params_gc.Wv)))
    check("multi_head backward: gradcheck Wv (err=$( @sprintf("%.2e", err) ))", ok)
end

# ===================================================================
# Tier 3 -- Model  (added later)
# ===================================================================

function tier3_tests()
    printstyled("\n── Tier 3: Model ──\n", bold=true)
    # placeholder -- will be filled by pico-mrg
end

# ===================================================================
# Tier 4 -- Data pipeline  (added later)
# ===================================================================

function tier4_tests()
    printstyled("\n── Tier 4: Data Pipeline ──\n", bold=true)
    # placeholder -- will be filled by pico-b8h
end

# ===================================================================
# Tier 5 -- Training loop  (added later)
# ===================================================================

function tier5_tests()
    printstyled("\n── Tier 5: Training Loop ──\n", bold=true)
    # placeholder
end

# ===================================================================
# Tier 6 -- Generation  (added later)
# ===================================================================

function tier6_tests()
    printstyled("\n── Tier 6: Generation ──\n", bold=true)
    # placeholder
end

# ===================================================================
# Entry point
# ===================================================================

function run_all_tests()
    empty!(TEST_RESULTS)
    println("="^60)
    printstyled("PicoGPT.jl Test Suite\n", bold=true)
    println("="^60)

    tier1_tests()
    tier2_tests()
    tier3_tests()
    tier4_tests()
    tier5_tests()
    tier6_tests()

    all_passed = test_summary()
    exit(all_passed ? 0 : 1)
end

# Only run when executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_tests()
end

# PicoGPT.jl — Detailed Code Reference

Function-by-function documentation with exact code locations, tensor shapes, and gradient derivations for every module in PicoGPT.jl.

---

## Table of Contents

- [config.jl — Hyperparameter structs](#configjl)
- [gradcheck.jl — Finite-difference verification](#gradcheckjl)
- [tensor_ops.jl — Primitive operations](#tensor_opsjl)
  - [Weight initialization](#weight-initialization)
  - [Embedding lookup](#embedding-lookup)
  - [Sinusoidal positional encoding](#sinusoidal-positional-encoding)
  - [Softmax](#softmax)
  - [Layer normalization](#layer-normalization)
  - [ReLU](#relu)
  - [Cross-entropy loss](#cross-entropy-loss)
- [attention.jl — Self-attention](#attentionjl)
  - [AttentionParams](#attentionparams)
  - [Linear projection](#linear-projection)
  - [Causal mask](#causal-mask)
  - [Single-head attention](#single-head-attention)
  - [Multi-head attention](#multi-head-attention)
- [model.jl — Transformer blocks and GPT](#modeljl)
  - [TransformerBlock](#transformerblock)
  - [GPT](#gpt)
  - [SGD update](#sgd-update)
- [train.jl — Training loop](#trainjl)
  - [Learning rate schedule](#learning-rate-schedule)
  - [Gradient clipping](#gradient-clipping)
  - [Training loop](#training-loop)
- [generate.jl — Text generation](#generatejl)
- [data.jl — Data pipeline](#datajl)
- [tests.jl — Test suite](#testsjl)

---

<a id="configjl"></a>
## config.jl — Hyperparameter structs

**Location:** [`config.jl`](../config.jl)

### `GPTConfig` ([line 11](../config.jl#L11))

```julia
struct GPTConfig
    vocab_size::Int   # V — number of unique tokens (data-determined)
    d_model::Int      # D — embedding / hidden dimension
    n_heads::Int      # H — number of attention heads
    n_layers::Int     # number of transformer blocks
    seq_len::Int      # T — context window length
    d_ff::Int         # feedforward hidden dimension (typically 4 × d_model)
end
```

**Default constructor** ([line 25](../config.jl#L25)): Accepts keyword arguments.  Computes `d_ff = 4 * d_model` by default.  Asserts `d_model % n_heads == 0`.

**Defaults:** `vocab_size=65`, `d_model=128`, `n_heads=4`, `n_layers=4`, `seq_len=256`, `d_ff=512`.

### `TrainConfig` ([line 38](../config.jl#L38))

```julia
struct TrainConfig
    batch_size::Int           # 32
    learning_rate::Float64    # 1e-3 (η_max)
    lr_min::Float64           # 1e-5 (η_min for cosine decay)
    warmup_steps::Int         # 100
    n_epochs::Int             # 20
    max_grad_norm::Float64    # 1.0
    eval_interval::Int        # 1 (evaluate every N epochs)
end
```

---

<a id="gradcheckjl"></a>
## gradcheck.jl — Finite-difference verification

**Location:** [`gradcheck.jl`](../gradcheck.jl)

### `numerical_gradient(f, x; eps=1e-5)` ([line 14](../gradcheck.jl#L14))

Central-difference approximation:

```
∂f/∂x_i ≈ (f(x + ε·e_i) − f(x − ε·e_i)) / (2ε)
```

**Input:** `f` — scalar-valued function, `x` — `AbstractArray{Float64}`
**Output:** Array of same shape as `x`
**Complexity:** 2N function evaluations for N elements

### `relative_error(a, b)` ([line 33](../gradcheck.jl#L33))

```
max_i |a_i − b_i| / max(|a_i|, |b_i|, 10⁻⁸)
```

The `10⁻⁸` floor prevents division by zero when both values are near zero.

### `check_gradient(f_loss, analytical_grad, x; eps, tol)` ([line 45](../gradcheck.jl#L45))

Compares analytical gradient against `numerical_gradient(f_loss, x)`.

**Returns:** `(pass::Bool, rel_err::Float64)` where `pass = (rel_err < tol)`.

**Threshold:** `tol = 1e-5`.  All PicoGPT gradients achieve errors below `1e-8`.

---

<a id="tensor_opsjl"></a>
## tensor_ops.jl — Primitive operations

**Location:** [`tensor_ops.jl`](../tensor_ops.jl)

### Weight initialization

#### `xavier_init(fan_in, fan_out)` ([line 28](../tensor_ops.jl#L28))

```julia
function xavier_init(fan_in::Int, fan_out::Int)
    std = sqrt(2.0 / (fan_in + fan_out))
    randn(fan_out, fan_in) .* std
end
```

**Returns:** `Matrix{Float64}` of shape `(fan_out, fan_in)`

**Derivation** ([line 18](../tensor_ops.jl#L18)):

For `y = Wx` with `x ∈ R^{fan_in}`:
```
Var[y_j] = fan_in · Var[w] · Var[x]
```
Setting `Var[y] = Var[x]` in both forward and backward: `Var[w] = 2/(fan_in + fan_out)`.

**Used for:** `Wq`, `Wk`, `Wv` (attention projections), `W1` (FFN first layer), `W_out` (output projection).

#### `gpt2_init(fan_in, fan_out; scale=1.0)` ([line 39](../tensor_ops.jl#L39))

```julia
function gpt2_init(fan_in::Int, fan_out::Int; scale::Float64=1.0)
    randn(fan_out, fan_in) .* (0.02 * scale)
end
```

**Used for:** `Wo` (attention output) and `W2` (FFN second layer) — the residual projections.  Called with `scale = 1/√(2·n_layers)` to prevent variance accumulation across layers.

#### `zeros_init(dims...)` ([line 46](../tensor_ops.jl#L46))

All biases are initialized to zero.

---

### Embedding lookup

#### `embed_forward(E, token_ids)` ([line 58](../tensor_ops.jl#L58))

**Inputs:**
- `E::Matrix{Float64}` — shape `(D, V)`, each column is a token embedding
- `token_ids::Matrix{Int}` — shape `(T, B)`, values in `[1, V]`

**Output:** `(h, cache)` where `h` has shape `(D, T, B)`

**Operation:** Column lookup — `h[:, t, b] = E[:, token_ids[t, b]]`

#### `embed_backward(cache, grad_h)` ([line 73](../tensor_ops.jl#L73))

**Output:** `grad_E::Matrix{Float64}` of shape `(D, V)`

**Operation:** Scatter-add — accumulates incoming gradients into the looked-up rows:
```julia
grad_E[:, token_ids[t, b]] .+= grad_h[:, t, b]
```

If a token appears K times, its gradient is the sum of K contributions.

---

### Sinusoidal positional encoding

#### `sinusoidal_pe(seq_len, d_model)` ([line 98](../tensor_ops.jl#L98))

**Output:** `Matrix{Float64}` of shape `(d_model, seq_len)`

**Formula** (Vaswani et al.):
```
PE[2i+1, pos+1] = sin(pos / 10000^(2i/D))
PE[2i+2, pos+1] = cos(pos / 10000^(2i/D))
```

No learnable parameters.  No backward pass needed.

The frequencies span from `1` (dimension 0) to `1/10000` (dimension D−1), creating a multi-scale position signal.

---

### Softmax

#### `softmax_forward(x)` ([line 133](../tensor_ops.jl#L133))

```julia
function softmax_forward(x::AbstractArray{Float64})
    x_max = maximum(x, dims=1)
    ex = exp.(x .- x_max)          # subtract max for numerical stability
    s = ex ./ sum(ex, dims=1)
    (s, (s,))
end
```

**Input:** Any array. Softmax computed along dimension 1.
**Output:** `(s, cache)` where `s` has same shape as `x`, `cache = (s,)`.

**Numerical stability:** Subtracting `max(x)` prevents overflow in `exp`.  The result is mathematically identical: `exp(x_i − c) / Σ exp(x_j − c) = exp(x_i) / Σ exp(x_j)`.

#### `softmax_backward(cache, grad_s)` ([line 143](../tensor_ops.jl#L143))

```julia
function softmax_backward(cache, grad_s)
    s, = cache
    dot = sum(s .* grad_s, dims=1)
    s .* (grad_s .- dot)
end
```

**Derivation** ([line 120](../tensor_ops.jl#L120)):

```
∂s_i/∂x_k = s_i(δ_{ik} − s_k)

∂L/∂x_k = Σ_i ∂L/∂s_i · s_i(δ_{ik} − s_k)
         = ∂L/∂s_k · s_k − s_k · Σ_i s_i · ∂L/∂s_i
         = s_k(∂L/∂s_k − Σ_j s_j · ∂L/∂s_j)
```

The `dot` term `Σ_j s_j · ∂L/∂s_j` is computed once per column and broadcast.

---

### Layer normalization

#### `layer_norm_forward(x, gamma, beta; eps=1e-5)` ([line 170](../tensor_ops.jl#L170))

**Inputs:**
- `x` — shape `(D, T, B)`
- `gamma`, `beta` — shape `(D,)`, learnable scale and shift

**Output:** `(y, cache)` where `y` has shape `(D, T, B)`

**Forward equations:**
```
μ       = (1/D) Σ_i x_i              computed per (t, b) position
σ²      = (1/D) Σ_i (x_i − μ)²
x̂_i     = (x_i − μ) / √(σ² + ε)
y_i     = γ_i · x̂_i + β_i
```

**Cache:** `(x, x_hat, gamma, mu, sigma_inv, D)` — all intermediates needed for backward.

#### `layer_norm_backward(cache, grad_y)` ([line 186](../tensor_ops.jl#L186))

**Output:** `(grad_x, grad_gamma, grad_beta)`

**Full derivation** ([line 154](../tensor_ops.jl#L154)):

```
∂L/∂γ     = Σ_{t,b} ∂L/∂y ⊙ x̂
∂L/∂β     = Σ_{t,b} ∂L/∂y
∂L/∂x̂     = ∂L/∂y ⊙ γ
∂L/∂σ²    = −(1/2) Σ_i ∂L/∂x̂_i · (x_i − μ) · σ⁻³
∂L/∂μ     = −σ⁻¹ Σ_i ∂L/∂x̂_i + ∂L/∂σ² · (−2/D) Σ_i(x_i − μ)
∂L/∂x_i   = ∂L/∂x̂_i · σ⁻¹ + ∂L/∂σ² · (2/D)(x_i − μ) + ∂L/∂μ · (1/D)
```

`x_i` affects the output through three paths: (1) directly via x̂, (2) through σ², (3) through μ.

---

### ReLU

#### `relu_forward(x)` ([line 217](../tensor_ops.jl#L217))

```julia
function relu_forward(x::AbstractArray{Float64})
    mask = x .> 0
    (x .* mask, (mask,))
end
```

**Cache:** `(mask,)` — Boolean array storing where `x > 0`.

#### `relu_backward(cache, grad_y)` ([line 225](../tensor_ops.jl#L225))

```julia
grad_y .* mask     # ∂L/∂x_i = ∂L/∂y_i · 𝟙[x_i > 0]
```

The gradient passes through where `x > 0` and is zeroed where `x ≤ 0`.

---

### Cross-entropy loss

#### `cross_entropy_forward(logits, targets)` ([line 249](../tensor_ops.jl#L249))

**Inputs:**
- `logits` — shape `(V, T, B)`
- `targets` — shape `(T, B)`, integers in `[1, V]`

**Output:** `(loss, cache)` where `loss` is a scalar

**Computation:** Fused softmax + negative log-likelihood:
```
probs = softmax(logits)          along dim 1
loss  = −(1/N) Σ_{t,b} log probs[target[t,b], t, b]
```
where `N = T × B`.

#### `cross_entropy_backward(cache)` ([line 269](../tensor_ops.jl#L269))

```julia
function cross_entropy_backward(cache)
    probs, targets, T, B = cache
    grad = copy(probs)
    for b in 1:B, t in 1:T
        grad[targets[t, b], t, b] -= 1.0
    end
    grad ./= (T * B)
end
```

**Output:** `grad_logits` of shape `(V, T, B)`

**The key identity:** `∂L/∂logits = (1/N)(softmax(logits) − one_hot(target))`

This is the fused softmax+CE gradient.  It avoids computing the Jacobian of softmax separately.

---

<a id="attentionjl"></a>
## attention.jl — Self-attention

**Location:** [`attention.jl`](../attention.jl)

### AttentionParams

#### `AttentionParams` struct ([line 13](../attention.jl#L13))

```julia
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
```

Mutable so that SGD can update weights in-place.

#### `init_attention_params(cfg)` ([line 36](../attention.jl#L36))

- `Wq`, `Wk`, `Wv`: Xavier init
- `Wo`: GPT-2 init with `scale = 1/√(2·n_layers)` (residual scaling)
- All biases: zeros

#### `AttentionGrads` struct ([line 25](../attention.jl#L25))

Immutable struct mirroring `AttentionParams` to hold gradients.

---

### Linear projection

#### `linear_forward(W, b, x)` ([line 63](../attention.jl#L63))

**Inputs:** `W` shape `(D_out, D_in)`, `b` shape `(D_out,)`, `x` shape `(D_in, T, B)`
**Output:** `(y, cache)` where `y` shape `(D_out, T, B)`

Applies `y[:, t, b] = W * x[:, t, b] .+ b` per position.

#### `linear_backward(cache, grad_y)` ([line 75](../attention.jl#L75))

**Gradient derivation** ([line 57](../attention.jl#L57)):

```
Forward:  y = Wx          (per position)
∂L/∂W   = Σ_{t,b} ∂L/∂y · xᵀ       (outer product, accumulated)
∂L/∂x   = Wᵀ · ∂L/∂y
∂L/∂b   = Σ_{t,b} ∂L/∂y
```

**Output:** `(grad_x, grad_W, grad_b)`

---

### Causal mask

#### `causal_mask(T)` ([line 101](../attention.jl#L101))

**Output:** `Matrix{Float64}` shape `(T, T)`.  `mask[i,j] = 0` if `j ≤ i`, `-Inf` if `j > i`.

After adding to attention scores and applying softmax, `exp(-Inf) = 0` ensures position `i` cannot attend to future position `j > i`.

---

### Single-head attention

#### `single_head_attention_forward(Q, K, V, mask)` ([line 132](../attention.jl#L132))

**Inputs:** `Q`, `K`, `V` each shape `(Dh, T, B)`, `mask` shape `(T, T)`
**Output:** `(out, cache)` where `out` shape `(Dh, T, B)`

**Steps:**
1. `scores[:,:,b] = (Q[:,:,b]' * K[:,:,b]) * (1/√Dh) + mask` — shape `(T, T, B)`
2. `attn[i,:,b] = softmax(scores[i,:,b])` — per-row softmax
3. `out[:,:,b] = V[:,:,b] * attn[:,:,b]'` — weighted sum of values

**Cache:** `(Q, K, V, attn, attn_caches, mask, scale, Dh, T, B)` — stores per-row softmax caches for backward.

#### `single_head_attention_backward(cache, grad_out)` ([line 161](../attention.jl#L161))

**Derivation** ([line 115](../attention.jl#L115)):

```
∂L/∂A[i,j,b] = Σ_d ∂L/∂out[d,i,b] · V[d,j,b]     → grad_out' * V
∂L/∂V[d,j,b] = Σ_i ∂L/∂out[d,i,b] · A[i,j,b]     → grad_out * attn
∂L/∂S         = softmax_backward(∂L/∂A)              (per row)
∂L/∂Q[d,i,b] = (1/√Dh) Σ_j ∂L/∂S[i,j,b] · K[d,j,b]  → K * grad_S' / √Dh
∂L/∂K[d,j,b] = (1/√Dh) Σ_i ∂L/∂S[i,j,b] · Q[d,i,b]  → Q * grad_S / √Dh
```

**Output:** `(grad_Q, grad_K, grad_V)`

---

### Multi-head attention

#### `multi_head_attention_forward(params, x, mask)` ([line 209](../attention.jl#L209))

**Input:** `x` shape `(D, T, B)`
**Output:** `(output, cache)` where `output` shape `(D, T, B)`

**Steps:**
1. Project: `Q = Wq * x + bq`, `K = Wk * x + bk`, `V = Wv * x + bv` — each `(D, T, B)`
2. Split each into H heads: rows `(h-1)*Dh+1 : h*Dh` → shape `(Dh, T, B)` per head
3. Run `single_head_attention_forward` per head
4. Concatenate head outputs → `(D, T, B)`
5. Output projection: `output = Wo * concat + bo`

#### `multi_head_attention_backward(cache, grad_output)` ([line 246](../attention.jl#L246))

**Steps** (reverse of forward):
1. Backward through `Wo` → `(grad_concat, grad_Wo, grad_bo)`
2. Split `grad_concat` into heads
3. `single_head_attention_backward` per head → `grad_Q_h`, `grad_K_h`, `grad_V_h`
4. Concatenate head gradients
5. Backward through `Wq`, `Wk`, `Wv` projections
6. Sum input gradients: `grad_x = grad_x_q + grad_x_k + grad_x_v` (fan-out chain rule)

**Output:** `(grad_x, AttentionGrads(...))`

---

<a id="modeljl"></a>
## model.jl — Transformer blocks and GPT

**Location:** [`model.jl`](../model.jl)

### TransformerBlock

#### `TransformerBlock` struct ([line 13](../model.jl#L13))

```julia
mutable struct TransformerBlock
    attn::AttentionParams           # multi-head attention params
    ln1_gamma::Vector{Float64}      # layer norm 1 (before attention)
    ln1_beta::Vector{Float64}
    W1::Matrix{Float64}             # (d_ff, D)  FFN first layer
    b1::Vector{Float64}             # (d_ff,)
    W2::Matrix{Float64}             # (D, d_ff)  FFN second layer
    b2::Vector{Float64}             # (D,)
    ln2_gamma::Vector{Float64}      # layer norm 2 (before FFN)
    ln2_beta::Vector{Float64}
end
```

#### `transformer_block_forward(block, h, mask)` ([line 38](../model.jl#L38))

**Input:** `h` shape `(D, T, B)`
**Output:** `(h3, cache)` where `h3` shape `(D, T, B)`

**Data flow (pre-norm residual):**
```
Sub-block 1:  ln1 → MHA → residual add
Sub-block 2:  ln2 → W1 → ReLU → W2 → residual add
```

```julia
# Sub-block 1
ln1_out = layer_norm(h)
attn_out = multi_head_attention(ln1_out)
h2 = h + attn_out                              # residual

# Sub-block 2
ln2_out = layer_norm(h2)
ff2 = W2 * relu(W1 * ln2_out + b1) + b2
h3 = h2 + ff2                                  # residual
```

#### `transformer_block_backward(cache, grad_h3)` ([line 56](../model.jl#L56))

**Output:** `(grad_h, grads::Dict)` where `grads` contains gradients for all block parameters.

Reverse of forward.  Key insight: at each residual connection, the gradient splits into two identical copies (one flows through the skip, one through the sub-block).

---

### GPT

#### `GPT` struct ([line 96](../model.jl#L96))

```julia
mutable struct GPT
    cfg::GPTConfig
    E_tok::Matrix{Float64}             # (D, V)
    pe::Matrix{Float64}                # (D, T)   fixed sinusoidal
    blocks::Vector{TransformerBlock}   # n_layers blocks
    ln_f_gamma::Vector{Float64}        # final layer norm
    ln_f_beta::Vector{Float64}
    W_out::Matrix{Float64}             # (V, D)
    b_out::Vector{Float64}             # (V,)
end
```

#### `init_gpt(cfg)` ([line 107](../model.jl#L107))

- `E_tok`: `N(0, 0.02)` — GPT-2 style
- `pe`: `sinusoidal_pe(T, D)` — fixed, non-learnable
- `blocks`: `n_layers` transformer blocks
- `W_out`: Xavier init
- All biases: zeros

#### `gpt_forward(model, token_ids)` ([line 121](../model.jl#L121))

**Input:** `token_ids` shape `(T, B)`, integers in `[1, V]`
**Output:** `(logits, cache)` where `logits` shape `(V, T, B)`

**Data flow:**
```
embed(token_ids) + PE[:, 1:T]  →  block_1 → ... → block_N  →  layer_norm  →  W_out * h + b_out
```

#### `gpt_backward(model, cache, grad_logits)` ([line 147](../model.jl#L147))

**Input:** `grad_logits` shape `(V, T, B)` — from `cross_entropy_backward`
**Output:** `grads::Dict{Symbol, Any}` with keys `:W_out`, `:b_out`, `:ln_f_gamma`, `:ln_f_beta`, `:blocks`, `:E_tok`

Blocks are processed in **reverse order** (last-to-first).

---

### SGD update

#### `sgd_update!(model, grads, lr)` ([line 181](../model.jl#L181))

In-place parameter update: `θ ← θ − η · ∂L/∂θ` for all parameters.

Walks the model structure: `E_tok`, `ln_f`, `W_out`, then per-block: attention params (Wq, bq, Wk, bk, Wv, bv, Wo, bo), ln1, FFN (W1, b1, W2, b2), ln2.

---

<a id="trainjl"></a>
## train.jl — Training loop

**Location:** [`train.jl`](../train.jl)

### Learning rate schedule

#### `lr_schedule(step, cfg, total_steps)` ([line 13](../train.jl#L13))

**Linear warmup** (step < warmup_steps):
```
η(t) = η_max · (t / warmup_steps)
```

**Cosine decay** (step >= warmup_steps):
```
progress = (t − warmup_steps) / (total_steps − warmup_steps)
η(t) = η_min + 0.5 · (η_max − η_min) · (1 + cos(π · progress))
```

At `t = 0`: `η = 0`.  At `t = warmup_steps`: `η = η_max`.  At `t = total_steps`: `η = η_min`.

---

### Gradient clipping

#### `clip_gradients!(grads, max_norm)` ([line 56](../train.jl#L56))

Global norm clipping:
```
‖g‖₂ = √(Σ g_i²)     across ALL parameters
g ← g · min(1, c / ‖g‖₂)
```

Uses `collect_grad_arrays` ([line 28](../train.jl#L28)) to recursively flatten the nested `Dict`/`AttentionGrads`/`Vector` structure into a flat list of arrays.

`scale_grads!` ([line 65](../train.jl#L65)) walks the same structure and multiplies every gradient by the scale factor, constructing new `AttentionGrads` objects (since they are immutable).

---

### Training loop

#### `train_epoch!(model, batches, cfg, step, total_steps)` ([line 87](../train.jl#L87))

Per batch: forward → loss → backward → clip → SGD update.

**Output:** `(avg_loss, step)`

#### `evaluate(model, batches)` ([line 110](../train.jl#L110))

Forward-only pass over validation batches.  Returns average loss.

#### `train!(model, train_batches, val_batches, cfg)` ([line 120](../train.jl#L120))

Outer loop: `n_epochs` iterations of `train_epoch!` + `evaluate`.  Prints train/val loss per epoch.

---

<a id="generatejl"></a>
## generate.jl — Text generation

**Location:** [`generate.jl`](../generate.jl)

#### `generate(model, context, max_new_tokens; temperature, top_k)` ([line 11](../generate.jl#L11))

**Inputs:**
- `context::Vector{Int}` — seed token IDs
- `max_new_tokens::Int` — number of tokens to generate
- `temperature::Float64` — sampling temperature (0 = greedy)
- `top_k::Int` — if > 0, only sample from top k tokens

**Output:** `Vector{Int}` of length `length(context) + max_new_tokens`

**Algorithm:**
1. Truncate context to last `seq_len` tokens
2. Forward pass (batch size 1)
3. Take logits at last position
4. Sample next token
5. Append and repeat

#### `sample_token(logits; temperature, top_k)` ([line 42](../generate.jl#L42))

Three modes:
- **Greedy** (`temperature ≈ 0`): `argmax(logits)`
- **Temperature**: `softmax(logits / T)` then sample from categorical
- **Top-k**: Zero out all logits below the k-th largest, then temperature sample

Sampling uses inverse CDF: draw uniform `r`, accumulate probabilities until `cumsum ≥ r`.

---

<a id="datajl"></a>
## data.jl — Data pipeline

**Location:** [`data.jl`](../data.jl)

#### `download_shakespeare(path="input.txt")` ([line 8](../data.jl#L8))

Downloads from Karpathy's char-rnn repo if not present.  Returns text as `String`.

#### `build_vocab(text)` ([line 19](../data.jl#L19))

**Output:** `(char_to_idx::Dict{Char,Int}, idx_to_char::Dict{Int,Char})`

Characters sorted alphabetically, 1-indexed.  Typical Shakespeare vocab: 65 characters.

#### `encode(text, char_to_idx)` / `decode(ids, idx_to_char)` ([lines 28, 34](../data.jl#L28))

Tokenization round-trip.  `encode("hello", c2i)` → `[i₁, i₂, i₃, i₄, i₅]`.

#### `train_val_split(data; split_frac=0.9)` ([line 41](../data.jl#L41))

Sequential split.  No shuffling — avoids context-window leakage.

#### `make_batches(data, seq_len, batch_size)` ([line 51](../data.jl#L51))

**Output:** `Vector{Tuple{Matrix{Int}, Matrix{Int}}}` — list of `(input, target)` pairs

Non-overlapping windows of `seq_len + 1`.  Each window yields:
- `input = window[1:seq_len]`
- `target = window[2:seq_len+1]`

Grouped into batches of shape `(seq_len, batch_size)`.  Excess windows that don't fill a complete batch are discarded.

---

<a id="testsjl"></a>
## tests.jl — Test suite

**Location:** [`tests.jl`](../tests.jl)

### Test harness

- `check(name, condition)` — prints `PASS`/`FAIL`, accumulates results
- `test_summary()` — prints final count, lists failures
- `run_all_tests()` — runs all 6 tiers, exits 0 on all-pass

### Tier summary

| Tier | Function | Tests | Key assertions |
|------|----------|-------|----------------|
| 1 | `tier1_tests()` | 12 | Softmax sums to 1, shift-invariant; LN mean≈0, var≈1; all gradchecks < 1e-5 |
| 2 | `tier2_tests()` | 7 | Causal mask shape; MHA output shape; gradcheck on Wq and Wv |
| 3 | `tier3_tests()` | 8 | Block/model shapes; all grads exist; **overfit-one-batch < 0.1** |
| 4 | `tier4_tests()` | 6 | Encode/decode round-trip; batch shapes; target offset; split coverage |
| 5 | `tier5_tests()` | 5 | SGD decreases loss; LR schedule endpoints; grad clip norm bound |
| 6 | `tier6_tests()` | 4 | Output length; greedy deterministic; top-k=1 = greedy |
| **Total** | | **42** | |

---

## References

- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017)
- Radford et al., [*Language Models are Unsupervised Multitask Learners*](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (2019)
- Jay Alammar, [*The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/)
- Andrej Karpathy, [*nanoGPT*](https://github.com/karpathy/nanoGPT)
- Glorot & Bengio, [*Understanding the difficulty of training deep feedforward neural networks*](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) (2010)

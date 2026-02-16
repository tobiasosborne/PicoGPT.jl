# Building a Transformer from Scratch

**A progressive walkthrough of PicoGPT.jl — from token embedding to text generation, with every gradient derived by hand.**

This tutorial walks through the complete architecture of a GPT-2-style transformer, as implemented in PicoGPT.jl.  We follow the same progressive-disclosure approach as Jay Alammar's [*The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/): start with the high-level picture, zoom into each component, then show the math.

If you haven't read Alammar's post, read it first.  It will give you the visual intuition; this tutorial gives you the matrix calculus and working Julia code.

> **Prerequisites:** Linear algebra (matrix multiplication, transpose, vector norms), basic calculus (chain rule, partial derivatives), and index notation.  No prior ML knowledge needed.

---

## Table of Contents

1. [The 10,000-foot view](#1-the-10000-foot-view)
2. [Token embedding and positional encoding](#2-token-embedding-and-positional-encoding)
3. [The attention mechanism](#3-the-attention-mechanism)
4. [Multi-head attention](#4-multi-head-attention)
5. [The transformer block](#5-the-transformer-block)
6. [The full GPT model](#6-the-full-gpt-model)
7. [The backward pass: manual backpropagation](#7-the-backward-pass-manual-backpropagation)
8. [Loss, optimizer, and training](#8-loss-optimizer-and-training)
9. [Text generation](#9-text-generation)
10. [Testing everything](#10-testing-everything)

---

## 1. The 10,000-foot view

A GPT model is a function that takes a sequence of tokens and predicts the next token at every position.  Here's the complete data flow with tensor shapes:

```
  "the cat sat"                          Raw text
       │
       ▼
  [47, 52, 28]                           Token indices  ∈ Z^(T)
       │
       ▼
 ┌─────────────┐
 │  E_tok       │  (D, V) = (128, 65)    Embedding matrix
 │  lookup      │
 └──────┬──────┘
        │
        ▼
   h₀ ∈ R^(D×T)                          (128, 3) — each token is now a vector
        │
   + PE ∈ R^(D×T)                         Sinusoidal position signal (fixed)
        │
        ▼
 ┌──────────────────────────────────────────────────┐
 │  Transformer Block  ×4                            │
 │                                                   │
 │   h ──┬──→ LayerNorm → Multi-Head Attention ──┐  │
 │       │                                        +  │  ← residual
 │       └────────────────────────────────────────┘  │
 │       │                                           │
 │   h ──┬──→ LayerNorm → W₁ → ReLU → W₂ ───────┐  │
 │       │                                        +  │  ← residual
 │       └────────────────────────────────────────┘  │
 └──────────────────────────────────────────────────┘
        │
        ▼
   LayerNorm                              Final normalization
        │
        ▼
 ┌─────────────┐
 │  W_out       │  (V, D) = (65, 128)    Output projection
 └──────┬──────┘
        │
        ▼
   logits ∈ R^(V×T)                       (65, 3) — score for each vocab token
        │
        ▼
   softmax → cross-entropy loss           Training: compare with target tokens
        or
   softmax → sample                       Generation: pick the next token
```

**Dimensions cheat sheet** (used throughout this tutorial):

| Symbol | Name | Default | Meaning |
|--------|------|---------|---------|
| `B` | Batch | 32 | Number of sequences processed in parallel |
| `T` | Time | 256 | Sequence length (context window) |
| `D` | Model | 128 | Embedding / hidden dimension |
| `H` | Heads | 4 | Number of attention heads |
| `Dh` | Head dim | 32 | `D / H`, dimension per head |
| `V` | Vocab | 65 | Number of unique tokens |
| `d_ff` | FFN | 512 | Feedforward hidden dimension |

In PicoGPT.jl, this is implemented in `gpt_forward` ([`model.jl:121`](../model.jl#L121)):

```julia
function gpt_forward(model::GPT, token_ids::Matrix{Int})
    T, B = size(token_ids)

    # 1. Token embedding + positional encoding
    h, cache_emb = embed_forward(model.E_tok, token_ids)
    h .+= model.pe[:, 1:T]

    # 2. N transformer blocks
    mask = causal_mask(T)
    block_caches = []
    for (i, block) in enumerate(model.blocks)
        h, cache_blk = transformer_block_forward(block, h, mask)
        push!(block_caches, cache_blk)
    end

    # 3. Final layer norm + output projection
    h, cache_ln_f = layer_norm_forward(h, model.ln_f_gamma, model.ln_f_beta)
    logits, cache_out = linear_forward(model.W_out, model.b_out, h)

    (logits, cache)
end
```

The output `logits` has shape `(V, T, B)` — for each position in each sequence, a score for every token in the vocabulary.  Softmax turns these scores into probabilities.

**The key idea:** every function returns `(result, cache)`.  The cache stores everything the backward pass needs.  This is how we do backpropagation without autodiff.

---

## 2. Token embedding and positional encoding

### 2.1 Token embedding

A neural network operates on continuous vectors, not integers.  The embedding step converts each token index into a learned vector.

An embedding matrix `E_tok` has shape `(D, V)` — each *column* is the D-dimensional vector for one token.  (Julia is column-major, so this layout makes column access contiguous.)

From [`tensor_ops.jl:58`](../tensor_ops.jl#L58):

```julia
function embed_forward(E::Matrix{Float64}, token_ids::Matrix{Int})
    D, V = size(E)
    T, B = size(token_ids)
    h = zeros(D, T, B)
    for b in 1:B, t in 1:T
        h[:, t, b] .= E[:, token_ids[t, b]]     # column lookup
    end
    (h, (E, token_ids))
end
```

This is just a table lookup: for each token index, grab the corresponding column of `E`.

**Backward pass ([`tensor_ops.jl:73`](../tensor_ops.jl#L73)):** The gradient doesn't "flow through" the lookup in the usual sense.  Instead, we scatter-add: for each token that was looked up, accumulate the incoming gradient into that token's embedding row.

```julia
function embed_backward(cache, grad_h)
    E, token_ids = cache
    grad_E = zeros(D, V)
    for b in 1:B, t in 1:T
        grad_E[:, token_ids[t, b]] .+= grad_h[:, t, b]
    end
    grad_E
end
```

If the word "the" appears 100 times, its embedding gradient is the sum of 100 incoming gradients.

### 2.2 Sinusoidal positional encoding

Self-attention is permutation-equivariant — it treats "the cat sat" and "sat the cat" identically.  Positional encoding breaks this symmetry by adding a position-dependent signal to each embedding.

Vaswani et al. (2017) use a Fourier basis:

```
PE(pos, 2i)     = sin(pos / 10000^(2i/D))
PE(pos, 2i + 1) = cos(pos / 10000^(2i/D))
```

Why sines and cosines?  For any fixed offset `k`, `PE(pos + k)` can be written as a linear function of `PE(pos)`.  This means the model can learn to attend to relative positions using linear operations.

From [`tensor_ops.jl:98`](../tensor_ops.jl#L98):

```julia
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
```

The PE matrix has shape `(D, T)` and is precomputed once.  It has no learnable parameters, so it requires no backward pass.

The combined input to the first transformer block is:

```julia
h .+= model.pe[:, 1:T]    # model.jl:127 — broadcast PE over batch
```

---

## 3. The attention mechanism

Attention is the central idea of the transformer.  If you read one section carefully, make it this one.

*For visual intuition, see [The Illustrated Transformer: Self-Attention](https://jalammar.github.io/illustrated-transformer/#self-attention-at-a-high-level).*

### 3.1 The intuition

For each position in the sequence, attention asks: *"Which other positions should I pay attention to?"*  It computes a weighted average of all positions, where the weights are determined by how "relevant" each position is to the current one.

The mechanism uses three projections of the input:

- **Query (Q):** "What am I looking for?"
- **Key (K):** "What do I contain?"
- **Value (V):** "What information do I provide?"

The attention weight from position `i` to position `j` is proportional to the dot product of query `i` and key `j`.

### 3.2 A worked example: attention on "the cat sat"

Before the math, let's trace through a concrete example with tiny numbers.  Suppose we have 3 tokens and `Dh = 2` (2-dimensional head).

```
Input vectors (after projection):
  Q = [ 1.0  0.5  0.2 ]     K = [ 0.9  0.4  0.1 ]     V = [ 0.3  0.7  0.1 ]
      [ 0.0  1.0  0.8 ]         [ 0.1  0.8  0.9 ]         [ 0.5  0.2  0.9 ]
       "the" "cat" "sat"          "the" "cat" "sat"          "the" "cat" "sat"
```

**Step 1: Compute scores** — `scores[i,j] = (1/√Dh) · qᵢᵀ kⱼ`

```
scale = 1/√2 ≈ 0.707

scores = Q' * K * scale =                    "the"  "cat"  "sat"
                               "the" attends [ 0.64  0.28   0.07 ]
                               "cat" attends [ 0.39  0.76   0.68 ]
                               "sat" attends [ 0.19  0.53   0.52 ]
```

**Step 2: Apply causal mask** — block future positions

```
                        "the"  "cat"  "sat"
         "the" attends [ 0.64  -Inf   -Inf ]   ← can only see itself
         "cat" attends [ 0.39   0.76  -Inf ]   ← can see "the" and "cat"
         "sat" attends [ 0.19   0.53   0.52]   ← can see everything
```

**Step 3: Softmax per row** — normalize into probabilities

```
                        "the"  "cat"  "sat"
         "the" attends [ 1.00   0.00   0.00]   ← forced to attend only to itself
         "cat" attends [ 0.41   0.59   0.00]   ← mostly attends to "cat"
         "sat" attends [ 0.27   0.38   0.35]   ← roughly uniform over all three
```

**Step 4: Weighted sum of values** — `out[:,i] = Σⱼ attn[i,j] · V[:,j]`

```
  out["the"] = 1.00 · V["the"]                         = [0.30, 0.50]
  out["cat"] = 0.41 · V["the"] + 0.59 · V["cat"]       = [0.54, 0.32]
  out["sat"] = 0.27 · V["the"] + 0.38 · V["cat"] + 0.35 · V["sat"] = [0.39, 0.53]
```

Position "the" can only see itself, so its output is just its own value vector.  Position "cat" blends information from "the" and "cat".  Position "sat" mixes all three, learning which context is most useful for predicting the next word.

This is the core of attention: **a learned, content-dependent weighted average**.

### 3.3 Scaled dot-product attention — the code

For a single head, given matrices Q, K, V each of shape `(Dh, T, B)`:

```
scores[i,j] = (1/√Dh) · Σ_d Q[d,i] · K[d,j]     # how relevant is position j to position i?
attn[i,:]   = softmax(scores[i,:] + mask[i,:])     # normalize into a probability distribution
out[:,i]    = Σ_j V[:,j] · attn[i,j]               # weighted sum of values
```

The `1/√Dh` scaling prevents the dot products from growing too large (which would push softmax into saturation).  This is the "scaled" in "scaled dot-product attention."

From [`attention.jl:132`](../attention.jl#L132):

```julia
function single_head_attention_forward(Q, K, V, mask)
    Dh, T, B = size(Q)
    scale = 1.0 / sqrt(Float64(Dh))

    # Compute attention scores: Q^T K / √Dh
    scores = zeros(T, T, B)
    for bb in 1:B
        scores[:, :, bb] .= (Q[:, :, bb]' * K[:, :, bb]) .* scale .+ mask
    end

    # Softmax per row (each position attends to all past positions)
    attn = zeros(T, T, B)
    for bb in 1:B, i in 1:T
        row = scores[i, :, bb]
        s, c = softmax_forward(row)
        attn[i, :, bb] = s
    end

    # Weighted sum of values
    out = zeros(Dh, T, B)
    for bb in 1:B
        out[:, :, bb] .= V[:, :, bb] * attn[:, :, bb]'
    end

    (out, cache)
end
```

### 3.4 The causal mask

GPT is autoregressive: when predicting position `t`, it can only see positions `1, 2, ..., t`.  The causal mask enforces this by setting future attention weights to zero.

From [`attention.jl:101`](../attention.jl#L101):

```julia
function causal_mask(T::Int)
    mask = zeros(T, T)
    for i in 1:T, j in 1:T
        if j > i
            mask[i, j] = -Inf     # after softmax, exp(-Inf) = 0
        end
    end
    mask
end
```

For a sequence of length 4, the mask looks like:

```
     j=1   j=2   j=3   j=4
i=1 [  0   -Inf  -Inf  -Inf ]     position 1 sees only itself
i=2 [  0     0   -Inf  -Inf ]     position 2 sees 1 and 2
i=3 [  0     0     0   -Inf ]     position 3 sees 1, 2, and 3
i=4 [  0     0     0     0  ]     position 4 sees everything
```

Adding `-Inf` before softmax ensures `exp(-Inf) = 0`, so future positions contribute nothing to the weighted sum.

### 3.5 Attention backward pass

The backward pass for attention is the most intricate part of the codebase.  Here is the full derivation, which appears in [`attention.jl:115`](../attention.jl#L115):

**Given** the forward equations:
```
S[i,j,b] = (1/√Dh) Σ_d Q[d,i,b] · K[d,j,b]
A[i,:,b] = softmax(S[i,:,b])
out[d,i,b] = Σ_j V[d,j,b] · A[i,j,b]
```

**The gradients** (derivation by chain rule):
```
∂L/∂V[d,j,b] = Σ_i ∂L/∂out[d,i,b] · A[i,j,b]
∂L/∂A[i,j,b] = Σ_d ∂L/∂out[d,i,b] · V[d,j,b]
∂L/∂S         = softmax_backward(∂L/∂A)          (per row i)
∂L/∂Q[d,i,b] = (1/√Dh) Σ_j ∂L/∂S[i,j,b] · K[d,j,b]
∂L/∂K[d,j,b] = (1/√Dh) Σ_i ∂L/∂S[i,j,b] · Q[d,i,b]
```

In matrix form ([`attention.jl:161`](../attention.jl#L161)):

```julia
function single_head_attention_backward(cache, grad_out)
    # ∂L/∂A = grad_out^T · V    (each row: how much should each attention weight change?)
    grad_attn[:, :, bb] .= grad_out[:, :, bb]' * V[:, :, bb]

    # ∂L/∂V = grad_out · A      (weighted by attention pattern)
    grad_V[:, :, bb] .= grad_out[:, :, bb] * attn[:, :, bb]

    # ∂L/∂S via softmax backward (per row)
    # ∂L/∂Q = K · ∂L/∂S^T / √Dh
    grad_Q[:, :, bb] .= K[:, :, bb] * grad_scores[:, :, bb]' .* scale

    # ∂L/∂K = Q · ∂L/∂S / √Dh
    grad_K[:, :, bb] .= Q[:, :, bb] * grad_scores[:, :, bb] .* scale
end
```

---

## 4. Multi-head attention

*See [The Illustrated Transformer: Multi-Head Attention](https://jalammar.github.io/illustrated-transformer/#multi-headed-attention).*

A single attention head has one notion of "relevance."  Multiple heads let the model attend to different things simultaneously — one head might track syntactic structure, another might track semantic similarity.

The idea, as a diagram:

```
     x ∈ R^(D×T×B)
     │
     ├──→ Wq ──→ Q_full (D,T,B) ──┬── rows 1:Dh   → Q₁ ──→ Attention₁ ──→ out₁
     │                              ├── rows Dh+1:2Dh→ Q₂ ──→ Attention₂ ──→ out₂
     │                              ├── ...
     │                              └── rows (H-1)Dh+1:D → Q_H → Attention_H → out_H
     ├──→ Wk ──→ K_full (D,T,B) ──┬── split same way into K₁..K_H ──────────────┘
     │                              │                                    (each head
     └──→ Wv ──→ V_full (D,T,B) ──┴── split same way into V₁..V_H      gets its
                                                                          own Q,K,V)
                                         │
                                         ▼
                                   concat(out₁, out₂, ..., out_H)   (D, T, B)
                                         │
                                         ▼
                                      Wo * concat + bo               (D, T, B)
                                         │
                                         ▼
                                      output
```

With `D = 128` and `H = 4`, each head operates on `Dh = 32` dimensions.  The heads run independently — head 1 might learn to track subject-verb agreement while head 3 tracks adjacency.

From [`attention.jl:209`](../attention.jl#L209):

```julia
function multi_head_attention_forward(params, x, mask)
    D, T, B = size(x)
    H = params.n_heads
    Dh = div(D, H)

    # Project Q, K, V
    Q_full, cache_q = linear_forward(params.Wq, params.bq, x)     # (D, T, B)
    K_full, cache_k = linear_forward(params.Wk, params.bk, x)
    V_full, cache_v = linear_forward(params.Wv, params.bv, x)

    # Split into heads and run attention independently
    for h in 1:H
        rows = (h-1)*Dh+1 : h*Dh
        Qh = Q_full[rows, :, :]                                    # (Dh, T, B)
        Kh = K_full[rows, :, :]
        Vh = V_full[rows, :, :]
        out_h, cache_h = single_head_attention_forward(Qh, Kh, Vh, mask)
    end

    # Concat → output projection
    output, cache_o = linear_forward(params.Wo, params.bo, concat)  # (D, T, B)
    (output, cache)
end
```

**The weight matrices** are defined in [`attention.jl:13`](../attention.jl#L13):

```julia
mutable struct AttentionParams
    Wq::Matrix{Float64}   # (D, D)  — query projection
    bq::Vector{Float64}   # (D,)
    Wk::Matrix{Float64}   # (D, D)  — key projection
    bk::Vector{Float64}
    Wv::Matrix{Float64}   # (D, D)  — value projection
    bv::Vector{Float64}
    Wo::Matrix{Float64}   # (D, D)  — output projection
    bo::Vector{Float64}
    n_heads::Int
end
```

With `D = 128` and `H = 4`, each head operates on `Dh = 32` dimensions.  The total number of attention parameters per block is `4 × (D² + D) = 4 × 16,512 = 66,048`.

### Multi-head backward

The backward pass ([`attention.jl:246`](../attention.jl#L246)) reverses the forward:
1. Backward through `Wo` output projection
2. Split the gradient into H heads
3. Run `single_head_attention_backward` for each head
4. Backward through `Wq`, `Wk`, `Wv` projections
5. Sum the three input gradients (since `x` feeds into all three projections)

```julia
grad_x = grad_x_q .+ grad_x_k .+ grad_x_v     # attention.jl:271
```

This `+` is because of the chain rule fan-out: when `x` is used in three places, its total gradient is the sum.

---

## 5. The transformer block

Each transformer block combines attention with a feedforward network (FFN), connected by residual connections and layer normalization.  PicoGPT uses **pre-norm** residual ordering (GPT-2 style).

Here is the full data flow through one block, with the residual "highway" on the left:

```
 h ∈ R^(D,T,B)
 │
 ├────────────────────────────────────┐
 │                                    │ (skip connection)
 ▼                                    │
 LayerNorm(γ₁, β₁)                   │
 │                                    │
 ▼                                    │
 Multi-Head Attention                 │
 │  (Wq, Wk, Wv → Q,K,V             │
 │   → scaled dot-product             │
 │   → concat → Wo)                   │
 │                                    │
 ▼                                    │
 + ←──────────────────────────────────┘   h₂ = h + MHA(LN(h))
 │
 ├────────────────────────────────────┐
 │                                    │ (skip connection)
 ▼                                    │
 LayerNorm(γ₂, β₂)                   │
 │                                    │
 ▼                                    │
 W₁ · x + b₁        (D → d_ff)      │   expand:  128 → 512
 │                                    │
 ▼                                    │
 ReLU                                 │
 │                                    │
 ▼                                    │
 W₂ · x + b₂        (d_ff → D)      │   contract: 512 → 128
 │                                    │
 ▼                                    │
 + ←──────────────────────────────────┘   h₃ = h₂ + FFN(LN(h₂))
 │
 ▼
 h₃ ∈ R^(D,T,B)    → next block (or final layer norm)
```

In code:

```
h = h + MultiHeadAttention(LayerNorm(h))     # sub-block 1
h = h + FFN(ReLU(LayerNorm(h)))              # sub-block 2
```

### 5.1 Layer normalization

Layer norm normalizes each feature vector to have zero mean and unit variance, then applies a learned affine transform:

```
x̂ = (x − μ) / σ
y = γ ⊙ x̂ + β
```

where `μ` and `σ²` are computed along the feature dimension (dim 1), and `γ`, `β` are learned parameters.

From [`tensor_ops.jl:170`](../tensor_ops.jl#L170):

```julia
function layer_norm_forward(x, gamma, beta; eps=1e-5)
    mu = mean(x, dims=1)                          # (1, T, B)
    diff = x .- mu
    var_ = mean(diff .^ 2, dims=1)                # (1, T, B)
    sigma_inv = 1.0 ./ sqrt.(var_ .+ eps)         # (1, T, B)
    x_hat = diff .* sigma_inv                      # (D, T, B)
    y = gamma .* x_hat .+ beta                     # broadcast (D,) over (D,T,B)
    (y, cache)
end
```

The `eps = 1e-5` prevents division by zero when the variance is tiny.

### 5.2 Layer norm backward — the hardest gradient in the codebase

The layer norm backward pass is the most involved derivation.  Here it is in full (from [`tensor_ops.jl:154`](../tensor_ops.jl#L154)):

```
∂L/∂γ    = Σ_{t,b} ∂L/∂y ⊙ x̂
∂L/∂β    = Σ_{t,b} ∂L/∂y
∂L/∂x̂    = ∂L/∂y ⊙ γ
∂L/∂σ²   = −(1/2) Σ_i ∂L/∂x̂_i · (x_i − μ) · σ^{−3}
∂L/∂μ    = −σ^{−1} Σ_i ∂L/∂x̂_i  +  ∂L/∂σ² · (−2/D) · Σ_i (x_i − μ)
∂L/∂x_i  = ∂L/∂x̂_i · σ^{−1}  +  ∂L/∂σ² · (2/D)(x_i − μ)  +  ∂L/∂μ · (1/D)
```

The key insight: `x_i` affects the output through three paths: (1) directly through `x̂`, (2) through the variance `σ²`, and (3) through the mean `μ`.  The chain rule sums over all three.

Implementation ([`tensor_ops.jl:186`](../tensor_ops.jl#L186)):

```julia
function layer_norm_backward(cache, grad_y)
    x, x_hat, gamma, mu, sigma_inv, D = cache

    grad_gamma = vec(sum(grad_y .* x_hat, dims=(2,3)))
    grad_beta  = vec(sum(grad_y, dims=(2,3)))

    dx_hat = grad_y .* gamma
    diff = x .- mu
    dvar = sum(dx_hat .* diff, dims=1) .* (-0.5) .* sigma_inv .^ 3
    dmu  = sum(dx_hat .* (-sigma_inv), dims=1) .+ dvar .* (-2.0 / D) .* sum(diff, dims=1)
    grad_x = dx_hat .* sigma_inv .+ dvar .* (2.0 / D) .* diff .+ dmu ./ D

    (grad_x, grad_gamma, grad_beta)
end
```

### 5.3 The feedforward network

The FFN is a simple two-layer MLP with ReLU activation:

```
FFN(x) = W₂ · ReLU(W₁ · x + b₁) + b₂
```

`W₁` has shape `(d_ff, D)` = `(512, 128)` — it projects up to a larger hidden dimension.  `W₂` projects back down.  This expansion-contraction pattern gives the model more representational capacity inside the block.

### 5.4 Residual connections

The `+` in `h = h + MHA(LN(h))` is the residual connection.  Its gradient is trivially simple: the gradient flows through the `+` unchanged.

```julia
h2 = h .+ attn_out    # model.jl:43 — residual connection
```

In the backward pass:

```julia
grad_h2_res = grad_h3       # model.jl:62 — gradient flows through unchanged
grad_ff2 = grad_h3          # model.jl:63 — same gradient goes both ways
```

This is why deep transformers can train at all.  Without residual connections, the gradient would have to propagate through dozens of matrix multiplications and nonlinearities, shrinking to near-zero (the vanishing gradient problem).  With residuals, the gradient always has a direct path from output to input.

### 5.5 Putting it together

From [`model.jl:38`](../model.jl#L38):

```julia
function transformer_block_forward(block, h, mask)
    # Sub-block 1: h = h + MHA(LN(h))
    ln1_out, cache_ln1 = layer_norm_forward(h, block.ln1_gamma, block.ln1_beta)
    attn_out, cache_attn = multi_head_attention_forward(block.attn, ln1_out, mask)
    h2 = h .+ attn_out

    # Sub-block 2: h = h + FFN(ReLU(LN(h)))
    ln2_out, cache_ln2 = layer_norm_forward(h2, block.ln2_gamma, block.ln2_beta)
    ff1, cache_ff1 = linear_forward(block.W1, block.b1, ln2_out)
    ff1_relu, cache_relu = relu_forward(ff1)
    ff2, cache_ff2 = linear_forward(block.W2, block.b2, ff1_relu)
    h3 = h2 .+ ff2

    (h3, cache)
end
```

---

## 6. The full GPT model

The GPT model stacks it all together ([`model.jl:96`](../model.jl#L96)):

```julia
mutable struct GPT
    cfg::GPTConfig
    E_tok::Matrix{Float64}             # (D, V)  token embeddings
    pe::Matrix{Float64}                # (D, T)  sinusoidal PE (fixed)
    blocks::Vector{TransformerBlock}   # n_layers transformer blocks
    ln_f_gamma::Vector{Float64}        # final layer norm
    ln_f_beta::Vector{Float64}
    W_out::Matrix{Float64}             # (V, D)  output projection
    b_out::Vector{Float64}             # (V,)
end
```

The data flows:

```
tokens ∈ Z^(T×B)
  → E_tok lookup → h ∈ R^(D×T×B)
  → h += PE
  → block_1(h) → block_2(h) → ... → block_N(h)
  → LayerNorm(h)
  → W_out · h + b_out → logits ∈ R^(V×T×B)
```

### Parameter count

With the defaults (`D=128`, `H=4`, `n_layers=4`, `d_ff=512`, `V=65`):

| Component | Parameters per block | Total |
|-----------|---------------------|-------|
| Attention (Wq, Wk, Wv, Wo + biases) | 4 × (128² + 128) = 66,048 | 264,192 |
| FFN (W1, W2 + biases) | 128×512 + 512 + 512×128 + 128 = 131,712 | 526,848 |
| Layer norms (γ, β × 2) | 2 × 2 × 128 = 512 | 2,048 |
| Token embedding | 128 × 65 = 8,320 | 8,320 |
| Output projection | 65 × 128 + 65 = 8,385 | 8,385 |
| Final layer norm | 2 × 128 = 256 | 256 |
| **Total** | | **~810K** |

### Weight initialization

Two schemes, chosen by position ([`tensor_ops.jl:28`](../tensor_ops.jl#L28)):

**Xavier** for most weights: `W ~ N(0, √(2/(fan_in + fan_out)))`

The derivation: if `y = Wx` and we want `Var[y_j] ≈ 1`, then `Var[y_j] = fan_in · Var[w] · Var[x]`.  Balancing forward and backward: `Var[w] = 2 / (fan_in + fan_out)`.

**GPT-2 init** for residual projections (Wo, W2): `W ~ N(0, 0.02 / √(2 · n_layers))`

The `1/√(2·n_layers)` scaling ensures that after summing residuals from all layers, the variance stays bounded.

---

## 7. The backward pass: manual backpropagation

This is the heart of PicoGPT.  Every module has a handwritten backward pass.  Here is the complete inventory:

### 7.1 Primitives

| Operation | Forward | Backward | Where |
|-----------|---------|----------|-------|
| Softmax | `s_i = exp(x_i) / Σ exp(x_j)` | `∂L/∂x_k = s_k(∂L/∂s_k − Σ_j s_j ∂L/∂s_j)` | [`tensor_ops.jl:133–148`](../tensor_ops.jl#L133) |
| Layer norm | `y = γ(x−μ)/σ + β` | 3-path chain rule through x̂, σ², μ | [`tensor_ops.jl:170–205`](../tensor_ops.jl#L170) |
| ReLU | `y = max(0, x)` | `∂L/∂x = ∂L/∂y · 𝟙[x > 0]` | [`tensor_ops.jl:217–228`](../tensor_ops.jl#L217) |
| Cross-entropy | `L = −(1/N) Σ log p(target)` | `∂L/∂logits = (1/N)(softmax − one_hot)` | [`tensor_ops.jl:249–277`](../tensor_ops.jl#L249) |
| Linear | `y = Wx + b` | `∂L/∂W = ∂L/∂y · xᵀ`, `∂L/∂x = Wᵀ · ∂L/∂y` | [`attention.jl:63–90`](../attention.jl#L63) |
| Embedding | table lookup | scatter-add | [`tensor_ops.jl:58–82`](../tensor_ops.jl#L58) |
| Attention | Q,K,V → scores → softmax → weighted sum | see Section 3.4 | [`attention.jl:132–193`](../attention.jl#L132) |
| Residual | `y = x + f(x)` | `∂L/∂x = ∂L/∂y` (pass-through) | [`model.jl:62`](../model.jl#L62) |

### 7.2 Full model backward

The backward pass reverses the forward pass.  Gradients flow from loss back to the first embedding:

```
                                              FORWARD (→)
loss ← cross_entropy ← logits ← W_out ← LN_f ← Block₄ ← Block₃ ← Block₂ ← Block₁ ← E_tok
  │                                                                                       ▲
  │                                   BACKWARD (←)                                        │
  └── ∂L/∂logits → ∂L/∂h → ∂L/∂h → ∂L/∂h → ∂L/∂h → ∂L/∂h → ∂L/∂h → ∂L/∂h → ∂L/∂E ──┘
                     │        │        │        │        │        │
                     ▼        ▼        ▼        ▼        ▼        ▼
                  ∂L/∂W_out ∂L/∂γ_f  (all block parameter gradients collected)  ∂L/∂E_tok

  Each "→" is a forward function call.     Each "←" is a backward function call.
  Forward stores a cache.                  Backward consumes that cache.
```

The implementation ([`model.jl:147`](../model.jl#L147)):

```julia
function gpt_backward(model, cache, grad_logits)
    # 1. Backward through output projection
    grad_h, grad_W_out, grad_b_out = linear_backward(cache_out, grad_logits)

    # 2. Backward through final layer norm
    grad_h, grad_ln_f_gamma, grad_ln_f_beta = layer_norm_backward(cache_ln_f, grad_h)

    # 3. Backward through transformer blocks (REVERSE order)
    for i in length(model.blocks):-1:1
        grad_h, blk_grads = transformer_block_backward(block_caches[i], grad_h)
    end

    # 4. Backward through embedding
    grad_E_tok = embed_backward(cache_emb, grad_h)

    grads    # Dict mapping parameter names → gradient arrays
end
```

Note the **reverse order**: the last block to execute in the forward pass is the first to execute in the backward pass.

### 7.3 Gradient checking — the physicist's unit check

How do we know the gradients are correct?  Finite-difference verification ([`gradcheck.jl:14`](../gradcheck.jl#L14)):

```julia
function numerical_gradient(f, x; eps=1e-5)
    grad = similar(x)
    for i in eachindex(x)
        old = x[i]
        x[i] = old + eps;  fp = f(x)
        x[i] = old - eps;  fm = f(x)
        x[i] = old
        grad[i] = (fp - fm) / (2 * eps)     # central difference
    end
    grad
end
```

For every backward function, we compare the analytical gradient against this numerical estimate.  If the relative error exceeds `1e-5`, something is wrong.  This is used throughout the test suite:

```
PASS  softmax backward: gradcheck (err=3.42e-10)
PASS  layer_norm backward: gradcheck x (err=9.65e-10)
PASS  cross_entropy backward: gradcheck (err=1.10e-09)
PASS  multi_head backward: gradcheck Wq (err=8.82e-09)
```

All errors are well below `1e-5`.

---

## 8. Loss, optimizer, and training

### 8.1 Cross-entropy loss

The loss function measures how well the model predicts the next token.  For logits at position `(t, b)`, we apply softmax to get a probability distribution, then take the negative log probability of the correct token:

```
L = −(1/(T·B)) Σ_{t,b} log softmax(logits)_{target[t,b]}
```

The fused softmax + cross-entropy backward has a beautifully simple result ([`tensor_ops.jl:269`](../tensor_ops.jl#L269)):

```julia
function cross_entropy_backward(cache)
    probs, targets, T, B = cache
    grad = copy(probs)                          # start with softmax output
    for b in 1:B, t in 1:T
        grad[targets[t, b], t, b] -= 1.0       # subtract 1 at correct class
    end
    grad ./= (T * B)                            # average over all positions
end
```

**The gradient of cross-entropy w.r.t. logits is simply `(softmax − one_hot) / N`.** This is one of the most important identities in deep learning.

### 8.2 SGD optimizer

Plain stochastic gradient descent — the simplest possible optimizer:

```
θ ← θ − η · ∂L/∂θ
```

Implemented in [`model.jl:181`](../model.jl#L181) as in-place updates:

```julia
function sgd_update!(model, grads, lr)
    model.E_tok .-= lr .* grads[:E_tok]
    model.W_out .-= lr .* grads[:W_out]
    # ... same for all parameters
end
```

### 8.3 Learning rate schedule

Transformers are sensitive to learning rate.  We use linear warmup followed by cosine decay ([`train.jl:13`](../train.jl#L13)):

```julia
function lr_schedule(step, cfg, total_steps)
    if step < cfg.warmup_steps
        return cfg.learning_rate * (step / cfg.warmup_steps)     # linear warmup
    end
    progress = (step - cfg.warmup_steps) / (total_steps - cfg.warmup_steps)
    cfg.lr_min + 0.5 * (cfg.learning_rate - cfg.lr_min) * (1.0 + cos(π * progress))
end
```

Warmup prevents early instability (large gradients on random weights).  Cosine decay smoothly reduces the learning rate, allowing the model to settle into a minimum.

### 8.4 Gradient clipping

Transformers can produce large gradient spikes.  Global norm clipping prevents this ([`train.jl:56`](../train.jl#L56)):

```
g ← g · min(1, c / ‖g‖₂)
```

If the global gradient norm `‖g‖₂` exceeds `c` (default 1.0), all gradients are uniformly scaled down.  This preserves the direction but limits the step size.

### 8.5 The training loop

The outer loop ([`train.jl:120`](../train.jl#L120)):

```julia
function train!(model, train_batches, val_batches, cfg)
    for epoch in 1:cfg.n_epochs
        train_loss, step = train_epoch!(model, train_batches, cfg, step, total_steps)
        val_loss = evaluate(model, val_batches)
        # print progress
    end
end
```

Each epoch iterates over all batches.  Each batch:
1. Forward pass → logits
2. Loss computation
3. Backward pass → gradients for all ~810K parameters
4. Gradient clipping
5. SGD update

### 8.6 What a training run looks like

When you run `julia run.jl`, you'll see output like this:

```
============================================================
PicoGPT.jl — From-Scratch GPT in Pure Julia
============================================================

[1/4] Loading data...
  Vocab size: 65
  Text length: 1115394 chars
  Train batches: 152
  Val batches: 16

[2/4] Initializing model...
  Parameters: 810497 (~0.81M)

[3/4] Training...
Epoch   1 | train loss: 2.8934 | val loss: 2.5821 | lr: 1.00e-03 | best val: 2.5821
Epoch   2 | train loss: 2.3647 | val loss: 2.2903 | lr: 9.97e-04 | best val: 2.2903
Epoch   3 | train loss: 2.1542 | val loss: 2.1103 | lr: 9.89e-04 | best val: 2.1103
...
Epoch  18 | train loss: 1.4231 | val loss: 1.5982 | lr: 2.14e-04 | best val: 1.4876
Epoch  19 | train loss: 1.4098 | val loss: 1.5841 | lr: 8.15e-05 | best val: 1.4876
Epoch  20 | train loss: 1.4012 | val loss: 1.5790 | lr: 1.00e-05 | best val: 1.4876

[4/4] Generating sample text...

FIRST CITIZEN:
The king is not the brother of the world,
And the state of the strong, and the death
Of the people of the state of the world...
```

**What to watch for:**
- **Loss starts at ~2.9** — close to `log(65) ≈ 4.17` (random guessing)
- **Drops rapidly** in the first 5 epochs as the model learns character frequencies
- **Val loss plateaus** around 1.5 — the model has learned basic English spelling and Shakespeare-like patterns but can't memorize the training set
- **Train < Val** — the gap indicates the model is slightly overfitting, which is expected for a small model

---

## 9. Text generation

Generation is where the model shows what it has learned.  The process is autoregressive — we feed a prompt, predict one token, append it, and repeat:

```
 Step 1:  "T" "h" "e" " "           → model → P(next) → sample "k"
 Step 2:  "T" "h" "e" " " "k"       → model → P(next) → sample "i"
 Step 3:  "T" "h" "e" " " "k" "i"   → model → P(next) → sample "n"
 Step 4:  "T" "h" "e" " " "k" "i" "n" → model → P(next) → sample "g"
          └─────── context ──────────┘           └── one new token each step
```

Each step runs a full forward pass (no backward — we're not training).  The logits at the *last* position give us the distribution over the next token.

From [`generate.jl:11`](../generate.jl#L11):

```julia
function generate(model, context, max_new_tokens; temperature=1.0, top_k=0)
    seq = copy(context)
    for _ in 1:max_new_tokens
        ctx = seq[max(1, end - T_max + 1):end]     # truncate to context window
        tokens = reshape(ctx, T_ctx, 1)              # batch size 1
        logits, _ = gpt_forward(model, tokens)       # forward only
        logits_last = logits[:, end, 1]              # logits at last position
        next_token = sample_token(logits_last; temperature, top_k)
        push!(seq, next_token)
    end
    seq
end
```

Three sampling modes ([`generate.jl:42`](../generate.jl#L42)):

**Greedy** (`temperature=0`): Always pick the highest-probability token.  Deterministic but repetitive.

**Temperature sampling**: Scale logits by `1/T` before softmax.  `T < 1` sharpens the distribution, `T > 1` flattens it.

**Top-k**: Zero out all but the top `k` logits before softmax.  Prevents sampling from the long tail of low-probability tokens.

Here's how they compare on the same logits:

```
Raw logits for next char after "The ki":   n:3.2  s:1.8  t:1.5  a:0.3  ...

Greedy (T=0):        n=100%  s=0%   t=0%   a=0%      → always "n"
Temperature T=0.5:   n=89%   s=7%   t=3%   a=0.3%    → almost always "n"
Temperature T=1.0:   n=51%   s=23%  t=18%  a=5%      → usually "n", sometimes "s"
Temperature T=2.0:   n=30%   s=22%  t=20%  a=12%     → quite random
Top-k (k=3, T=1.0):  n=55%   s=25%  t=20%  a=0%      → only top 3 candidates
```

Higher temperature = more creative but less coherent.  `temperature=0.8` is a good balance for Shakespeare.

---

## 10. Testing everything

PicoGPT follows strict TDD: every function is tested before it's used.  The 42 tests are organized in 6 tiers matching the build order.

The most important test is the **overfit-one-batch** test at Tier 3:

```julia
# A 2-layer model with D=32 must drive loss below 0.1 on 4 sequences in 200 steps
for step in 1:200
    logits, cache = gpt_forward(model, tokens)
    loss, cache_ce = cross_entropy_forward(logits, targets)
    grad_logits = cross_entropy_backward(cache_ce)
    grads = gpt_backward(model, cache, grad_logits)
    sgd_update!(model, grads, lr)
end
# loss must be < 0.1
```

If a small model can't memorize 4 sequences, you have a backward pass bug.  This test caught the `grad_attn` transpose bug during development (see [`attention.jl:171`](../attention.jl#L171) — the fix was removing an erroneous transpose on the attention gradient).

---

## Where to go from here

1. **Read the source files** in dependency order: `config.jl` → `tensor_ops.jl` → `attention.jl` → `model.jl` → `train.jl` → `generate.jl`

2. **Run `julia tests.jl`** and verify all 42 tests pass

3. **Run `julia run.jl`** and watch the loss drop over 20 epochs

4. **Modify something** — change `n_heads`, `d_model`, or `n_layers` in `config.jl` and see what happens

5. **Derive a gradient yourself** — pick any backward function, work through the chain rule on paper, and check that you get the same formula

## References

- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017)
- Radford et al., [*Language Models are Unsupervised Multitask Learners*](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (2019) — GPT-2
- Jay Alammar, [*The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/) — the visual companion to this tutorial
- Andrej Karpathy, [*nanoGPT*](https://github.com/karpathy/nanoGPT) — the inspiration for minimal GPT implementations
- Glorot & Bengio, [*Understanding the difficulty of training deep feedforward neural networks*](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) (2010) — Xavier initialization

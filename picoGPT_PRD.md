# picoGPT.jl -- Product Requirements Document


## 1. Purpose

A from-scratch GPT implementation in pure Julia for training and inference on a
single laptop CPU. Every operation -- forward pass, backward pass, weight update
-- is written by hand with no automatic differentiation. The target reader is a
theoretical physicist fluent in linear algebra and index notation but who has
never built a neural network.

After reading this codebase, you should be able to derive every gradient on
paper and explain what every line does. There is no magic.


## 2. Hard Constraints

| Constraint               | Decision                                              |
|--------------------------|-------------------------------------------------------|
| Language                 | Julia (>= 1.10)                                      |
| Dependencies             | LinearAlgebra (stdlib) only. Nothing else.            |
| Automatic differentiation| None. All gradients are manual.                       |
| Hardware target          | Single laptop CPU, no GPU, no CUDA                    |
| Parameter budget         | ~1M parameters (trainable in minutes on CPU)          |
| Dataset                  | Shakespeare (~1.1M chars, downloaded at runtime)      |
| Tokenization             | Character-level (vocab size ~65)                      |
| Deliverable format       | .jl source files, runnable via julia run.jl           |


## 3. Architecture Specification

GPT-2 architecture throughout.

### 3.1 Model Hyperparameters (defaults)

All fields of a GPTConfig struct:

| Parameter   | Value | Notes                              |
|-------------|-------|------------------------------------|
| vocab_size  | 65    | Data-determined, character-level   |
| d_model     | 128   | Embedding dimension                |
| n_heads     | 4     | Configurable                       |
| n_layers    | 4     | Transformer blocks                 |
| seq_len     | 256   | Context window                     |
| d_ff        | 512   | = 4 * d_model, feedforward hidden  |

Yields roughly 0.8--1.2M parameters.

### 3.2 Component Decisions

| Component            | Choice                          | Rationale                                                    |
|----------------------|---------------------------------|--------------------------------------------------------------|
| Positional encoding  | Sinusoidal (Vaswani)            | No learnable params; closed-form Fourier basis               |
| Attention            | Multi-head, causal mask         | Honest to GPT; n_heads adjustable via config                 |
| Normalization        | Layer norm (pre-norm)           | Full mean + variance; instructive backward pass              |
| Activation           | ReLU                            | Trivial forward/backward; keeps focus on attention           |
| Weight init          | Both Xavier and GPT-2 style     | Side-by-side with Var[y] = n Var[w] Var[x] derivation       |
| Softmax              | Numerically stable (sub max)    | From the start; no detour                                    |
| Optimizer            | Plain SGD with LR warmup        | Transformers are the target, not optimizer engineering        |
| Gradient clipping    | Global norm clip                | Prevents NaN; clean operation                                |

### 3.3 Architecture Flow

    x in Z^(B x T)  [token indices]
      -> Token embedding E_tok[x] in R^(B x T x d)
      -> Add sinusoidal PE(0:T-1) in R^(T x d)
      -> h_0

      x n_layers:
        h = h + MultiHeadCausalAttention(LayerNorm(h))    [pre-norm residual]
        h = h + FFN_ReLU(LayerNorm(h))

      -> LayerNorm -> Linear(d -> V) -> logits in R^(B x T x V)
      -> softmax -> cross-entropy loss

### 3.4 Residual Convention

Pre-norm: normalize before attention/FFN, then add residual. Easier to train
at small scale, cleaner backward pass.


## 4. Manual Backpropagation Specification

Every module f exposes:

- forward(f, x) -> (y, cache) where cache stores intermediates for backward.
- backward(f, cache, grad_y) -> (grad_x, grad_params...) computing dL/dx and dL/dtheta.

### 4.1 Primitives Requiring Handwritten Gradients

Each gets a forward, backward, and inline derivation in comments:

1. **Matmul**: dL/dA = (dL/dC) B^T,  dL/dB = A^T (dL/dC)
2. **Softmax**: dL/dx_i = s_i (dL/ds_i - sum_j s_j dL/ds_j)
3. **Layer norm**: Full derivation with d/dmu and d/dsigma^2 terms
4. **ReLU**: dL/dx = dL/dy .* 1[x > 0]
5. **Cross-entropy**: dL/dlogits = softmax(logits) - one_hot(target)
6. **Embedding lookup**: Scatter-add into embedding rows
7. **Residual**: Gradient flows through unchanged
8. **Causal mask**: Same upper-triangle zero mask in backward

### 4.2 Gradient Checking

Finite-difference checker:

    (f(x + eps * e_i) - f(x - eps * e_i)) / (2 * eps)  ~=  df/dx_i

Standalone utilities in gradcheck.jl. Run on small random inputs (D=4, T=3,
B=2). Relative error threshold: 1e-5. The physicist s unit check.

### 4.3 Testing Strategy: TDD Is the Law

Every function is tested before it is used. The cycle is: write test, watch it
fail, implement, watch it pass, move on. No exceptions.

Testing infrastructure: a minimal harness in tests.jl using only @assert and
helpers. No Test stdlib. A check(name, condition) function that prints PASS/FAIL
and accumulates results.

What gets tested, in order:

**Tier 1 -- Tensor ops (test before building attention)**

- Softmax forward: output sums to 1, all entries positive, invariant under
  constant shift.
- Softmax backward: agrees with finite-difference gradcheck.
- Layer norm forward: output has mean approx 0, variance approx 1 along feature dim.
- Layer norm backward: gradcheck.
- ReLU forward: matches max(0, x) on random inputs including negatives.
- ReLU backward: gradcheck.
- Cross-entropy forward: recovers -log(1/V) for uniform logits, 0 for perfect
  prediction.
- Cross-entropy backward: gradcheck. Verify fused softmax-CE gradient matches
  the two-step version.

**Tier 2 -- Attention (test before building model)**

- Single-head attention on identity-like inputs: Q=K should produce
  uniform-ish attention weights.
- Causal mask: attention weights are zero above diagonal.
- Multi-head: output shape is (D, T, B) regardless of n_heads; n_heads=1
  matches single-head implementation.
- Forward-backward round trip: gradcheck on Q, K, V projection weights with
  small random inputs (D=8, T=4, H=2, B=2).

**Tier 3 -- Model (test before training)**

- Single transformer block: forward produces correct output shape, backward
  produces gradients for all parameters.
- Full model forward: logits shape is (V, T, B).
- Full model backward: every parameter tensor has a corresponding gradient
  tensor of the same shape.
- Overfit test: a 2-layer model with D=32 on a single batch of 4 sequences
  should drive loss below 0.1 within 200 SGD steps. If it cannot memorise a
  tiny batch, something is broken.

**Tier 4 -- Data pipeline (test before training)**

- Tokenization round-trip: encode then decode recovers original string.
- Batch shapes: input is (T, B) integers in [1, V], target is same shape.
- No off-by-one: target[t] == input[t+1] for sequential text.
- Train/val split: val data starts exactly where train data ends, no overlap.

**Tier 5 -- Training loop (test before full run)**

- Single SGD step decreases loss on a fixed batch. If it does not, LR is wrong
  or gradients are wrong.
- LR schedule: warmup is linear, value at t=0 is 0 (or eta_min), value at
  t=warmup_steps is eta_max.
- Gradient clipping: after clipping, ||g||_2 <= c + epsilon.

**Tier 6 -- Generation**

- Greedy on a trained model is deterministic (same seed, same output).
- Temperature=0 matches greedy.
- Top-k with k=1 matches greedy.
- Output length matches requested max_new_tokens.

The overfit-one-batch test at Tier 3 is the most important single test. If a
small model cannot memorise 4 sequences, you have a bug in your backward pass.
Find it before scaling up.


## 5. Data Pipeline

- **Dataset**: Shakespeare input.txt, downloaded from Karpathy s char-rnn repo
  if not present.
- **Tokenization**: Character-level. char_to_idx / idx_to_char from unique
  characters.
- **Split**: Sequential 90/10. No shuffling (avoids context-window leakage).
- **Batching**: Pre-chunk at startup. Non-overlapping windows of seq_len + 1
  (input/target offset by 1), grouped into batches. Deterministic, no random
  sampling.


## 6. Training Specification

### 6.1 Optimizer

Plain SGD: theta <- theta - eta * grad_L(theta)

### 6.2 LR Schedule

Linear warmup over warmup_steps, then cosine decay:

    eta(t) = eta_max * (t / warmup_steps)                              if t < warmup_steps
    eta(t) = eta_min + 0.5*(eta_max - eta_min)*(1 + cos(pi*progress))  otherwise

where progress = (t - warmup_steps) / (total_steps - warmup_steps).

### 6.3 Gradient Clipping

Global norm: g <- g * min(1, c / ||g||_2). Default c = 1.0.

### 6.4 Defaults

| Parameter      | Value       |
|----------------|-------------|
| batch_size     | 32          |
| learning_rate  | 1e-3        |
| warmup_steps   | 100         |
| n_epochs       | 20          |
| max_grad_norm  | 1.0         |
| eval_interval  | every epoch |


## 7. Inference / Generation

Autoregressive: feed context, take logits at last position, sample, append,
repeat.

Three modes:

- Greedy (argmax)
- Temperature sampling (scale logits by 1/T before softmax)
- Top-k (zero out all but top k logits before softmax)

Forward pass only -- no caches, no backward. Separate lean code path.


## 8. File Structure

    picoGPT.jl/
      README.md         -- overview, usage, architecture
      config.jl         -- GPTConfig struct, defaults
      tensor_ops.jl     -- softmax, layer_norm, relu, cross_entropy (fwd + bwd + derivations)
      attention.jl      -- multi-head causal self-attention (fwd + bwd)
      model.jl          -- GPT struct, full forward/backward wiring
      data.jl           -- download Shakespeare, tokenize, batch
      train.jl          -- training loop, SGD, LR schedule, gradient clipping
      generate.jl       -- autoregressive sampling
      gradcheck.jl      -- finite-difference gradient verification utilities
      tests.jl          -- full TDD test suite (Tiers 1--6)
      run.jl            -- entry point: data -> model -> train -> generate

### 8.1 Dependency Graph

    run.jl -> config.jl, data.jl, model.jl, train.jl, generate.jl
    model.jl -> attention.jl, tensor_ops.jl
    train.jl -> tensor_ops.jl
    generate.jl -> tensor_ops.jl
    tests.jl -> gradcheck.jl, config.jl, tensor_ops.jl, attention.jl, model.jl, data.jl, train.jl, generate.jl
    gradcheck.jl -> (standalone utilities, no dependencies)

All via include(). No Pkg, no module declarations.


## 9. Code Conventions

### 9.1 Commenting Philosophy

Every backward function gets a block comment deriving the gradient in index
notation or matrix calculus. Example for layer norm:

    # BACKWARD: Layer Norm
    # Forward: y = gamma .* x_hat + beta,  x_hat = (x - mu) / sigma
    # dL/dgamma = sum_batch (dL/dy .* x_hat)
    # dL/dbeta  = sum_batch dL/dy
    # dL/dx_hat = dL/dy .* gamma
    # dL/dsigma^2 = -(1/2) sum_i dL/dx_hat_i (x_i - mu) sigma^{-3}
    # dL/dmu  = -sigma^{-1} sum_i dL/dx_hat_i + dL/dsigma^2 (-2/d) sum_i (x_i - mu)
    # dL/dx_i = dL/dx_hat_i sigma^{-1} + dL/dsigma^2 (2/d)(x_i - mu) + dL/dmu (1/d)

### 9.2 Naming

- Dimensions: B (batch), T (seq len), D (d_model), H (n_heads), Dh (D/H),
  V (vocab_size)
- Forward returns (result, cache)
- Backward takes (cache, grad_output), returns (grad_input, grad_params...)
- No single-letter names except i, j, k and dimension letters above

### 9.3 Array Layout

Julia is column-major. Tensors stored as:

- Embeddings: (D, V) -- each column is an embedding vector
- Hidden states: (D, T, B) -- contiguous along embedding dimension
- Attention weights: (T, T, H, B)

Layout choices commented where they affect performance.


## 10. Success Criteria

| Criterion          | Target                                                        |
|--------------------|---------------------------------------------------------------|
| Loss               | Val loss below 1.5 (char-level CE on Shakespeare)             |
| Generation         | Recognizable pseudo-Shakespearean output                      |
| Gradient check     | All manual gradients match finite diff, rel error < 1e-5      |
| Performance        | Full training < 30 min on modern laptop CPU                   |
| Readability        | A physicist can read attention.jl and re-derive every gradient |
| Tests              | All Tiers 1--6 pass via julia tests.jl                        |


## 11. Out of Scope

- KV caching for fast inference
- BPE or any subword tokenization
- Dropout
- Mixed precision
- GPU support
- Checkpointing
- Any Julia package beyond LinearAlgebra


## 12. Implementation Order

TDD cycle: write tests first (they fail), then implement until they pass.

 1. config.jl -- structs
 2. gradcheck.jl -- finite-difference utilities (needed by all tests below)
 3. Write Tier 1 tests in tests.jl (they fail)
 4. tensor_ops.jl -- implement until Tier 1 passes
 5. Write Tier 2 tests (they fail)
 6. attention.jl -- implement until Tier 2 passes
 7. Write Tier 3 tests (they fail)
 8. model.jl -- implement until Tier 3 passes, including overfit-one-batch
 9. Write Tier 4 tests (they fail)
10. data.jl -- implement until Tier 4 passes
11. Write Tier 5 tests (they fail)
12. train.jl -- implement until Tier 5 passes
13. Write Tier 6 tests (they fail)
14. generate.jl -- implement until Tier 6 passes
15. run.jl -- glue, full training run
16. README.md

The overfit-one-batch test at step 8 is the single most important test. If a
small model cannot memorise 4 sequences, you have a backward pass bug. Find it
before scaling up.

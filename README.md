# PicoGPT.jl

**A from-scratch GPT implementation in pure Julia.  Every gradient by hand.  No automatic differentiation, no GPU, no magic.**

PicoGPT.jl builds a complete GPT-2-style language model — forward pass, backward pass, weight update — using nothing but `LinearAlgebra`.  It trains on Shakespeare and generates pseudo-Shakespearean text, all on a single laptop CPU in under 30 minutes.

The target reader is a **theoretical physicist** (or anyone comfortable with linear algebra and index notation) who has never built a neural network.  After reading this codebase, you should be able to derive every gradient on paper and explain what every line does.

```
julia run.jl
```

```
FIRST CITIZEN:
The king is not the brother of the world,
And the state of the strong, and the death
Of the people of the state of the world...
```

## Why this exists

Most transformer tutorials either (a) hide the gradients behind autodiff or (b) show forward-pass pseudocode without a working backward pass.  PicoGPT does neither.  Every `backward` function has an inline derivation in index notation, every gradient is verified against finite differences, and the whole thing trains end-to-end.

If you want to understand *Attention Is All You Need* at the matrix-calculus level, this is the codebase to read.


## Inspirations and references

This project stands on the shoulders of:

- **[Andrej Karpathy's minGPT / nanoGPT](https://github.com/karpathy/nanoGPT)** — the gold standard for minimal GPT implementations.  PicoGPT follows Karpathy's philosophy of radical simplicity but goes further by removing autodiff entirely.  We also use his [char-rnn Shakespeare dataset](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare).

- **[Jay Alammar's "The Illustrated Transformer"](https://jalammar.github.io/illustrated-transformer/)** — the single best visual introduction to the transformer architecture.  If you're reading PicoGPT's code and want a picture of what's happening, Alammar's post is your companion.  Our tutorial in `docs/` follows his progressive-disclosure style: high level first, then zoom into each component, then the math.

- **[Vaswani et al., "Attention Is All You Need" (2017)](https://arxiv.org/abs/1706.03762)** — the original transformer paper.  Our sinusoidal positional encoding, scaled dot-product attention, and multi-head architecture come directly from this paper.

- **[Radford et al., "Language Models are Unsupervised Multitask Learners" (2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)** — the GPT-2 paper.  Our pre-norm residual ordering, weight initialization scheme, and overall architecture follow GPT-2.


## Architecture at a glance

```
tokens ∈ Z^(B×T)
  │
  ▼
Token Embedding  E_tok[x]           (D, V) lookup table
  + Sinusoidal PE                   sin/cos Fourier basis, no learnable params
  │
  ▼
  h₀  ∈ R^(D×T×B)
  │
  │  ×n_layers:
  │  ┌─────────────────────────────────────────────┐
  │  │  h = h + MultiHeadAttention(LayerNorm(h))   │  pre-norm residual
  │  │  h = h + FFN(ReLU(LayerNorm(h)))            │
  │  └─────────────────────────────────────────────┘
  │
  ▼
LayerNorm → Linear(D → V) → logits ∈ R^(V×T×B)
  │
  ▼
softmax → cross-entropy loss
```

### Default hyperparameters (~1M parameters)

| Parameter     | Value | Notes                          |
|---------------|-------|--------------------------------|
| `vocab_size`  | 65    | Character-level (data-driven)  |
| `d_model`     | 128   | Embedding dimension            |
| `n_heads`     | 4     | Attention heads                |
| `n_layers`    | 4     | Transformer blocks             |
| `seq_len`     | 256   | Context window                 |
| `d_ff`        | 512   | FFN hidden dim (4 × d_model)   |


## Quick start

**Requirements:** Julia >= 1.10.  No packages to install.

```bash
# Clone
git clone https://github.com/tobiasosborne/PicoGPT.jl.git
cd PicoGPT.jl

# Run tests (42 tests across 6 tiers)
julia tests.jl

# Train on Shakespeare and generate text
julia run.jl
```

That's it.  No `Pkg.add`, no `conda`, no `pip`.


## File structure

```
PicoGPT.jl/
  config.jl         GPTConfig + TrainConfig structs
  tensor_ops.jl     softmax, layer_norm, relu, cross_entropy, embedding, PE, weight init
  attention.jl      multi-head causal self-attention (fwd + bwd)
  model.jl          GPT struct, full forward/backward, SGD update
  data.jl           download Shakespeare, tokenize, batch
  train.jl          training loop, LR schedule, gradient clipping
  generate.jl       autoregressive sampling (greedy / temperature / top-k)
  gradcheck.jl      finite-difference gradient verification
  tests.jl          TDD test suite (Tiers 1–6, 42 tests)
  run.jl            entry point
  docs/             tutorial and detailed walkthrough
```

### Dependency graph

```
run.jl ──→ config.jl, data.jl, model.jl, train.jl, generate.jl
model.jl ──→ attention.jl, tensor_ops.jl
train.jl ──→ tensor_ops.jl
generate.jl ──→ tensor_ops.jl
tests.jl ──→ everything + gradcheck.jl
```


## The forward/backward contract

Every module follows the same pattern:

```julia
forward(args...)  → (result, cache)      # cache stores intermediates
backward(cache, grad_output) → (grad_input, grad_params...)
```

The `cache` is the key idea: the forward pass saves everything the backward pass needs (inputs, masks, intermediate values).  No recomputation, no global state.


## Manual gradients — a taste

Here's the softmax backward pass from [`tensor_ops.jl:143`](tensor_ops.jl#L143):

```julia
function softmax_backward(cache, grad_s)
    s, = cache
    dot = sum(s .* grad_s, dims=1)     # Σ_j s_j · ∂L/∂s_j
    s .* (grad_s .- dot)               # s_k (∂L/∂s_k − Σ_j s_j ∂L/∂s_j)
end
```

The derivation (in the source file above the function):

```
∂s_i/∂x_k = s_i (δ_ik − s_k)
∂L/∂x_k   = Σ_i ∂L/∂s_i · s_i (δ_ik − s_k)
           = s_k (∂L/∂s_k − Σ_i s_i ∂L/∂s_i)
```

Every backward function in the codebase has a derivation like this.  See the [full tutorial](docs/tutorial.md) for all eight.


## Testing philosophy

**TDD is the law.**  Every function is tested before it's used.  The test suite has 6 tiers that mirror the build order:

| Tier | What | Key tests |
|------|------|-----------|
| 1 | Tensor ops | Softmax sums to 1, LayerNorm normalizes, all gradchecks pass |
| 2 | Attention | Causal mask blocks future, multi-head shapes, gradcheck on Wq/Wv |
| 3 | Model | Block/model shapes, **overfit-one-batch** (loss < 0.1 in 200 steps) |
| 4 | Data | Tokenization round-trip, batch shapes, no off-by-one |
| 5 | Training | SGD decreases loss, LR warmup/decay schedule, gradient clipping |
| 6 | Generation | Greedy is deterministic, top-k=1 matches greedy, correct length |

The **overfit-one-batch** test at Tier 3 is the most important.  If a small model can't memorize 4 sequences, you have a backward pass bug.

```bash
julia tests.jl
# Results: 42/42 passed
```


## Array layout

Julia is column-major.  All tensors are laid out for contiguous access along the embedding dimension:

| Tensor | Shape | Convention |
|--------|-------|------------|
| Hidden states | `(D, T, B)` | Embedding × Time × Batch |
| Embeddings | `(D, V)` | Each column = one embedding vector |
| Attention weights | `(T, T, B)` | per head; `attn[i,j]` = position i attends to j |
| Logits | `(V, T, B)` | Vocab × Time × Batch |


## Notation guide

Used consistently across the codebase:

| Symbol | Meaning |
|--------|---------|
| `B` | Batch size |
| `T` | Sequence length |
| `D` | `d_model`, embedding dimension |
| `H` | Number of attention heads |
| `Dh` | `D / H`, per-head dimension |
| `V` | Vocabulary size |
| `d_ff` | FFN hidden dimension |


## Documentation

- **[Tutorial: Building a Transformer from Scratch](docs/tutorial.md)** — a progressive walkthrough of the entire architecture, from token embedding to text generation, with every gradient derived.  Modeled after [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/).

- **[Detailed Code Reference](docs/reference.md)** — function-by-function documentation with Julia code excerpts, shapes, and gradient derivations for every module.


## Hard constraints

| Constraint | Decision |
|------------|----------|
| Language | Julia >= 1.10 |
| Dependencies | `LinearAlgebra` (stdlib) only |
| Autodiff | None — all gradients are manual |
| Hardware | Single laptop CPU |
| Parameters | ~1M |
| Dataset | Shakespeare (~1.1M chars) |
| Tokenization | Character-level |


## License

Apache 2.0.  See [LICENSE](LICENSE).


## Acknowledgments

Built with care by a physicist who wanted to understand transformers the hard way.  Thanks to Karpathy for showing that GPT can fit in one file, and to Alammar for showing that attention can fit in one picture.

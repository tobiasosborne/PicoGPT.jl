# Figures

The catalogue of every figure in the book: what it answers, how it encodes, how the reader
handles it, where its numbers come from, and how it prints. Figure builders implement against
this list and the glyphs of `DESIGN.md` §5. Rules in brief (DESIGN.md §10):

- File `web/src/figures/NN-slug/fig-<id>.svelte`; placed in prose by `::: {.figure #fig-<id> attrs}`.
  The div's content is the caption.
- A pure function of (specimen, attrs, state, width). Two modes: `interactive`, and `print`
  (static, complete; animations become small multiples of their phases).
- Numbers only from the specimen (`getContext('specimen')()`, adapter
  `web/src/lib/model/specimen.js`: `windowTrace(start)`, `params(θ)`, `theta`, `theta0`, `loss`,
  `provenance`, `layout`, `config`) or from `artifacts/` through a server load.
  **No invented numbers**, and no hand-typed numbers in the prose.
- Value ink for forward quantities, covector ink (bars, hatching) for adjoints; sign by fill or
  direction; forward runs down, backward up; scales shared across small multiples.
- The first figure that uses a glyph exports its "how to read" `Key`.
- Check each figure at `/figures/<id>/`, with `?print`, and with `?film&t=…` if it animates.

Status: **built** (in the repository), **planned** (specified here, not yet built).
"Primitives" names components in `web/src/figures/primitives/`; *spec* means a glyph is
specified in DESIGN.md §5 but has no component yet. Its first builder writes the primitive.

---

## 0 · Prologue (`00-prologue`)

| id | the one question | encoding | interaction | data | print | status |
|---|---|---|---|---|---|---|
| `fig-specimen-anatomy` | What is the specimen, all of it? | every parameter block of θ as a `MatrixGlyph` (vectors as one-row matrices), stacked in forward order (down: E, block, final); shapes and names as direct labels; key: matrix glyph | point at a block: name, shape, role, parameter count; point at a square: its entry | `params(theta)` (trained θ from the artifact) | static, trained θ | **built** |
| `fig-two-inks` | What do the two inks mean? | one affine map y = W x in value ink (spokes of x, matrix W, spokes of y), with its pullback x̄ = Wᵀȳ in covector ink beside it, the arrows of computation pointing down and up; key: value vs covector marks (dot vs bar, solid vs hatched) | scrub forward → backward (one clock) | one real column of `ln1` and of the pullback of the specimen's loss (engine `backward`) | the two states side by side | **built** |
| `fig-three-names` | Why three names? | none (the `::: names` table already does this) | — | — | — | not needed |

## 1 · The task (`01-task`)

| id | question | encoding | interaction | data | print | status |
|---|---|---|---|---|---|---|
| `fig-chain-rule` | How does the probability of a string factor into next-character predictions? | the corpus line in Literata; under each character a bar of p(xₜ \| x₍<t₎), and a running sum of −log p as a staircase (value ink); the product as a single number at the end | drag the window; click a character to open its full distribution (V bars) | `windowTrace(s).probs` at the targets | one window, all bars | **built** |
| `fig-contexts-to-distributions` | A language model is a map from contexts to distributions: what does that map look like? | small multiples, one column per prefix length t = 1…T: the V probabilities as bars (rows = characters, aligned across columns) | drag the window | `windowTrace(s).probs` | one window | **built** |
| `fig-training-windows` | What exactly is the training set? | the 18-character corpus, and below it the 10 windows as a staircase of rows; inputs in ink, the targets (shifted by one) in value ink; N = T·B columns counted | point at a column to see (window, position) | `corpus`, artifact `batch` | static | **built** |
| `fig-entropy-floor` | What can no model learn? | the 80 targets in the layout of `fig-training-windows`; under each, a dot of area = the least surprise any model can have on it (−log n_{c,y}/n_c); the trained model's surprise as a ring; the mean, the floor, computed and checked against `lab.floor` | point at a prediction: its context and followers | `lab.train`, corpus, `predictions(θ)` | static | **built** |

## 2 · Symbols become vectors (`02-vectors`)

| id | question | encoding | interaction | data | print | status |
|---|---|---|---|---|---|---|
| `fig-one-hot` | What is a character to the model? | the vocabulary as 7 slots; one-hot e_a as one filled dot; E e_a as the selected column of E, drawn as a `SpokeGlyph` (D = 8 spokes). **First spoke glyph of the book: its Key is here** (`fig-attention-pipeline` keeps a reminder) | click a character | `params(theta)` E | all 7 characters as small multiples | **built** |
| `fig-embedding-matrix` | What does E look like, and how did it get there? | E as `MatrixGlyph` 8×7 (columns labelled by characters), and the 7 columns as spoke glyphs on a shared scale | scrub training steps 0…200 (θ(t) recomputed by the engine from θ₀ with the artifact's hyperparameters) | `theta0`, artifact `train.hyper`, engine `trainStep` | steps 0 and 200 side by side | **built** |
| `fig-positions` | How is position written down? | P (8×8) as four sinusoid pairs: each row a hairline curve sampled at t = 1…8 (dots), frequencies ω_i labelled; and P as a `MatrixGlyph` | point at a position column | artifact `P` (or engine `positionalEncoding`) | static | **built** |
| `fig-rotation` | What does a shift do to the positional code? | each (sin, cos) pair of P as a clock: the positions as dots at angle ω_r(t−1), hands P_t (ink) and P_{t+k} (value), the turn ω_r k as an arc; the Gram matrix PᵀP (Toeplitz) with the diagonal of offset k outlined | drag a hand round its clock (t), drag the shift scale (k); arrow keys | adapter `P` | t, k from attrs | **built** |
| `fig-x0` | What enters the first block? | for each position of a window: spoke of E[:, xₜ] + spoke of P[:, t] = spoke of X₀[:, t]; three rows of small multiples, shared scale | drag the window | `windowTrace(s).X0`, θ, P | one window | **built** |

## 3 · Energies and probabilities (`03-energies`)

| id | question | encoding | interaction | data | print | status |
|---|---|---|---|---|---|---|
| `fig-energy-ladder` | What is softmax? | **energy ladder** (*spec*): rungs at E_a = −z_a labelled with the character, occupation bars ∝ p_a; key for the ladder | drag a rung (a logit); slide the whole ladder (nothing changes: shift invariance); drag τ | `windowTrace(s).Z` at one position | τ = 1 | planned |
| `fig-temperature` | What does temperature do to a distribution? | the same ladder as small multiples at τ ∈ {¼, ½, 1, 2, 4}; entropy under each | none (the interactive version is the ladder) | Z at one position | static | planned |
| `fig-logsumexp` | Why subtract the maximum? | the logits ×k on a number line; exp(z) overflowing (bars running off the page, marked ∞) vs exp(z − max) staying in [0, 1] | drag k | Z | k = 1 and k = 400 | planned |

## 4 · Measuring surprise (`04-surprise`)

| id | question | encoding | interaction | data | print | status |
|---|---|---|---|---|---|---|
| `fig-surprise` | What does the loss charge for each prediction? | per position: a bar of −log p(yₜ) in nats under the character; a hairline at ln V labelled *uniform guessing*; the mean marked as the loss | drag the window; scrub training step (θ₀ … θ_final) | engine forward at θ(t) | trained and untrained, stacked | planned |
| `fig-cross-entropy-kl` | Cross-entropy = entropy + KL: how big is each part? | three aligned bars per context (H of the data's next-character distribution, KL, CE); the floor marked | click a context | empirical next-character distributions from the corpus; model probs; artifact `train.entropy_floor` | static | planned |
| `fig-gradient-p-minus-e` | What is the gradient of the loss with respect to the logits? | p as value bars; e_y as an open outline bar; p − e_y as **covector** bars (hatched; negative at y). **First covector glyph: export the covector Key here** | click a position | probs, targets | one position | planned |

## 5 · Attention (`05-attention`)

| id | question | encoding | interaction | data | print | status |
|---|---|---|---|---|---|---|
| `fig-attention-reads` | What does each character read? | window as large Literata characters with position indices; `Arcs` from the reader to each source j ≤ i, width ∝ A_ij; head 1 above, head 2 below; labels ≥ 0.1; corpus strip with the window bracket; **Key: arcs** | click/←→ a reader; hover previews; drag the window along the corpus | `windowTrace(s).heads[h].A` | reader and window from attrs | **built** |
| `fig-attention-pipeline` | How do query and keys become weights, and weights an output? | top→bottom: keys k_j (`SpokeGlyph`), query q_i (reader's column), scores (lollipops about 0), weights (bars, Σ = 1), values v_j muted with A_ij v_j solid, output o_i; temperature axis (log) with √d marked; reminder of the spoke key (the key is in `fig-one-hot`) | click a reader or a head's name; drag τ (double-click: back to √d); ↑↓ on the τ slider | `heads[h].q/k/v/S` (weights recomputed from q·k/τ; equal to the model's A at τ = √d) | τ = √d, reader/head from attrs | **built** |
| `fig-attention-triangle` | What do the attention matrices of both heads look like, and how are they made? | `TriangleDots` small multiples (head 1, head 2); clock t: scores S for every pair (signed: filled/open) → mask removes the upper triangle → softmax per row → A; **Key: lower-triangle dots** | scrub/play t; point at a row (both heads); click to pin | `heads[h].S` (before the mask) and `.A` | 3 phases × 2 heads, small multiples | **built** |
| `fig-heads-direct-sum` | How do two heads share the space? | W_q, W_k, W_v as `MatrixGlyph` 8×8 with rows 1–4 / 5–8 bracketed as head 1 / head 2; Q for a window as 8 rows of spokes split in two | point at a head's block | θ, `windowTrace(s).heads` | static | planned |
| `fig-sqrt-d` | Why divide by √d? | histograms (dot strips) of q·k for random q, k drawn from the initialisation distribution at d = 1, 4, 16, 64, unscaled vs scaled; the softmax entropy under each | none | seeded draws from the init distribution (state the seed in the caption) | static | planned |

## 6 · The block (`06-block`)

| id | question | encoding | interaction | data | print | status |
|---|---|---|---|---|---|---|
| `fig-residual-stem` | What is the shape of the whole model? | **residual stem** (*spec*): per token column a vertical stem; attention and MLP branch off and merge at + nodes; branch width ∝ ‖update‖; key for the stem | click a column; scrub the forward pass (value ink flows down) | `windowTrace(s)`: X0, Xmid, Xout, O, Y | stem for one column + small multiples of ‖update‖ for all | planned |
| `fig-layernorm-sphere` | What does LayerNorm do, geometrically? | spoke glyphs before / after centring / after scaling, with the RMS circle √D marked; a 2-D sketch of projection onto 1⊥ and the sphere | click a column | `Xmid`, `ln2`, γ₂, β₂ | one column, three stages | planned |
| `fig-mlp` | What does the MLP compute? | U = W₁F̂ + b₁ as 32 bars, the ReLU cut drawn as the zero line eating the negative bars, R, then Y = W₂R + b₂ as a spoke glyph; W₁, W₂ as `MatrixGlyph` | click a column | `windowTrace(s)` U, R, Y; θ | one column | planned |
| `fig-composition` | The model as one composition of maps | the named spaces and maps in a column (ℝ^V → ℝ^{D×T} → … → Δ^V), shapes in JuliaMono | none | `config` | static | planned |

## 7 · Pulling back (`07-pulling-back`)

| id | question | encoding | interaction | data | print | status |
|---|---|---|---|---|---|---|
| `fig-forward-backward` | How does the loss travel back through the model? | the stem of `fig-residual-stem` with the forward pass (value, down) and the backward pass (covector, up, hatched); at each node ‖x̄‖ | scrub one clock: forward then backward (`Scrubber ink="covector"` for the second half) | engine `forward` + `backward` (the `bars` trace) | forward and backward side by side | planned |
| `fig-affine-pullback` | What is the pullback of an affine map? | W (value matrix), ȳ (covector spokes), x̄ = Wᵀȳ (covector spokes), W̄ = ȳxᵀ (covector `MatrixGlyph`, hatched); **Key: covector matrix** | click a column | engine backward: `bars.blocks[0].Q`, `gradParams.Wq` | one column | planned |
| `fig-softmax-pullback` | Why does no gradient come from the future? | S̄ as a covector `TriangleDots` (hatched dots; nothing above the diagonal), with A beside it | point at a row | `bars.blocks[0].S` | static | planned |
| `fig-gradcheck` | Is the derivation right? | for every parameter block, \|analytic − finite difference\| on a log axis as a dot strip; a hairline at 1e−8 | recompute (seeded) | artifact `grad` vs engine finite differences | static | planned |

## 8 · Learning (`08-learning`)

| id | question | encoding | interaction | data | print | status |
|---|---|---|---|---|---|---|
| `fig-train-live` | What does learning look like, step by step? | the θ field of the cover (type weight ∝ \|θᵢ\|), the loss curve growing, the attention arcs of one reader updating | **train** / pause / reset; drag the learning rate | engine `trainStep` from `theta0` with the artifact's hyperparameters | steps 0, 10, 50, 200 as small multiples | planned |
| `fig-loss-curve` | How fast does it learn, and how far can it go? | **loss curve** (*spec*): log step vs nats, ln V line labelled *uniform guessing*, the entropy floor, greedy samples in the margin at their steps | point at a step | artifact `train.loss`, `train.entropy_floor`, samples via engine `generate` | static | planned |
| `fig-adam` | What does Adam change? | per-parameter step sizes \|Δθᵢ\| for GD and AdamW as two dot strips, grouped by block (log scale) | none | engine `step` on the specimen, both optimisers, from the same θ | static | planned |
| `fig-schedule` | What is the learning rate doing? | warmup + cosine curve with the actual η, η_min and warmup marked | none | artifact `train.hyper` → engine `learningRate` | static | planned |
| `fig-init-variance` | Why these initial variances? | activation RMS through the forward pass at init, for three init scales, as small multiples | none | engine `init(config, seed, {matrixStd})` + `forward` | static | planned |

## 9 · Speaking (`09-speaking`)

| id | question | encoding | interaction | data | print | status |
|---|---|---|---|---|---|---|
| `fig-sampling-tree` | What happens when the model writes? | a tree: at each step the energy ladder of the next character, the drawn branch in ink, alternatives faint with their probabilities | step / play (scrubbable); slide τ; reseed | engine `generate` with `seededRandom(seed)` | three steps as small multiples | planned |
| `fig-shakespeare-temperature` | What does the real model say at different temperatures? | the same prompt continued at τ ∈ {0.5, 0.8, 1.0, 1.2}, set as text columns | none | `artifacts/shakespeare/*.json` samples | static | planned |
| `fig-shakespeare-checkpoints` | What did the real model learn, and when? | its loss curve with samples at checkpoints in the margin | point at a checkpoint | `artifacts/shakespeare/*.json` | static | planned |
| `fig-shakespeare-attention` | Do the real model's heads read anything recognisable? | `TriangleDots` small multiples for every head of every layer on one line of text | point at a head | exported attention patterns | static | planned |

## Epilogue (`10-epilogue`)

| id | question | encoding | interaction | data | print | status |
|---|---|---|---|---|---|---|
| `fig-scale` | How big is big? | parameter counts as area-encoded squares: the specimen, the Shakespeare model, and published models | none | specimen and Shakespeare counts computed; published counts **cited in the caption** (the only external numbers in the book) | static | **built** |

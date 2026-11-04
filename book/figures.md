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
| `fig-energy-ladder` | What is softmax? | **energy ladder** (`EnergyLadder`): rungs at their true energies E_a = −z_a on the rail, labels and occupation bars on shelves dodged apart (leaders) where levels (nearly) coincide; τ-ruler standing on the lowest rung (faint 2τ…4τ); free energy F marked on the axis; live 𝒵, F, H and p(a) = exp(z/τ)/𝒵; corpus strip picks the context; **Key: energy ladder** | drag a rung (snaps to exact ties); drag the axis (shift: 𝒵, F change, p does not); drag the ruler's end (τ; snaps to 1); click/←→ the corpus; reset | `windowTrace(s).Z` at one position | τ = 1, model's logits | **built** |
| `fig-temperature` | What does temperature do to a distribution? | two contexts (attrs `contexts="w:p,…"`) × ladders at τ → 0, ¼, ½, 1, 2, 4, τ → ∞ (exact limits), one energy scale, entropy under each; beside each row the path of softmax(z/τ) through the simplex, projected as the frontispiece projects it, the seven τ marked | point at a ladder (its point on the path) | Z at two positions | static | **built** |
| `fig-logsumexp` | Why subtract the maximum? | the seven e^{βz_a} on a log axis spanning all of Float64 (limits from `Number.MAX_VALUE/MIN_VALUE`), as written vs max subtracted, ∞ and 0 bins hatched; the sum Σ marked; the probabilities each recipe returns (NaN shown) | drag β (log track 1…1000; the first-overflow β marked) | Z at one position | β = 1 and β = 200 | **built** |

## 4 · Measuring surprise (`04-surprise`)

| id | question | encoding | interaction | data | print | status |
|---|---|---|---|---|---|---|
| `fig-surprise` | What does the loss charge for each prediction? | all N = 80 predictions: row b = window b set under the corpus, each bar −ln q(y) under the character it predicts; outline = untrained, solid = at the step; ln V hairline; the mean = the loss; the floor; **Key: surprise bars** | scrub training (log step; the run replayed by the adapter and verified against the artifact's loss); point at a bar | `predictions(θ)`, `trajectory()` | outline untrained, solid trained | **built** |
| `fig-cross-entropy-kl` | How much of the loss can learning remove? | predictions grouped by context (window prefix): the open contexts in full (their windows; p̂ outlined vs q solid over the vocabulary; H + KL split bar), every decided context together, and the loss = floor + divergence with the floor formula (n ln m terms) built from the data | scrub training step | `predictions(θ)`, `trajectory()`, corpus | trained | **built** |
| `fig-gradient-p-minus-e` | What is the gradient of the loss with respect to the logits? | forward down in value ink: z (lollipops) → p (bars; e_y outlined) → ℓ; backward up in covector ink: N z̄ = p − e_y hatched about zero, above z; if the context is shared, the sum n(p − p̂) of its windows; **Key: covector (first in the book)** | choose any of the 80 predictions on the corpus strip; untrained / trained | `predictions(θ)` | window 3, position 1, trained | **built** |

## 5 · Attention (`05-attention`)

| id | question | encoding | interaction | data | print | status |
|---|---|---|---|---|---|---|
| `fig-attention-reads` | What does each character read? | window as large Literata characters with position indices; `Arcs` from the reader to each source j ≤ i, width ∝ A_ij; head 1 above, head 2 below; labels ≥ 0.1; corpus strip with the window bracket; **Key: arcs** | click/←→ a reader; hover previews; drag the window along the corpus | `windowTrace(s).heads[h].A` | reader and window from attrs | **built** |
| `fig-attention-pipeline` | How do query and keys become weights, and weights an output? | top→bottom: keys k_j (`SpokeGlyph`), query q_i (reader's column), scores (lollipops about 0), weights (bars, Σ = 1), values v_j muted with A_ij v_j solid, output o_i; temperature axis (log) with √d marked; reminder of the spoke key (the key is in `fig-one-hot`) | click a reader or a head's name; drag τ (double-click: back to √d); ↑↓ on the τ slider | `heads[h].q/k/v/S` (weights recomputed from q·k/τ; equal to the model's A at τ = √d) | τ = √d, reader/head from attrs | **built** |
| `fig-attention-triangle` | What do the attention matrices of both heads look like, and how are they made? | `TriangleDots` small multiples (head 1, head 2); clock t: scores S for every pair (signed: filled/open) → mask removes the upper triangle → softmax per row → A; **Key: lower-triangle dots** | scrub/play t; point at a row (both heads); click to pin | `heads[h].S` (before the mask) and `.A` | 3 phases × 2 heads, small multiples | **built** |
| `fig-heads-direct-sum` | How do two heads share the space? | W_q, W_k, W_v as `MatrixGlyph` 8×8 with rows 1–4 / 5–8 bracketed as head 1 / head 2; Q for a window as 8 rows of spokes split in two | point at a head's block | θ, `windowTrace(s).heads` | static | planned |
| `fig-sqrt-d` | Why divide by √d? | histograms (dot strips) of q·k for random q, k drawn from the initialisation distribution at d = 1, 4, 16, 64, unscaled vs scaled; the softmax entropy under each | none | seeded draws from the init distribution (state the seed in the caption) | static | planned |

## 6 · The block (`06-block`)

| id | the one question | encoding | interaction | data | print | status |
|---|---|---|---|---|---|---|
| `fig-residual-stem` | What is the shape of the whole model? | **residual stem** (`primitives/ResidualStem`, `stem.js`): per token column a vertical stem, width ∝ ‖x‖/√D; attention and MLP branch off and merge at ⊕, width ∝ ‖update‖/√D; arcs = the exact share of a later token's attention update handed over by an earlier one (the only marks between stems); the selected stem large with spoke glyphs at stations; **Key: stem** (`StemKey`) | click/←→ a stem; drag the corpus strip (windows 1–10); scrub the forward pass (x ← x + s·f(x) drawn literally) | `lab.gradTrace(theta)` at the trained θ | complete t = 1 state | **built** |
| `fig-layernorm-sphere` | What does LayerNorm do, geometrically? | exact meridian plane (1ᵀx/√D, ‖Πx‖) with the circle ‖x‖ = √D: centre → scale (to just inside; inset magnifies √D − ‖x̂‖) → affine; one column's spoke glyphs at four stages with the RMS = 1 ring; **Key: meridian** | scrub stages; click a point/character, ←→ | LN₂ on `Xmid`, γ₂, β₂, trained θ | stages as the static state | **built** |
| `fig-mlp` | What does the MLP compute? | f̂ spoke; u_k as 32 lollipops (unfired open); r_k·W₂[:,k] spoke glyphs under firing units; Σ → y; **Key: switched sources** | click a column, ←→; point at a unit, ↑↓ | `lab.gradTrace(theta)` U, R, ln2; θ | one column | **built** |
| `fig-composition` | The model as one composition of maps | spine of spaces with shapes; maps with formulas; parameter arrays as squares of area ∝ count (P open: fixed); counts from the layout, self-checked against paramCount | none | `layout`, `config` | static | **built** |

## 7 · Pulling back (`07-pulling-back`)

| id | question | encoding | interaction | data | print | status |
|---|---|---|---|---|---|---|
| `fig-forward-backward` | How does the loss travel back through the model? | the stems of `fig-residual-stem`: forward (value, down) then backward (covector, up, hatched bands, bar tips; `ResidualStem ink="covector"`); ‖x̄‖ at each station; reversed attention arcs (exact per-source shares of the adjoint); at ⊕ the adjoint is copied, at forks adjoints add; **Key: pullback** (`PullbackKey`) | one clock (forward, then backward with a covector scrubber); click a stem; drag the window; θ₀ / after training (only ambiguous predictions still send covectors) | `lab.gradTrace(θ)` (every intermediate and its adjoint) | forward and backward side by side | **built** |
| `fig-affine-pullback` | What is the pullback of an affine map? | readout at θ₀: W_out (value Hinton), z̄ = (p − e_y)/N (hatched bars), x̄ = W_outᵀz̄, this column's rank-one share z̄xᵀ and the full W̄_out (hatched Hinton); **Key: covector matrix** | click a column; move the window; point at a square | `lab.gradTrace(theta0)` | one column | **built** |
| `fig-softmax-pullback` | Why does no gradient come from the future? | A (value), Ā over the whole square (covector, signed), then Ā → minus its A-mean → ×A = S̄ (lower triangle, hatched/open dots); row sums of S̄ read out | point/click a row; head names; covector scrubber through the pullback | `lab.gradTrace(theta0)` heads | four stages | **built** |
| `fig-gradcheck` | Is the derivation right? | left: \|FD − analytic\| vs h (10⁻¹…10⁻¹³) for one entry of E, W_q, W₁, W_out with h² and uℓ/h guides; right: all 1007 errors at a chosen h grouped by array, each array's largest gradient as a covector bar; live timing of forward vs forward+backward | click one of five checked steps, ←→ | `lab.evaluator()` at θ₀ (≈10⁴ forward passes, via `computed()`) | h = 10⁻⁶ | **built** |

## 8 · Learning (`08-learning`)

| id | question | encoding | interaction | data | print | status |
|---|---|---|---|---|---|---|
| `fig-train-live` | What does learning look like, step by step? (also the book's loss curve: there is no separate `fig-loss-curve`) | **loss curve** (`primitives/LossChart`; **Key: loss curve**, `LossChartKey`): log step vs nats, ln V *uniform guessing*, the entropy floor (computed from the corpus windows), greedy samples of "to be or" at checkpoints in a staircase above; η per step below; every parameter as a Hinton square (one vmax); what one reader reads (`Arcs`) | train/pause, one step, reset; drag η on a log axis; click AdamW / gradient descent, warmup + cosine / constant η; seed ‹ › (0 = the book's θ₀) | `lab.run()` (the artifact's recipe; reproduces `train.loss` to ~1e-13), `lab.evaluator().look`, `lab.greedy` | steps 0, 10, 30, 200 + full curve | **built** |
| `fig-step-size` | How large a step can gradient descent take? | GD at η = ¼, ½, 1: loss curves and the sharpness Λ(θ_t) (power iteration on finite-difference Hessian–vector products) against 2/η: the edge of stability | point along a curve | `lab.run({optimizer:'gd'})`, `lab.evaluator().sharpness` (via `computed()`) | static | **built** |
| `fig-minibatch` | Is a minibatch gradient an unbiased estimate, and how noisy? | every minibatch gradient for b = 1, 2, 5, 10, projected onto a fixed plane (covector bars), the full gradient as a cross at the exact mean, rms circles vs the finite-population formula; before / after training (at the floor the ambiguous windows still pull) | click before/after; point at a mark | `lab.evaluator().grad(θ, windows(idx))` (via `computed()`) | both states | **built** |
| `fig-adam` | What does Adam change? | per parameter, grouped by array: \|g_i\| (covector bars) and \|Δθ_i\|/η_t (value dots) on one log axis; step 1 = sign descent | scrub the step through the book's run; point at a row | `lab.run()` | steps 1 and 200 | **built** |
| `fig-schedule` | What are the learning rate and the clip doing? | η_t (warmup + cosine, values labelled) and ‖g_t‖ before clipping against the clip radius (hatched ball exterior) on aligned step axes | point at a step, ←→ | `lab.run()` | static | **built** |
| `fig-slingshot` | What does Adam do at the floor? | plain Adam (λ = 0, no clip), constant η vs warmup + cosine, 4000 steps: η_t, loss − floor (log, min–max envelope per bin, spikes labelled), ‖Δθ‖ | point at a spike; focus a run, ←→ | `lab.run(...)` (via `computed()`) | static | **built** |
| `fig-init-variance` | Why these initial variances? | RMS of every forward station at init (rows top→bottom = forward), one dot per seed, medians joined; book's variances vs ×9 and ÷9 (all weight matrices but E); attention evenness and initial loss vs ln V below; **Key: profile** | point at a row | `lab.initTheta(seed)` (artifact `init` distributions), `lab.gradTrace` (via `computed()`) | static | **built** |

## 9 · Speaking (`09-speaking`)

| id | question | encoding | interaction | data | print | status |
|---|---|---|---|---|---|---|
| `fig-sampling-tree` | What happens when the model writes? | one column per drawn character: [0, 1] cut into p = softmax(z/τ) in vocabulary order (the CDF the sampler inverts), u as a tick, chosen segment solid, alternatives outlined; a line through the chosen segments; a bracket where contexts leave the training set; **Key: sampling column** | drag τ (far left: τ → 0, greedy); draw again (seed); play/scrub the writing; point at a column | adapter `speak()` (engine `sample` with `seededRandom`, u recorded) | τ → 0, 1, 2 as rows | **built** |
| `fig-shakespeare-temperature` | What does the real model say at different temperatures? | the same prompt at every exported τ (last checkpoint) as text columns (HTML); words absent from the training text dot-underlined; share of real words as a bar per column | none | `artifacts/shakespeare/samples.json`; word counts from `artifacts/data/tinyshakespeare.txt` (server, `lib/server/artifacts.js`) | static | **built** |
| `fig-shakespeare-checkpoints` | What did the real model learn, and when? | loss on training and held-out text (log step, step 0 set apart), ln V *uniform guessing*; the τ = 1 sample at each checkpoint in the margin with leaders and its share of real words | point at a sample (whole text) | `artifacts/shakespeare/{loss,samples}.json` | static | **built** |
| `fig-shakespeare-attention` | Do the real model's heads read anything recognisable? | the held-out passage with each character's surprise beneath it and arcs for the chosen reader/head; all 16 heads as lower-triangle dot matrices; per head self / prev / copy (copy measured over the repeated line, found automatically) | point at a character (row in every head); click a head (arcs) | `artifacts/shakespeare/attention.json` (sparse, w ≥ 0.01) | static | **built** |

## Epilogue (`10-epilogue`)

| id | question | encoding | interaction | data | print | status |
|---|---|---|---|---|---|---|
| `fig-scale` | How big is big? | parameter counts as area-encoded squares: the specimen, the Shakespeare model, and published models | none | specimen and Shakespeare counts computed; published counts **cited in the caption** (the only external numbers in the book) | static | **built** |

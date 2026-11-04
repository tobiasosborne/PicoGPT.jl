# PicoGPT — the brief

This is the single source of truth for the redesign. Every contributor reads it first.
If something here is wrong, flag it and leave the rest alone.

## 0. What we are making

There are three artefacts. Each must be excellent; none may be traded off against another.

1. **The code.** A from-scratch GPT in Julia, written "from the book": every line earns its
   place, the structure mirrors the mathematics, it is fast, and it uses no autodiff.
   Gradients are derived by hand.
2. **The interactive book (web).** A long-form explorable explanation. Prose plus live
   figures driven by a real model running in the browser, with animations and short films.
3. **The print book (PDF).** The same text and the same figures, typeset to a standard
   worthy of a beautifully printed science book. **Built from the same HTML** with a print
   stylesheet (CSS Paged Media via Paged.js or Vivliostyle), rendered to PDF by headless
   Chromium. No Typst or LaTeX anywhere: one renderer for web and print.

**Reader.** A BSc physics or mathematics student who knows linear algebra, basic analysis
and basic probability. The maths in this subject is easy; what makes it hard is jargon and
ML conventions. We remove that difficulty: every ML term is defined in terms the reader
already owns.

**Tonal reference.** Hohm, *Lecture Notes on Statistical Physics and Neural Networks*,
arXiv:2605.06394. Copy these habits:
- Objects are maps between named spaces, e.g. $E\in\mathbb R^{D\times V}$, $P:\mathcal X\to[0,1]$.
- Index form and matrix form appear side by side.
- "Any respectable object has at least three names" (maths / physics / ML).
- Softmax *is* the Gibbs distribution, so sampling temperature is literally temperature.
- Cross-entropy is derived from KL divergence.
- Finite sets come first.
- Dry wit is allowed. Hype is not.

**Anti-goals.**
- Neon-on-black "AI" styling, glowing neural blobs, purple gradients, generic Tailwind
  cards, dashboard UIs with dropdown menus.
- A clone of `tobiasosborne.github.io/mipstar-lambda`, i.e. warm-paper, teal, boxed
  "instrument" panels. That project is a data point, not the target.
- A clone of 3Blue1Brown's look.

## 1. The organising idea

> A transformer is a composition of maps. Running it pushes values forward; training pulls
> one number, the loss, back through the transposed derivatives.

Everything is built on this idea:

- **The code.** Every operation returns its value *and its pullback*.
- **The prose.** The backward pass is the adjoint of the forward pass.
- **The visual system.** Two inks.
  - **Value ink** marks forward quantities: activations and parameters.
  - **Covector ink** marks backward quantities: gradients and adjoints, written with a bar, x̄.

## 2. Notation (binding for code, prose, and figures)

| symbol | meaning |
|---|---|
| $V$ | vocabulary size (number of distinct characters) |
| $D$ | model width ("d_model") |
| $H$, $d=D/H$ | number of heads, head width |
| $T$ | context length; $N$ = number of token columns in a batch ($N = T\cdot B$) |
| $L$ | number of blocks |
| $F$ | MLP hidden width (= $4D$) |
| $X\in\mathbb R^{D\times N}$ | the **residual stream**: one column per token |
| $E\in\mathbb R^{D\times V}$ | embedding; the column $E e_a$ belongs to character $a$ |
| $P\in\mathbb R^{D\times T}$ | positional code (sinusoidal, fixed, not learned) |
| $\theta\in\mathbb R^{p}$ | **all** parameters as one vector |
| $\bar x$ | adjoint / gradient of the loss w.r.t. $x$ |
| $A\in\mathbb R^{T\times T}$ | attention matrix, **rows = queries (readers), columns = keys (sources)**; row-stochastic, lower-triangular |

Arrays are column-major. The column is the token. Index $t$ is position; index $a$ is a vocabulary entry.

## 3. The model (exact semantics; the Julia, JS, and prose all implement this)

```
X  = E[:, x] + P[:, pos]                           # embed
for each block ℓ:
    Â = LN(X; γ1, β1)
    Q = Wq Â + bq ;  K = Wk Â + bk ;  V = Wv Â + bv   # D×N each
    for each head h, each sequence:  (rows of Q,K,V for head h = (h-1)d+1 : hd)
        S = Q_hᵀ K_h / √d                          # T×T, S[i,j] = q_i·k_j/√d
        A = softmax_rows(S + M)                     # M[i,j] = -∞ for j>i, else 0
        O_h = V_h Aᵀ                                # d×T
    X = X + (Wo O + bo)                             # residual add
    F̂ = LN(X; γ2, β2)
    X = X + (W2 relu(W1 F̂ + b1) + b2)               # residual add
Z = Wout LN(X; γf, βf) + bout                       # logits, V×N
loss = mean over columns n of [ logsumexp(Z[:,n]) − Z[y_n, n] ]
```

- LN uses ε = 1e-5. In geometric terms it does three things: centre (project orthogonally to
  **1**), scale onto the sphere of radius √D, then apply the diagonal affine map.
- The loss is written as log-sum-exp. Never write `log(p + 1e-10)`.
  The gradient w.r.t. Z is exactly `(softmax(Z) − onehot(y)) / N`.
- Parameter order in θ is fixed:
  `E`; then per block: `γ1 β1 Wq bq Wk bk Wv bv Wo bo γ2 β2 W1 b1 W2 b2`; then `γf βf Wout bout`.
  - Each matrix is stored column-major as a contiguous slice of θ.
  - Shapes: `Wq,Wk,Wv,Wo: D×D`, `W1: F×D`, `W2: D×F`, `Wout: V×D`.
- Optimiser: AdamW is the default for real training; plain gradient descent is available
  for teaching. Also included:
  - global-norm clipping;
  - a warmup + cosine learning-rate schedule.

## 4. The specimen

This is a model small enough to see every number. It runs live in the browser.

| quantity | value |
|---|---|
| corpus | `to be or not to be` (18 characters) |
| V | 7; vocabulary sorted: `' ' b e n o r t` |
| D, H, d | 8, 2, 4 |
| F | 32 |
| L | 1 |
| T | 8 |
| training set | every length-T window of the corpus, all positions predicted |
| arithmetic | Float64 |

The total parameter count is computed and printed. Do not hard-code it.

**The real model** is character-level Tiny Shakespeare at roughly the old defaults
(D=128, H=4, L=4, T=256) or larger if speed allows. It trains in Julia only. Its results
(loss curves, samples at checkpoints, attention patterns, embeddings) are exported as data
for the book.

## 5. The specimen trace contract (Julia → JS conformance)

Julia writes `artifacts/specimen.json`:

```jsonc
{
  "config": {"V":7,"D":8,"H":2,"F":32,"L":1,"T":8,"eps":1e-5},
  "vocab": [" ","b","e","n","o","r","t"],
  "corpus": "to be or not to be",
  "layout": [{"name":"E","shape":[8,7],"offset":0}, ...],   // θ layout, offsets 0-based
  "P": {"shape":[8,8],"data":[...]},                         // column-major
  "batch": {"x":[[...]...], "y":[[...]...]},                 // B windows, 1-based token ids
  "theta0": [...],                                           // initial θ (flat)
  "forward": { "X0":{shape,data}, "blocks":[{ "ln1","Q","K","V","S","A","O","Xmid","ln2","U","R","Y","Xout" }], "Xf", "Z", "probs", "loss" },
  "grad": [...],                                             // ∇θ at theta0, same layout as θ
  "train": {"optimizer":"adamw", "hyper":{...}, "steps":N, "loss":[...], "theta_final":[...]}
}
```

- `S` is stored **before** masking. `A` has shape `[T,T,H,B]`.
- All other arrays are column-major `{shape, data}`.
- The JS engine must reproduce every array to ≤1e-12 (absolute) and the loss trajectory to
  ≤1e-9.

## 6. Repository layout

```
Project.toml                 Julia package (stdlib only unless there is a strong reason)
src/PicoGPT.jl               module; file order mirrors book order
src/*.jl
test/runtests.jl             Test stdlib; generic gradcheck over every pullback
scripts/                     train_shakespeare.jl, export_specimen.jl, export_book_data.jl
artifacts/                   generated data (specimen.json, shakespeare/*.json)
book/chapters/NN-slug.md     the prose: ONE source for web and print (pandoc markdown)
book/figures.md              catalogue of figures: id, question it answers, data, chapter
web/                         the interactive book (Vite; owner chooses framework)
web/src/specimen/            JS model engine (pure ES module, zero deps)
print/                       print stylesheet + Paged.js/Chromium PDF build
media/                       rendered films (mp4/webm) + posters
DESIGN.md                    the visual system (written by the design lead)
```

The old root-level `.jl` files, `docs/`, `picoGPT_PRD.md`, and `HANDOFF.md` will be removed
once their replacements exist.

## 7. Book source conventions (pandoc markdown, consumed by web AND print)

```
# Chapter title {#ch-attention}

::: claim
One sentence: the chapter's mathematical claim in plain language.
:::

::: names
| maths | physics | ML |
|---|---|---|
| row-stochastic matrix | transition kernel | attention weights |
:::

::: aside
Sidenote / margin note. Use it for asides, history, and "in the code" pointers.
:::

::: {.figure #fig-attention-arcs}
Caption text. The figure id maps to a web component AND a static print rendering.
:::

::: {.code file="src/attention.jl" fn="attention"}
:::      ← excerpt pulled from the real source at build time (never pasted by hand)

::: exercise
For the reader.
:::

Inline math $x$, display math $$...$$ (LaTeX syntax, rendered by KaTeX at build time).
```

## 8. Chapters (draft arc)

0. **Prologue.** What this is. The specimen. How to read it: three names, two inks.
1. **The task.** Text as a finite sequence. The chain rule of probability, which makes the
   model autoregressive. A language model is a map from contexts to distributions.
2. **Symbols become vectors.** One-hot vectors, the embedding matrix, positions as
   Fourier features.
3. **Energies and probabilities.** Logits are negative energies. Softmax is the Gibbs
   distribution. Temperature. Shift invariance and log-sum-exp.
4. **Measuring surprise.** Cross-entropy, KL divergence, maximum likelihood. The baseline
   ln V. The gradient p − e_y.
5. **Attention.** A row-stochastic mixing of columns; query, key, value; the 1/√d scaling;
   the causal mask as lower-triangularity; heads as a direct-sum decomposition.
   *Tokens interact only here.*
6. **The block.** The residual stream as an Euler step x ← x + f(x). LayerNorm as projection
   to a sphere. The MLP. The whole model as one composition.
7. **Pulling back.** The chain rule as a composition of transposes. Each pullback derived.
   The gradient check.
8. **Learning.** Gradient flow and its discretisation; minibatch noise; Adam as diagonal
   preconditioning; schedules; clipping; why the initial variances are what they are.
9. **Speaking.** Sampling, temperature, and what the Shakespeare model learned, shown honestly.

**Epilogue.** What we left out (BPE, scale, GPUs), plus a dictionary of jargon.

## 9. Design direction (the design lead writes DESIGN.md; these are the givens)

- **An illustrated science book that came alive.** Print-first discipline: every figure has
  a beautiful static state that survives black-and-white printing. The PDF is the web book's
  HTML under a print stylesheet (`@page`, running heads, margin notes as print sidenotes,
  figures in their static state), paginated by Paged.js in headless Chromium.
- **Paper, near-black ink, and exactly two spot inks** (value / covector), each with one
  meaning everywhere. Specify colours in OKLCH.
- **Information design informed by Federica Fragapane.**
  - Glyphs where every mark is a real number.
  - A "how to read" key the first time each glyph appears: one enlarged instance with
    hairline leader lines.
  - Small multiples.
  - Direct labelling instead of legend boxes.
  - Generous white space.
- **Candidate glyphs.**
  - A vector as a radial spoke glyph.
  - Attention as arcs between characters, and as a lower triangle of area-encoded dots with
    *nothing* above the diagonal.
  - The residual stream as a stem, with blocks branching off it and merging back.
  - Distributions as bars on an energy ladder.
  - The loss curve with a reference line labelled "uniform guessing" at ln V, and
    samples in the margin.
- **Typography.** A variable editorial serif for prose, KaTeX/Typst maths matched to it,
  a mono for code and shapes, and small caps or a grotesk for labels. Tufte-style sidenotes.
  Fonts must be open-licensed so they can be shipped.
- **Motion only where something happens in time:** forward pass, backward pass, a training
  step, sampling.
  - The reader can scrub it.
  - `prefers-reduced-motion` is respected.
  - The forward pass flows in value ink; the backward pass flows back in covector ink.
- **Interactivity is direct manipulation**, not dropdowns:
  - drag a logit;
  - slide the temperature;
  - click a token;
  - press "train" and watch the specimen learn live.
- **Films** are rendered from the *same* figure code (headless capture), not a second engine.

## 10. Working rules for agents

- Stay inside your assigned paths. Do not commit; the orchestrator commits.
- No placeholder content presented as final. No invented numbers: every number comes
  from the model.
- Report back: what you built, how you verified it, what is unfinished, and any decision
  you made that others must know.

## 11. Frontispieces: traces (owned by the orchestrator)

Each chapter opens with a **trace**: a plotter-style image that records a real computation.
Spec:
- Hairline single-stroke paths in ink, at most one accent ink.
- Density encodes time or multiplicity.
- No decoration; every stroke is a path the model actually took.

Examples:
- Softmax paths across the simplex as temperature varies.
- Embedding trajectories during training.
- Every attention arc drawn over the course of training.
- Every parameter θ_i(t) across a training run (the cover candidate).
- Loss curves from many seeds.
- A sampling tree.

Each trace exports both as a web/print image and as a plotter-ready SVG (stroke-only paths).
Layout must reserve a frontispiece slot at each chapter opening and on the cover.

## 12. Findings the prose must honour (from book/notes/derivations.md and the traces)

- **b_k never learns.** Its gradient is identically zero: q·b_k shifts each softmax row by a
  constant. Keep it in the model on purpose and use it as a reader puzzle ("one group of these
  1007 numbers can never learn").
- **LayerNorm with ε > 0 maps to just inside the sphere of radius √D**, not onto it. Say so
  precisely.
- **The specimen's loss floor is exactly (2 ln 2 + 3 ln 3)/80 ≈ 0.05853 nats, not 0.** Of the 80
  predictions (10 windows × 8 positions), five come at position 1 of a window, where the context is
  one character long:
  - two windows begin with "o", continued by " " or "r" (ln 2 each);
  - three begin with " ", continued by "b", "o" or "n" (ln 3 each).
  Every other prediction is determined by its context. Training reaches about 0.0588. Compute
  these numbers in figures; never hard-code them. The surprise trace
  (web/public/traces/surprise.svg) shows the two plateaux.
- **Plain Adam at the floor periodically "slingshots"** (loss spikes after convergence). This
  is a real exhibit for chapter 8. A cosine schedule suppresses it.
- `book/notes/derivations.md` is the mathematical reference for all chapters; parameter
  counts are 1,007 (specimen) and 810,049 (V=65, D=128, F=512, L=4).

## 13. The canonical recipe (binding for Julia, the JS engine, figures, and traces)

There is ONE initialisation and ONE optimiser. Julia's `src/parameters.jl:initial` and
`artifacts/specimen.json` ("init", "train", "conventions") are the reference.

**Initialisation** (mean, std):
- `E` ~ (0, 1)
- `Wq Wk Wv W1` ~ (0, 1/√D)
- `Wo` ~ (0, 1/√(2L·D))
- `W2` ~ (0, 1/√(2L·F))
- `Wout` ~ (0, 1/D)
- every γ = 1
- every β and bias = 0

**Optimiser.** AdamW:
- β1 = 0.9, β2 = 0.95, ε = 1e-8.
- Weight decay λ applies to matrices only (parameters with 2-D shapes), not to γ, β or biases.
- Global-norm clipping before the moments.

**Schedule.** η_t = η·t/warmup for t ≤ warmup, then cosine from η to η_min.

**Specimen recipe:** η = 0.02, η_min = 0.002, warmup = 20, λ = 0.01, clip = 1, 200 steps,
full batch.

**Trajectory convention:** `loss[k+1]` is the loss after k updates, k = 0…steps.

Every JS consumer (engine, `web/src/lib/model/*`, figures, traces) calls ONE engine
implementation of init and step. There must be no second implementation of training in the
web code.

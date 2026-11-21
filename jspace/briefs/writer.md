# Brief for the authors of *Poised to speak* (prose + figures)

You own a set of chapters of a short book, end to end: the prose in `jspace/book/chapters/NN-slug.md`
and every figure for it in `jspace/figures/NN-slug/`. The book explains the mathematics of
Anthropic's paper *Verbalizable Representations Form a Global Workspace in Language Models*
(Gurnee, Sofroniew, Lindsey et al., Transformer Circuits Thread, July 2026;
https://transformer-circuits.pub/2026/workspace/index.html — read its Introduction, Methods, "The
J-space", "Structure Supports Function" and the appendix "Formalization of the J-space"; the
authors were also given a local plain-text copy, `jspace/briefs/paper.txt`, which is not in the
repository: read the paper at its URL) to the reader of PicoGPT, by computing every object in it on
the two models we own completely. It is a **supplement** to PicoGPT: same reader, same voice,
same design system, same pipeline, its own PDF (`jspace/PoisedToSpeak.pdf`). It is not part of
the main book and must not touch it.

Read, in this order:
1. `BRIEF.md` §1 (organising idea), §2 (notation), §7 (source conventions), §12 (findings the prose
   honours). The supplement inherits all of it.
2. `DESIGN.md`: the visual system and the figure contract (§5–§11). Binding.
3. `jspace/README.md`, `jspace/edition.js` (the outline and the data hooks).
4. `jspace/briefs/analysis-report.md` and `jspace/artifacts/README.md`: what was computed, where
   every number lives, and the decisions of the analyst. `jspace/src/JLens.jl` is the code you
   quote with `::: {.code file="jspace/src/JLens.jl" fn="…"}`.
5. `book/chapters/05-attention.md` + `web/src/figures/05-attention/`: the reference chapter for
   voice, density and figure craft. Also `book/chapters/07-pulling-back.md` (the pullback) and
   `book/chapters/06-block.md` (the residual stream), because this book stands on them.
6. `jspace/book/figures.md`: the figure catalogue for this book. A strong proposal, not a cage.

## What this book is for

The reader has read PicoGPT (or is reading it). They know: the residual stream X ∈ ℝ^{D×N}, one
column per token; the block as two residual updates; the readout Z = Wout LN(X_L) + bout; the
pullback as the transpose of the derivative; the two inks. They have not read any interpretability
paper. The paper's difficulty is, again, vocabulary ("residual stream at layer ℓ", "unembedding",
"lens", "steering", "ablation", "sparse frame", "occupancy", "broadcast", "workspace") plus one
real mathematical idea: **replace the rest of the model by its averaged derivative**. Our job is to
make that idea and everything built on it elementary, exact, and seen whole on the specimen, and
then to test the paper's structural claims honestly on the Shakespeare model.

The title says the thesis: the J-lens finds what the model is *poised to say*.

## The mathematics, chapter by chapter (the arc is in `jspace/edition.js`)

Use the paper's definitions exactly as the analysis implements them (the report says how). Depth
ℓ ∈ {0, ½, 1, …, L}: X_0 after embedding, X_{ℓ−½} after attention of block ℓ, X_ℓ after its MLP.
Specimen: 0, ½, 1. Shakespeare: nine depths. Both models' numbers are in the artifacts.

**0 Prologue.** The paper's claim in plain words, then the plan. Three names from the start:
residual stream / state / "activation"; the readout Wout / the unembedding W_U; an averaged
gradient / a covector / "a lens vector". Say plainly what we can and cannot do: the paper's
models have vocabularies of tokens (V ≫ D, overcomplete lens vectors); ours have characters
(V < D: 7 vectors in ℝ⁸, 65 in ℝ¹²⁸), so some statements change shape and we say so; the paper's
behavioural experiments (introspection, auditing, consciousness talk) are out of scope; we take no
position on the philosophy, as the paper itself does not.

**1 Reading the stream early.** The stream at every depth as a column vector; the readout as a
map; the **logit lens**: apply the readout to an early column (J = I). Show it on the specimen
(every depth, one window), and that it is unreliable early. Why: the stream's coordinates are not
preserved between depths; the only thing that turns depth-ℓ coordinates into final coordinates is
the rest of the model, which is not linear. Shakespeare: the logit lens's top-1 agreement with
the model's own output by depth.

**2 The averaged Jacobian.** The map "the rest of the model" from X_ℓ[:, t] to X_L[:, t′], t′ ≥ t;
its derivative, a D×D block K_{t′t} per pair of positions (in the specimen 36 blocks of 8×8 for
one window; zero for t′ < t by causality: a triangle, like attention). Index and matrix form side
by side. **The average** J_ℓ over t, t′ ≥ t and over texts, and the paper's reason for averaging:
a single context's Jacobian mixes what the model is *disposed* to say with what it is *using* the
concept for right now; the average keeps the disposition. Show one window's blocks beside the
per-window averages and the grand average. How it is computed: by pullbacks — row i of K_{t′t} is
the pullback of e_i (quote the Julia); and because the pullback is linear, one pullback with e_i
in every column gives the sum over t′ ≥ t at once. Also the two limiting variants (self-only
t′ = t and future-only t′ > t) in one sentence each.

**3 The Jacobian lens.** lens_ℓ(x) = softmax(Wout LN(J_ℓ x) + bout). The **J-lens vectors** v_a =
rows of Wout J_ℓ. Insist on the two inks: v_a = J_ℓᵀ Woutᵀ e_a is a **covector**, the averaged
gradient of logit a with respect to the stream at depth ℓ; draw it with the covector glyph (bar
tips, covector ink); the stream column x is a value (value ink); the readout is the pairing
⟨v_a, x⟩. Facts to show, not state: at the last depth J_L = (2/(T+1))·I, so the J-lens is the
model's own output there (LayerNorm removes the scalar); at depth 0 the J-lens and logit lens
differ most. Specimen: the readouts of both lenses at every depth and every position of a window,
beside the model's prediction. Shakespeare: the lens grid for the "GREMIO" prompt (one character
per position per depth, weight of type ∝ probability: a typographic figure), and the per-depth
curves: agreement with the model, accuracy against the actual next character, for both lenses.

**4 The J-space.** V vectors v_a and a sparsity k. Cones: nonnegative combinations of k of them;
the J-space is the union of the C(V, k) cones, not a subspace (a subspace would contain −v_a). The
distance d_k(x) to the nearest cone; the J-component x_J and the remainder. In the specimen the
minimisation is exact: every subset tried, nonnegative least squares on each (quote the Julia).
Show one column's J-coordinates (bars per character) for each k, and the fraction of ‖x‖²
captured as k grows, against a random frame of the same size: the control the paper uses. Then the
honest shape change: with V < D even k = V leaves a remainder. Shakespeare: the greedy pursuit, the
curves per depth, the **occupancy** (where the J frame stops beating random directions), and the
share of variance in the full span of the 65 vectors against a random 65-dimensional subspace.
Define "sparse frame" in one sentence as the paper does and then stop using the word.

**5 Reading and writing.** Reading: a probe is ⟨v_a, x⟩. Writing: steer x ← x + α v_a; ablate
(project out); **swap in lens coordinates**: V = [v_s v_u], c = V⁺x with the pseudoinverse derived
as least squares in two lines, x′ = x + V(σ(c) − c); what stays fixed (the component orthogonal to
the plane of v_s, v_u) and why that matters. Show on the specimen what each does to the output
distribution at that position **and at later positions** (an edit at t is read by later readers
through attention). Then the paper's test of whether the J-space is *privileged*: a concept vector
split into its J-component and remainder, each used to steer at equal size; report what the
specimen shows, whatever it shows. Shakespeare: the one swap example from the report.

**6 Is it a workspace?** The paper's three structural signatures, tested: (i) a band of depths
(CKA between the lens-vector sets of different depths, and the per-depth curves of chapter 3);
(ii) limited capacity (chapter 4's occupancy and variance share); (iii) broadcast: MLP gain of
J-lens directions against random and neuron-output directions, and attention heads' OV gain and
label preservation on J-lens directions against rotated, readout and random populations. Report
the small models' answers plainly, including "no such band in four blocks" if that is what the
data says. This chapter is where the reader learns to read such evidence, not where we win.

**Epilogue.** What the paper does that we could not (SAEs, multi-token lenses, behavioural
experiments, post-training comparisons, counterfactual reflection training), one paragraph each,
in our words; what the small models leave open; a `::: dictionary` (the pipeline builds it from
every `::: names` table of the book, as the main epilogue does: check `book/chapters/10-epilogue.md`).

## The voice

Hohm's lecture notes, as in PicoGPT: every object a map between named spaces; index and matrix
forms side by side; three names; finite sets first; dry wit, no hype, no anthropomorphism beyond
what is defined. Every ML term defined in the reader's language before use. Never type a number
a figure can compute; load-bearing prose numbers go through `{{value:name}}` (add names to
`proseValues` in `jspace/edition.js`, read from the artifacts). Each chapter: `# Title {#ch-slug}`,
`::: claim`, `::: names`, sections with ids, asides (history, "in the code", "in the paper"), 3–5
exercises. Cite the paper by section name in asides ("The paper's Methods define…"). Refer to
PicoGPT's chapters by name and web link
(https://tobiasosborne.github.io/PicoGPT.jl/pulling-back/), not by `@ch-…` (the cross-reference
index is this book's own).

## Figures

Every figure answers one question (the catalogue has it). Rules: DESIGN.md §5–§10, and:
- Numbers only from `getContext('specimen')()` and `getContext('artifacts')().jspace.{specimen,shakespeare}`
  (the J-lens exports, whole). No fetches, no invented numbers, no hand-typed numbers.
- Value ink for stream columns and parameters; **covector ink for J-lens vectors, Jacobian rows
  read as covectors, and gradients**; the J-space component of a value is a value. Hatching, bar
  tips, as the primitives do.
- Reuse the primitives (`SpokeGlyph`, `MatrixGlyph`, `TriangleDots`, `HowToRead`, `Scrubber`,
  `scales.js`). A glyph this book introduces (a cone, a lens grid, a depth axis) gets a primitive
  in `jspace/figures/primitives/` and a "how to read" key in the first figure that uses it.
- Two modes, both server-rendered: `interactive` and `print` (static, complete, small multiples
  for anything that animates; no controls drawn). Print figures must fit ≤ 200 mm with caption.
- Direct manipulation: click a position, drag α, choose k with a hairline track, not dropdowns.
- Imports from outside `web/` use the aliases `#lib/*`, `#figures/*`, `#specimen` (set up).
- Check each figure at `/figures/<id>/`, `?print`, at phone width, at night. Run
  `cd web && BOOK_EDITION=jspace npm run dev` (port 5173; `BOOK_STRICT=1` to fail on problems) and
  `node web/scripts/shots.mjs /figures/<id>/ name --print` for screenshots.

## Working rules

- Stay inside `jspace/book/chapters/NN-*.md` for your chapters and `jspace/figures/NN-*/` (plus
  `jspace/figures/primitives/` for shared glyphs; coordinate by naming them clearly). Do not edit
  `web/`, `book/`, `src/`, `print/`. Do not commit. Add prose values to `jspace/edition.js`
  `proseValues` only (keep the edits minimal and merge-safe).
- Delete the stub `00-prologue.md` / `fig-j-stub.svelte` if they are in your set.
- Build strictly before reporting: `cd web && BOOK_EDITION=jspace BOOK_STRICT=1 npm run build`
  must pass with your chapters; `node print/build.mjs --edition=jspace --proof` must produce the
  PDF; look at your pages in `jspace/proof/pages/`.
- Report back in `jspace/briefs/report-<your chapters>.md`: what you built, how you verified it,
  what is unfinished, decisions others must know (new primitives, prose values added, claims you
  could not support from the data and therefore did not make).

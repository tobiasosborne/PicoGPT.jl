# Report: chapters 0–2 of *Poised to speak* (prologue, the logit lens, the averaged Jacobian)

Author of `jspace/book/chapters/00-prologue.md`, `01-lenses.md`, `02-jacobian.md` and of the figures
in `jspace/figures/00-prologue/`, `01-lenses/`, `02-jacobian/`. Nothing committed. The stub prologue
and `fig-j-stub.svelte` are gone.

## What I built

**Chapter 0, Prologue** (`#ch-prologue`). The paper's claim in plain words (J-lens vector as an
averaged first-order push towards saying *a*, now or later; J-space; the workspace analogy and its two
kinds of evidence); the one idea, *replace the rest of the model by its averaged derivative*, with the
lens formula; the two models; depth notation (X₀, X_{ℓ−½} after attention of block ℓ, X_ℓ after its
MLP; X_{1/2} = X_mid for the specimen); the shape change V < D (7 vectors in ℝ⁸, 65 in ℝ¹²⁸: span a
subspace, a state is in general not a combination of them) against the paper's V ≫ D; what we cannot
do (behavioural experiments; no position on philosophy, as the paper); how to read the book (prereq
chapters of PicoGPT by web link, two inks). Names table: column of the stream / state / "activation";
depth / Euler time / "layer"; W_out / measurement / "unembedding W_U"; J_ℓ / averaged linear response /
"the lens matrix"; v_a, an averaged gradient / covector / "J-lens vector". Two asides (paper sections;
global workspace theory: Baars 1988, Dehaene–Kerszberg–Changeux PNAS 1998 with DOI). Four exercises.

**Chapter 1, Reading the stream early** (`#ch-lenses`). The stream at every depth (recursion in
half blocks, all depths in one space); the readout R: ℝ^D → Δ_V as a map (defined everywhere; reads a
direction: R(x + c𝟙) = R(x), R(cx) = R(x) at ε = 0, with an ε/σ² correction "negligible as long as
nothing shrinks the column; @ch-jlens meets a case where something does"); the logit lens (= identity
in place of the rest of the model; exact at L; at depth 0 a bigram model with a clock); a worked
disagreement that proves the information is in the column: window 1, position 7, depth ½, where the
logit lens says *t*, the model *r*, and everything after depth ½ in the specimen acts column by
column, so the model's *r* is a function of that column alone (checked in `proseValues`, which throws
if the lens ever agrees there); why it fails (residual sum; coordinates not preserved; only F_ℓ
translates, and it is nonlinear and mixes positions); the Shakespeare curve. Asides: readout in
`src/model.jl` (`gpt`), the paper's "Comparison to Related Techniques", history (nostalgebraist 2020,
tuned lens arXiv:2303.08112 and the paper's "skips ahead" criticism). Code: `residual_trace`,
`lens_logits`. Four exercises.

**Chapter 2, The averaged Jacobian** (`#ch-jacobian`). F_ℓ on a window; its derivative in T×T blocks
K_{t′t} of size D×D, index and matrix form; causality (block lower-triangular, 36 blocks); what the
blocks are at each specimen depth (1: identity; ½: diagonal only, K_tt = I + D(MLP∘LN)(x_t), varying by
column; 0: attention fills the triangle); the average in **the scale-free convention**
J_ℓ = (BT)⁻¹ Σ_b Σ_t Σ_{t′≥t} K_{t′t}, its meaning, its relation to the paper's pair mean
(× 2/(T+1)) and why we sum (J_L = I); disposition vs use, with the specimen's spread; present and
future, J = J^self + J^future, one sentence each on the paper's self-only / future-only variants;
computing by pullbacks (row i of K_{t′t} is the pullback of e_i at t′; one pullback with e_i in every
column gives all target sums at once; D pullbacks per window, delivered at every depth on the way
down; J^self needs D·T); the Shakespeare computation; code: `pull_to_depth`, `jacobian_block`,
`averaged_jacobians`. Asides: retarded Green's function / static susceptibility; the paper's Methods
and "Methodological Details and Ablations" (we use the final depth, full gradients, the plain mean,
drop nothing). Five exercises.

**Figures** (all server-rendered in both modes, numbers only from the specimen adapter and the J-lens
exports, computed at render time):

| id | question | notes |
|---|---|---|
| `fig-j-idea` (+ key) | what "replace the rest by its averaged derivative" means | one column's stem through depths 0, ½, 1 (spokes), a bypass per depth through the Hinton glyph of J_ℓ to its lens readout, the model's readout at the foot; click a character / drag the window; point at a depth. Print = the same picture, all bypasses alike. Key: one row enlarged. |
| `fig-j-depths` | what the stream is at each depth | rows = depths, columns = positions, spoke glyphs on one scale over all windows; model readout under the last row; click a column (lengths of x and of both updates in the readout). |
| `fig-j-logit-lens` | what the readout says applied early | cells of 7 bars per depth × position; top character in ink if it agrees with the model, grey if not; next character underlined; "k of 8 agree" per row; the last row is verified to be the model's readout (max error < 1e-4) before it is labelled so. |
| `fig-j-logit-lens-depth` (+ key) | how reliable the logit lens is by depth (Shakespeare) | dots = agreement with the model, rings = accuracy, references at 1/V and at the model's own accuracy; depth axis; point at a depth (agreement, accuracy, top-3, KL). Key: the depth axis (first use). |
| `fig-j-blocks` (+ key) | what the derivative of the rest of the model is | the 36 raw blocks of window 1 as a triangle of Hinton glyphs, nothing above the diagonal; window 1's own average and the grand J beside it; one area scale for all depths; depth chosen on a depth axis; point at a block. Print: depth 0. Key: the top-left corner enlarged. |
| `fig-j-average` | how much one context's Jacobian differs from the average | the ten per-window J⁽ᵇ⁾ at depth ½ with their relative Frobenius distance from the mean, then J = J^self + J^future at each depth (future empty after ½); the split is checked numerically (2e−6, export rounding) and the readout says so. |
| `fig-j-pullback` | how J is computed | window 1, depths down, covector spoke glyphs of what arrives in each column when e_i is pulled back from one target column or from every column; covector Scrubber for the backward pass; the mean over sources and the grand J₀ with its row i in covector ink; click a bottom glyph / "every column" / a row of J; arrow keys. One spoke scale per depth row, labelled ("spoke 0.85"), because the seeds and the summed rows differ by an order of magnitude (DESIGN's precedent of explicitly labelled separate scales). Print: both seeds stacked, at t = 1. |

**Primitives** (new, in `jspace/figures/primitives/`, for anyone):

- `jdata.js`: `at(tensor, ...i)` (0-based, column-major), `matrixOf`, `fibre`, `rowsOf`, `argmax`,
  `frob`, `tv` (total variation), `depthLabel` (0, ½, 1, 1½…), `depthWords`, `depthScale`.
- `DepthAxis.svelte` (+ `depthAxisHeight`): the hairline depth axis with ticks 0, ½, …, L, optional
  "embed/attn/MLP" micro labels and "block b" brackets, `sel`, `onpick(j)`. Its "how to read" key
  is in `fig-j-logit-lens-depth` (first use); later figures need no key for it.
- `DistBars.svelte` (+ `distBarsWidth`): a distribution over the V characters as bars on a common
  baseline; top character labelled (optionally grey via `markMuted`, optionally with p), characters
  under the bars, the true next character underlined (or ticked, without characters), `muted`.
- `SpokeSparse.svelte`: the book's spoke glyph (same classes and encoding) that omits spokes shorter
  than `cut` px. `SpokeGlyph` draws a tip even for an exact zero, so a basis covector e_i or an
  emptied block shows as a blob on the hub; use this one where most entries are exactly zero.

## Prose values added (`jspace/edition.js`, one function `lensValues`, merged after `writingValues`)

`lensSpecimenColumns`, `lensLogitAgreeZero`, `lensLogitAgreeHalf`, `lensExampleModel`,
`lensExampleLogit`, `lensShPositions`, `lensShPerWindow`, `lensShWindows`, `lensShLogitAgreeZero`,
`lensShLogitAccZero`, `lensShAccModel`, `lensShLogitAgreeLast`, `jacWindowDevMin`, `jacWindowDevMax`,
`jacShSources`, `jacShPairs`, `jacShLength`. Two guards throw (and so fail the build) if the data
stop supporting the prose: the worked example must remain a disagreement, and the logit lens's
agreement at depth 3½ must stay below one half ("changes the top character at more than half of the
positions").

## How I verified it

- Read the follow-up section of `analysis-report.md` (2026-11-06) and the README after the re-export:
  every number my prose uses is unchanged by the scale-free convention (logit-lens metrics, sampling
  counts, specimen agreement counts; the per-window deviations are scale-invariant). The figures read
  `conventions.jacobian_factor` and divide the exported `blocks` by it (they are exported as
  ((T+1)/2)·K); J_L = I, self + future = J and the per-window mean = J were checked from the files.
- The export's batch column n = t + (b−1)T has window b starting at corpus position b − 1 (checked
  against `artifacts/specimen.json` batch); `fig-j-idea` also compares the export's model argmax with
  the specimen engine's at render time and says so in its readout if they ever disagree.
- `cd web && BOOK_EDITION=jspace BOOK_STRICT=1 npm run build`: passes (exit 0) with all eight chapter
  files present, and prints no warning for my figures (the a11y warnings of an earlier run are fixed).
- Dev server on :5181 (`BOOK_EDITION=jspace BOOK_STRICT=1 npx vite dev --port 5181`, now stopped); every figure
  looked at, at `/figures/<id>/` and `?print`, at 1440 px and at phone width (390 px) in night mode,
  with a Playwright script that waits for `data-figure-ready`; chapter pages checked for unexpanded
  values and for the code excerpts (all four functions resolve).
- `node print/build.mjs --edition=jspace --proof`: writes `jspace/PoisedToSpeak.pdf` (81 pages at the
  last run, with every chapter present). Its four warnings are all outside my chapters (fig-j-grid
  oversize, widows on pp. 35 and 39, a names block on p. 63). My chapters are proof pages 7–29; I
  looked at every figure page. `pdftotext` finds no unexpanded `{{value:…}}`.
- One strict web build failed in between, on `jsShTwentyFiveJ` (chapter 4's value, used before it
  was defined; the other author added it within minutes). The rebuild passed. Builds of the edition
  race while the authors work: a red build may be someone else's half-written value.

## Decisions others must know

1. **Convention for J (ch. 2 defines it).** J_ℓ = (BT)⁻¹ Σ_b Σ_t Σ_{t′≥t} K_{t′t} ("mean over sources of
   the sum over targets"); J_L = I; the paper's pair mean is (2/(T+1))·J_ℓ. Chapters 3–6 should use
   this J and refer back to @ch-jacobian.
2. **Forward references that chapter 3 must honour** (it is a stub as I write): ch. 1 says the ε
   correction is negligible "as long as nothing shrinks the column; @ch-jlens meets a case where
   something does"; ch. 2 says "@ch-jlens shows what that [ε] does to the lens when J_ℓ is shrunk by
   2/(T+1)"; the prologue's exercise 3 ends "@ch-jlens shows what ε > 0 does to it". The data for this
   lesson are `lenses.literal.*` (Shakespeare final error 0.19). If ch. 3 drops the lesson, these
   three sentences need editing.
3. **Pipeline gap:** `{{value:name}}` names must be letters only (`/\{\{value:([A-Za-z]+)\}\}/`). A name
   with a digit is silently left in the text, and `BOOK_STRICT` does not catch it. I hit this and
   renamed; no other chapter has one now. Worth a check in `web/src/lib/server/prose-values.js`
   (not mine to edit).
4. Names used: J^self_ℓ, J^future_ℓ (superscripts), J⁽ᵇ⁾ for one window's average, K_{t′t} with rows =
   targets, columns = sources (as attention: rows read).
5. `jspace/book/figures.md` still says "planned" for my figures; I did not edit it (outside my paths).
   Built as catalogued, except: `fig-j-logit-lens` has three rows (depth 1 *is* the model's readout,
   labelled as such) instead of a separate model row; `fig-j-average` adds the self/future split at
   every depth; `fig-j-pullback` uses per-depth labelled spoke scales.

## Claims I did not make

- Nothing about the J-lens versus the logit lens (chapter 3's), beyond the definition.
- No claim that per-window variation in the specimen *is* "use" in the paper's sense beyond what it
  measures: ten overlapping windows of one line differ from their mean by 16–53 % (Frobenius) at depth ½.
- No claim about the Shakespeare per-window Jacobian spread (`per_window_frobenius`): norms alone say
  little about direction, so I left it out.
- No vocabulary sizes or widths for the paper's models (not public); only V ≫ D, as the paper says.

## Unfinished

- Nothing required is missing. Films were not rendered (the pullback is film-ready: its clock is
  `attrs.t`).

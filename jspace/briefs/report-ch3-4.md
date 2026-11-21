# Report: chapters 3 and 4 of *Poised to speak*

Author of `jspace/book/chapters/03-jlens.md` (The Jacobian lens) and `04-jspace.md` (The J-space),
the figures in `jspace/figures/03-jlens/` and `04-jspace/`, and three shared primitives. Nothing
committed. All numbers in figures are computed at render time from the J-lens exports (and the
specimen engine for stream columns); all load-bearing numbers in the prose are `{{value:…}}`.
Built against the scale-free exports (J_L = I; `lenses.literal` present), re-read
`analysis-report.md` (section "2026-11-06 — scale-free J …") and `artifacts/README.md` before
fixing any prose number.

## What I built

### Chapter 3, The Jacobian lens (`#ch-jlens`)

Claim, names table (lens / susceptibility / probe / free propagation / top lens token), sections:
replacing the rest of the model (lens_ℓ(x) = R(J_ℓ x), in the notation R, F_ℓ of ch. 1 and the
J_ℓ formula of ch. 2; what J_ℓx is exactly — the averaged first-order effect of x *as a
disturbance*, now and later — and what the lens throws away); lens vectors are covectors (v_a =
J_ℓᵀ w_a written block by block as the averaged gradient of a's score summed over t′ ≥ t; probe
pairing; the exact LayerNorm form is left to ch. 5's `@sec-probe`, which already writes it out);
**at the last depth the lens is the model** (K_{t′t} = δ I ⇒ J_L = I; the paper's pair weighting
gives 2/(T+1)·I; LN_ε(sy) = LN_{ε/s²}(y); the ε lesson with the `lenses.literal` numbers: specimen
1.7×10⁻⁵, Shakespeare 0.19 with 99.4 % top-1 agreement, effective floor ≈ 0.17) — this honours the
three forward references of ch. 0–2; both lenses on the specimen; along a real text (the GREMIO
grid; the colon under the speakers' names); depth by depth (Shakespeare, reported as measured); in
the code (`lens_frame`, `lens_logits`); 5 exercises. Asides: linear response, tuned lens (Belrose et
al. 2023, arXiv:2303.08112), exact LayerNorm pointer, coordinate-free pairing, in the paper ×2.

### Chapter 4, The J-space (`#ch-jspace`)

Claim, names table (spectroscopy names for the physics column: mixtures of pure spectra, best
nonnegative fit, fitted abundances, residual, overcomplete set/coherent states, null model, number of
lines worth fitting), sections: from a list of vectors to a set (the covector→column identification
via the stream's Euclidean inner product, pointing to `@sec-steer`; cones; 𝒥_k = ∪ C_S; not a
subspace, not convex; why nonnegative and sparse; "sparse frame" defined once and dropped); the
nearest point (NNLS, KKT, ⟨x_J, x − x_J⟩ = 0 ⇒ Pythagoras, Moreau aside; the exact 3-D picture;
the four regions of the plane; the nearly antiparallel pair ␣/n at depth ½, cos −0.98); every subset,
exactly (quotes `nnls`, `exact_decomposition`; non-nested best subsets, 138 of 1440 cases;
cancelling coefficients); against chance (random frames; V < D shape change: span vs cone vs all);
sixty-five characters in 128 dimensions (quotes `nnomp`; occupancy; results reported plainly);
5 exercises. Asides: the paper's span/cone discrepancy in its appendix, Moreau (1962), Mallat–Zhang
(1993, doi:10.1109/78.258082) and gradient pursuit.

### Figures (8, plus 3 keys)

| id | question | notes |
|---|---|---|
| `fig-j-lens-vectors` (+ key "Reminder: covector spokes") | What are the J-lens vectors vs the readout rows? | covector spokes, one scale for all 28; cosine with w_a under each; depth 1 = readout rows (cos 1.00). Key picks the vector whose largest ± components point to opposite sides. |
| `fig-j-readouts` | What do both lenses read at each depth and position? | paired bars per cell: J solid, **logit open** (catalogue said "hatched ink"; open value bars instead — both readouts are values); tick under the next character; drag window; two blocks of 4 on phones. |
| `fig-j-grid` (+ key "How to read a lens grid") | What does the J-lens read along a real text? | **new glyph** `LensGrid`: type weight ∝ p (200 → 900); rows depths, columns positions, wraps in whole lines; web toggles lens, print sets both plates on one page. Depth-4 row labelled "4 = model" (J_L = I), so no separate model row. |
| `fig-j-lens-depth` | How do the lenses compare with model and text by depth? | three panels (agreement, accuracy, KL) on `DepthAxis`; J filled, logit open, **pair-weighted (literal) J-lens as crosses**; references: the model, its accuracy, 1/V. |
| `fig-j-cone` (+ key "How to read a cone") | What is a cone of two vectors, and its nearest point? | **the key figure, new glyph** `ConeGlyph`. Exact orthographic view of span{v_s, v_u, x} (3-D contains the whole problem): disc = the span, hatched cone (covector hairlines, drawn as real lines, print-safe), ghosts −v, dotted polar-cone edges, x above the plane, foot p, x_J, remainder dashed with right-angle marks, J-coordinates on the edges. Right: **the triangle of all 21 pairs** as small cones with the share left over (J-space k = 2 = their union; min in bold). Click a pair, a column, a depth; drag p in the plane. Fits computed in closed form; checked against the export (0.3191 for the default). |
| `fig-j-decomposition` | J-coordinates of a column as k grows? | one column at the three depths; dot area ∝ ‖c_a v_a‖ (one scale); missed share bar vs random tick per row. |
| `fig-j-variance` | How much of the specimen's stream is in the J-space, against chance? | three depths; mean captured share vs k, J vs random, columns faint; span ticks and 7/8. |
| `fig-j-occupancy` | **New id** (catalogue's `fig-j-variance` split in two): the Shakespeare curves | 3 × 3 small multiples, occupancy marked, k = 25 ticked, span ticks; 2 columns on phones. |

### Primitives (new, `jspace/figures/primitives/`)

- `cone.js` — pure geometry: `nnlsExact` (all supports, k ≤ 8), `planeOf`, `inPlane`, `nearestInPlane`
  (cone / edge / apex regions), `pairFit`, `polarEdges`, `hatchCone` (analytic clipping, no
  patterns), `axonometric` (orthographic view and its inverse on the plane), `dot`, `norm`.
- `ConeGlyph.svelte` — the cone glyph in its plane through any linear view `P(u,v)`.
- `LensGrid.svelte` — the typographic lens grid; module exports `lensGridLayout`, `gridChar`
  (␣, ¶), `weightOf`.
Reused: `jdata.js`, `DepthAxis`, `TickTrack` (ch. 0–2 and 5–7 authors), `CorpusStrip`,
`SpokeGlyph`, `HowToRead`, `scales.js` (main book).

## Prose values (`jspace/edition.js`)

One line `Object.assign(values, jlensValues(read));` after the others, and the function
`jlensValues` appended at the end of the file (re-read immediately before the edit; no name
collides with the other authors' values — checked). 46 names, prefixes `jl…` (ch. 3) and `js…`
(ch. 4), e.g. `jlLitErrShakespeare` (0.19), `jlLitErrSpecimen`, `jlLitAgreeLast`,
`jlEpsEffShakespeare`, `jlSpecAgree{Zero,Half}{J,Logit}`, `jlShGapMin/Max` (3.4/7.7),
`jlColon{J,Logit,Depth}`, `jsConePair`, `jsConeMiss`, `jsOpposite{Pair,Cos}`, `jsNotNested`,
`jsSpecTopOne`, `jsShOcc{Last,Early,EarlyDepth}`, `jsShTwentyFive{J,Random}`, `jsShSpan…`.

## Verification

- `cd web && BOOK_EDITION=jspace BOOK_STRICT=1 npm run build`: **exit 0**, with all eight chapter
  files present; every page and figure page prerendered.
- `node print/build.mjs --edition=jspace --proof --skip-build` (after that build): **exit 0**,
  `jspace/PoisedToSpeak.pdf`, 81 pages; my chapters are pp. ~31–50 of `jspace/proof/pages/`
  (looked at: pp. 37 grid, 38 lens-depth, 44 cone, and the contact sheet). Four warnings, three
  mine: two widows (p. 35 "nothing to correct.", p. 39 last line of §3.6) and "fig-j-grid: 433 mm
  tall … printed at 38 %". The page itself (p. 37) shows the grid at full column width with both
  plates in two blocks each, legible, on one page, so the warning appears to come from an earlier
  layout state of the figure (before it measured its print width); I did not find the cause. The
  fourth warning (p. 63, names block) is not in my chapters.
- Own dev server (port 5182) and `web/scripts/shots.mjs`: every figure at `/figures/<id>/` and
  `?print` (wide), phone width (cone, grid, readouts, lens-depth), night (cone); both chapter pages
  full-length. Fixed on the way: a sign error in the polar-cone edge, four display equations that
  overflowed the column (split), label collisions (keys of cone/grid/covector, lens-depth,
  decomposition, variance, occupancy, cone coefficients), print grid wrapping into 4 blocks per
  plate.
- Numbers: the engine's stream columns reproduce the export's `j_space[j].norm` to 4×10⁻⁶ relative
  (window b ↔ corpus start b − 1); coefficient-based distances reproduce `distance` to the printed
  digits; my closed-form k = 2 fit reproduces the export's exact one. Every prose claim was checked
  against the arrays (two were tightened after checking: depths 2½–3 lead random frames for a few
  vectors before falling behind; "the best single vector at depth 1 is the predicted row" holds in
  75 of 80 columns, now stated with that count).

## Decisions others must know

1. **Pipeline gap (also found by the ch. 0–2 author):** `{{value:name}}` names must be letters only;
   a name with a digit is silently left in the text and `BOOK_STRICT` does not catch it
   (`web/src/lib/server/prose-values.js`). I renamed mine.
2. **Covector → column.** Chapter 4 needs v_a as columns before chapter 5 discusses the conversion.
   Ch. 4 takes the Euclidean identification "as the paper does" and points to `@sec-steer`, which
   "examines it as a choice". Ch. 5's sentence "This chapter makes the choice" now reads slightly
   late; suggest "This chapter examines the choice @ch-jspace made silently" (not my file).
3. **Catalogue (`jspace/book/figures.md`) not edited** (outside my paths): add `fig-j-occupancy`
   (split from `fig-j-variance`); `fig-j-readouts` uses open bars, not hatched ink; `fig-j-grid` has
   no separate model row; `fig-j-cone` adds the 21-pair triangle; `fig-j-lens-depth` has three
   panels and the literal lens as crosses.
4. Cross-references into ch. 3–4 used by others all resolve (`@ch-jlens`, `@ch-jspace`,
   `@fig-j-lens-depth`); I reference `@sec-probe`, `@sec-steer`, `@ch-writing`, `@ch-workspace`.

## Claims not made

- That the J-lens is better than the logit lens in the Shakespeare model: it is worse at every
  intermediate depth on agreement, accuracy and KL, and the text says so; the paper's claim
  (interpretable intermediates) is stated as untestable here, not as confirmed or refuted.
- Any workspace band, plateau of occupancy or J-space advantage before the last depth (there is
  none; at the last depth the lens vectors are the readout rows, and the text says that is
  expected).
- Why the depth-0 lens vectors are several times longer than the readout rows (stated, not
  explained), and the variance of the Shakespeare final stream (not exported; the ε lesson is told
  with the exported discrepancy only).
- The colon in the GREMIO grid is presented as an easy corpus regularity that the J-lens surfaces
  and the logit lens does not at that depth, not as model cleverness.

## Unfinished

- The two print widows above (they move with other chapters' pagination).
- `fig-j-readouts` cells are dense at phone width (bars ≈ 2 px); legible but tight.
- Keys are laid out for the margin width; at very narrow widths they stack under the caption as
  designed but were only checked on one phone width.

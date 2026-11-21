# Report: chapters 5, 6 and the epilogue of *Poised to speak*

Author of `05-writing.md` (Reading and writing), `06-workspace.md` (Is it a workspace?),
`07-epilogue.md`, and their figures. Written against the **scale-free** exports (specimen
re-exported 08:06, Shakespeare 08:12 on 2026-11-06; `analysis-report.md` section "2026-11-06 —
scale-free J …" re-read before every number was finalised). Nothing committed.

## What I built

**Chapter 5, Reading and writing** (`#ch-writing`). Sections: one covector, two uses
(`#sec-two-uses`); the probe, with the exact relation between probe and lens logit
(`#sec-probe`: $z_a=\langle w_a,x\rangle/s(J_\ell x)+\text{const}$, $w_a=J_\ell^\top C\,\mathrm{diag}(\gamma)W_{\text{out}}^\top e_a$);
steering as the covector→vector conversion $v\mapsto v^\sharp$ (a choice of inner product) and a
step of gradient ascent on the averaged logit, first-order side effects through the Gram matrix,
**saturation** of steering, and ablation as a point $\alpha^*$ of the steering line (`#sec-steer`);
the swap with the pseudoinverse derived in two lines, what it keeps fixed and what it does not, and
why later positions see an edit only through attention still ahead (`#sec-swap`); the paper's
"privileged" test (`#sec-privileged`); code (`swap_coordinates`); 5 exercises.

**Chapter 6, Is it a workspace?** (`#ch-workspace`). How to read a structural statistic (control,
confound, scale) (`#sec-signatures`); band of depths: CKA, then the paper's per-depth lens statistics
(`#sec-band`); limited capacity: occupancy and span share against random subspaces, with the exact
$k/D$ expectation (`#sec-capacity`); broadcast across depth (MLP gain, `#sec-mlp-gain`) and across
positions (OV gain and label preservation against the rotated control, `#sec-ov`); verdict
(`#sec-verdict`); code (`linear_cka`, `structure_statistics`); 5 exercises.

**Epilogue** (`#ch-epilogue`): one paragraph each on SAEs, multi-token lenses (template lens via
Stein's identity, tied back to the $\sharp$ choice of ch. 5; oracle lens), behavioural experiments,
base vs post-trained comparisons, counterfactual reflection training (`#sec-beyond`); what the small
models leave open, each as a concrete computation (`#sec-open`); the `::: dictionary`
(`#sec-dictionary`). The dictionary builds: with every chapter present it lists 53 terms; it is
assembled from all chapters' `::: names` tables by `book.lua`, so it grows as the others' tables do.

**Figures** (all numbers computed at render time from `getContext('artifacts')().jspace.*` and the
specimen engine; no typed numbers):

| id | folder | question | notes |
|---|---|---|---|
| `fig-j-steer` | 05-writing | What does adding α v_a do to the output at that position? | rows = depths, small multiples over α (export grid −4…4; even α only at phone width); line through a's bars = p_a(α); open ring = exact **α→∞ limit** softmax(Wout LN(v_a)+bout), computed from the trained θ; ablation drawn as the point α* = −⟨v_a,x⟩/‖v_a‖² on each row's axis. Direct manipulation: corpus line, character row, α track. |
| `fig-j-swap` | 05-writing | What does a lens-coordinate swap do here and later? | slots aligned with the window; before (outline) / after (solid); G = VᵀV and c printed per depth; TV per position, "unchanged" when exactly zero (depths ½, 1). Default: window 1, position 4, ‘e’↔‘t’ (rewrites four later positions at depth 0). |
| `fig-j-swap-gremio` | 05-writing | **new** (catalogue had none): the one Shakespeare swap | α = 0, .25, .5, 1, 2 small multiples, characters that reach any top 4, rest summed; stacked on phones. |
| `fig-j-privileged` | 05-writing | Is the J-part of a concept the part that drives the output? | per depth × character: Δln p for v_a (hatched, covector), J-component (solid), remainder (open); under each: J share of ‖m_a‖² and **cos(v_a, r_a)**. |
| `fig-j-cka` | 06-workspace | Do the lens vectors of different depths agree? | Hinton squares of **1 − CKA** (CKAs all lie in 0.76–1, so area ∝ CKA would be uniform) below the diagonal, CKA numbers above; one scale for both models. |
| `fig-j-band` | 06-workspace | **new**: Is there a band of depths? | Shakespeare, 5 rows on one depth axis: agreement (J vs logit), excess kurtosis (J vs logit), effective dimension (singular directions of Wout J for 90 %), occupancy, span share (J vs random). Uses the shared `DepthAxis`. |
| `fig-j-gain` | 06-workspace | Do the MLPs amplify J-lens directions? | per block, both models: medians + quartiles of J-lens, Wout rows, previous-block neuron directions; random quartiles dashed. |
| `fig-j-heads` | 06-workspace | Do heads relay J-lens directions? | gain × label preservation per head, J (bar) joined to its rotated control (ring), Wout rows (square); both models; labels de-overlapped. |

Inks: J-lens vectors and statistics of the J-lens covectors in covector ink (bars, hatching);
stream columns, J-components, neuron directions and output distributions in value ink; controls
(logit lens, rotated, random) in near-black ink as rings/dashes. No third hue.

**Shared code.** New primitive `jspace/figures/primitives/TickTrack.svelte` (hairline track with
labelled ticks; click or drag to snap; ←/→/Home/End; interactive mode only). Chapter-local helpers
(not figures, ignored by the registry glob): `05-writing/jdata.js` (tensor access, interned
probability table, lens vectors; written before the shared `primitives/jdata.js` existed — the
chapter-6 figures use the shared one; the two could be merged) and `05-writing/CharRow.svelte` (a
row of vocabulary characters to choose from).

**Prose values** (`jspace/edition.js`, two minimal edits: `Object.assign(values, writingValues(read))`
and `…workspaceValues(read)`, each function appended at the end of the file):
`steerOneZero steerOneHalf steerOneOne steerFourOne steerAlphaMax privilegedJWins privilegedCases
privilegedOrthogonal privilegedExceptions gremioDepth gremioPair gremioBefore gremioAfter`, and
`wsDepths wsBlocks wsCkaInnerMin wsCkaLastMin wsCkaLastMax wsKurtosisPositive wsAccGapMin wsAccGapMax
wsEffDims wsRank wsOccupancy wsSpanBelow wsSpanExpected wsSpanRandomMin wsSpanRandomMax
wsSpecSpanExpected wsSpecSpanRandom wsSpanLast wsMlpJ wsMlpLogit wsMlpNeurons wsSpecMlpJ
wsSpecMlpLogit wsHeads wsHeadJMin wsHeadJMax wsHeadRotMin wsHeadRotMax wsHeadLogitMax
wsHeadPreserveWins wsHeadGainWins wsSpecHeadJ wsSpecHeadRot`. `privilegedExceptions` generates a
whole sentence from the data (which characters/depths, and whether cos(v_a, r_a) explains them), so
the prose stays true if the export changes.

## Findings the prose rests on (others may want them)

1. **Steering saturates, exactly.** For |α| → ∞ the edited column is αv_a + O(1) all the way to the
   final LayerNorm (every other map is behind a LayerNorm or bounded), so the output tends to
   softmax(Wout LN(v_a) + bout), independent of the column and of the depth. This explains every
   curve that rises and falls back (e.g. ‘o’ at depth 0: ≈1 at α = 1, limit 0.15). Under the new
   convention α = 1 makes the steered character the top one in 269/480 (depth 0), 278/480 (½),
   11/480 (1) of the (column, non-top character) pairs.
2. **Ablation = steering by α\*** = −⟨v_a,x⟩/‖v_a‖² (the one point of the line where the probe reads 0).
3. **J^future vanishes at depth ½ in the specimen** (checked: max |Jfuture[:,:,½]| = 0), so swaps
   and ablations at depths ½ and 1 leave every later position exactly unchanged; at depth 0 they
   propagate through attention. `fig-j-swap` shows this; the prose ties it to ch. 2's triangle.
4. **The privileged test is half arithmetic.** The remainder r_a of an exact NNLS fit is orthogonal
   to every lens vector with a positive coefficient (KKT). cos(v_a, r_a) = 0 in 19 of 21
   (character, depth) cases, so the remainder cannot move the averaged logit of a at first order.
   The J-component beats the remainder in 19/21; the exceptions are ‘t’ at depth 0 (where v_t is
   *not* in the fit, cos = 0.39 — the remainder wins, as the arithmetic predicts) and ‘r’ at depth 0
   (cos = 0; the model departs from first order). The specimen's J share of ‖m_a‖² is large (29–99 %)
   versus the paper's 6–7 %. These concept results are scale-invariant and unchanged by the re-export.
5. **Shakespeare swap**: ‘H’↔‘r’ at depth 2, chosen for maximal disruption among 2,080 pairs; the
   prediction moves from a line break (0.90) to ‘a’ (0.70) at α = 1 and to ‘r’ at α = 2. The prose
   says plainly that this is not the paper's Soccer→Rugby experiment.
6. **Chapter 6, Shakespeare** (all from the new exports): no middle CKA block (CKA ≥ 0.91 among all
   depths before the last; the last depth 0.76–0.82 to the others); every per-depth statistic drifts
   towards its final value with no middle regime; J-lens trails the logit lens by 3.4–7.7 points of
   top-1 accuracy at every intermediate depth; excess kurtosis positive only at depth 4; effective
   dimension 16 → 25, then 33 of 65; occupancy 1,1,1,1,1,2,3,5,37; J span share below the random
   65-dim span (expected exactly 65/128 ≈ 0.51; measured 0.50–0.52) at every depth before 3½; MLP
   gain of J-lens directions 1.07, 0.91, 0.90, 1.34 (below Wout rows in blocks 2–4, below random in
   2–3); **attention heads: J label preservation 0.05–0.32 in all 16 heads vs −0.05–0.02 rotated**,
   J gain above rotated in 15/16. Specimen heads: J preservation (0.27, −0.05) matched by rotated
   (0.26, −0.08).
7. **A caveat I state in ch. 6 and leave as an exercise/open question**: J-lens vectors at the input
   of block b are defined by pulling back through block b (its OV maps included), so alignment with
   OV can be inherited by construction; the rotated control cannot separate inheritance from
   arrangement. A control that could (J computed with the head's path removed) was not computed.

## Verification

- `cd web && BOOK_EDITION=jspace BOOK_STRICT=1 npm run build` on the real repository: **exit 0**
  (last run after all eight chapter files existed; every page and figure page prerendered). An
  earlier run failed only on another author's then-unfinished prose value
  (`lensLogitAgreeZero`), which breaks every figure page because the figure route renders every
  chapter; while that lasted I verified my chapters with a scratch root (`BOOK_ROOT` with symlinks
  to the repository's `web/`, `artifacts/`, `src/`, `jspace/{edition.js,figures,artifacts,src,traces}`
  and only chapters 05–07): strict build exit 0. Nothing was written into the repository by that.
- On the full-book dev server my three chapter pages render with no unresolved cross-references and
  no maths errors; all references into chapters 1–4 (`@ch-lenses`, `@ch-jacobian`, `@ch-jlens`,
  `@ch-jspace`, `@fig-j-lens-depth`) and between my chapters resolve.
- Figures checked with `node web/scripts/shots.mjs` (own dev servers on ports 5187/5188) at
  `/figures/<id>/` and `?print` for all eight; at phone width `fig-j-steer`, `fig-j-swap-gremio`,
  `fig-j-band`; at night `fig-j-swap`, `fig-j-heads`. Fixed on the way: label collisions in
  `fig-j-privileged` and `fig-j-steer`, crowded bars in `fig-j-swap-gremio` (rebuilt, stacked on
  phones), clipped labels at phone width in `fig-j-steer` and `fig-j-band`, and five display equations
  that overflowed the text column (split, or reduced to the part that matters).
- Print: first from a scratch copy of the script on the chapters-only root (36 pages; fixed a display
  that ran into the margin and a chapter-6 names table that did not fit its opening page, after which
  it reported no warnings), then the real run, `node print/build.mjs --edition=jspace --proof`
  (with `PRINT_WORK` in my scratch directory so it could not collide with another author's run):
  **exit 0, `jspace/PoisedToSpeak.pdf` 81 pages**, proof in `jspace/proof/pages/`. My chapters are
  pp. 51–81 (5 from p. 51, 6 from p. 63, epilogue from p. 75, dictionary p. 79). Its three warnings
  are all in chapter 3 (`fig-j-grid` 222 mm tall, printed at 88 %; widows on pp. 35 and 39).
- The dictionary in the full build has entries from every chapter and merges senses across them
  (e.g. "probe": chapter 3 and chapter 5). Some entries from chapters 0–4 show doubled quotation
  marks (“"J-lens"”) because those tables put quotes inside the ML cell; the pipeline adds its own.

## Unfinished, and claims not made

- `fig-j-steer` shows the edited position only; later positions under steering exist in the export
  for window 1 only (`steer_later_window1`), and the later-position story is told by `fig-j-swap`
  (all columns). A later-positions panel for window 1 could be added.
- I did not update `jspace/book/figures.md` (outside my files): ch. 5 adds `fig-j-swap-gremio`; ch. 6
  adds `fig-j-band`; `fig-j-cka` draws 1 − CKA rather than CKA (reason above).
- Not claimed: anything about the paper's models beyond what its text states (its numbers are cited
  as its measurements: occupancy ≈ 25, ≤ 10 % excess variance, MLP gain ≈ 10×, recall@25 0.67 vs 0.86,
  6–7 % J share, ~15 % of SAE features). Not claimed: that the heads' label preservation is
  broadcast (only that it is far above the rotated control, with the construction caveat). Not
  claimed: why the J-lens trails the logit lens; the epilogue proposes J^self for Shakespeare as the
  test (it was omitted from the export).
- No frontispiece traces for `writing`, `workspace`, `epilogue` (the slot shows the dev placeholder).

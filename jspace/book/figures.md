# Figures of *Poised to speak*

The catalogue: what each figure answers, how it encodes, how the reader handles it, its data (paths
are into `getContext('artifacts')().jspace.specimen` / `.shakespeare`, see `jspace/artifacts/README.md`),
and how it prints. Rules in `jspace/briefs/writer.md` and `DESIGN.md` §5–§10. Ids are `fig-j-*`.
Every figure below is **built** (6 November 2026) unless marked otherwise; the authors' reports in
`jspace/briefs/report-ch*.md` record where a built figure departs from its row (encodings chosen
against the data, keys added). Three ids were added while building: `fig-j-swap-gremio`,
`fig-j-band` and `fig-j-occupancy` (split from `fig-j-variance`, which keeps the specimen).

## 0 · Prologue (`00-prologue`)

| id | the one question | encoding | interaction | data | print |
|---|---|---|---|---|---|
| `fig-j-idea` | What does "replace the rest of the model by its averaged derivative" mean? | the model as a stem (reuse `ResidualStem`/`stem.js`): depths marked on the stem; from depth ℓ, two routes to the readout: the real rest of the model (the stem, value ink) and the straight line "J_ℓ then readout" (a hairline bypass); the lens's readout beside the model's at the end, for one column; key: depth marks | click a depth | specimen: depths, lens probs, model probs for one column | the three depths as small multiples |

## 1 · Reading the stream early (`01-lenses`)

| id | question | encoding | interaction | data | print |
|---|---|---|---|---|---|
| `fig-j-depths` | What is the stream at each depth? | one window's 8 columns at depths 0, ½, 1 as spoke glyphs (shared scale), rows = depths, columns = positions; the readout's probabilities under the last row | drag the window; click a column | specimen `forward` (book artifact) or the J-lens export's stream states | static |
| `fig-j-logit-lens` | What does the readout say if applied early? | small multiples: rows = depths, columns = positions; in each, the V probabilities as bars; the model's own prediction as the last row; the true next character marked | drag the window | specimen logit-lens probs per depth | one window |
| `fig-j-logit-lens-depth` | How reliable is the logit lens by depth, in the Shakespeare model? | nine depths on a hairline axis; top-1 agreement with the model's output as dots; accuracy against the true character as rings; the uniform-guess line | point at a depth | shakespeare lens statistics | static |

## 2 · The averaged Jacobian (`02-jacobian`)

| id | question | encoding | interaction | data | print |
|---|---|---|---|---|---|
| `fig-j-blocks` | What is the derivative of the rest of the model? | the 36 blocks K_{t′t} of one window at depth 0 as `MatrixGlyph`s laid out in a triangle (rows = targets t′, columns = sources t; nothing above the diagonal, as attention); the grand average J as one larger glyph; key: Hinton glyph reminder, triangle | click a depth (0, ½, 1); point at a block to read its indices | specimen `jacobian.blocks` | depth 0 triangle + J |
| `fig-j-average` | How much does one context's Jacobian differ from the average? | the 10 per-window averages at depth ½ as small multiples, the grand average, and J^self and J^future beside it, shared scale; ‖K_b − J‖ under each | point | specimen per-window averages, J, variants | static |
| `fig-j-pullback` | How is J computed? | the pullback as a figure: a covector e_i placed at target columns (covector ink, up the stem), arriving at every source column; the row it produces drawn into J; the "e_i in every column" trick as the same picture with all columns lit; scrub forward→backward | scrub | specimen one window, rows of K via the artifact | two states side by side |

## 3 · The Jacobian lens (`03-jlens`)

| id | question | encoding | interaction | data | print |
|---|---|---|---|---|---|
| `fig-j-lens-vectors` | What are the J-lens vectors, and how do they differ from the readout's rows? | for each depth: the 7 covectors v_a as `SpokeGlyph`s in covector ink (bar tips), labelled by character; beside them the 7 rows of Wout (the logit-lens covectors); shared scale; the cosine between v_a and the Wout row under each; key: covector glyph reminder | click a depth | specimen `lens.vectors`, `lens.logit_vectors` | all depths |
| `fig-j-readouts` | What do the two lenses read at each depth and position? | the grid: rows = depths (0, ½, 1, model), columns = positions of a window; in each cell V bars, J-lens solid, logit lens hatched ink; the true next character marked; the last row is the model | drag the window; click a cell for numbers | specimen lens probs per depth | one window |
| `fig-j-grid` | What does the J-lens read along a real text, depth by depth? | the "GREMIO" prompt set as a line of characters; under each, a column of 9 cells (depths) holding the top J-lens character, weight of type ∝ its probability; the model's prediction as the bottom row; the logit lens on hover/toggle; the true next character in ink | point at a cell for top-3; toggle lens | shakespeare `grid` | static, both lenses as two plates |
| `fig-j-lens-depth` | How do the lenses compare to the model and the truth by depth? | nine depths; four curves as dots on hairlines: J-lens agreement, logit-lens agreement, J-lens accuracy, logit-lens accuracy; KL as ring size | point | shakespeare lens statistics | static |

## 4 · The J-space (`04-jspace`)

| id | question | encoding | interaction | data | print |
|---|---|---|---|---|---|
| `fig-j-cone` | What is a cone of two vectors, and what is the nearest point of it? | **key figure, new glyph**: the plane of two actual J-lens vectors v_s, v_u (depth ½); the cone between them shaded by hairline rules; an actual column x projected into the plane; its nearest point x_J on the cone (a drop of the perpendicular, or a vertex); the remainder as a dashed leader; the same with −v drawn ghosted to show a subspace would include it | drag x within the plane; pick the pair | specimen `jspace` exact decomposition (k = 2) | static |
| `fig-j-decomposition` | What are the J-coordinates of a column, as k grows? | one column at one depth: for k = 1…7, a row of V bars (the nonnegative coefficients, value ink), and at the right the remainder ‖x − x_J‖/‖x‖ as a bar; the chosen subset's characters bold | click a column / depth; k on a track | specimen `jspace` | static, one column |
| `fig-j-variance` | How much of the specimen's stream lies in the J-space, against chance? | three depths: mean captured share vs k (exact, k = 1…7), J frame (value ink) against random frames (ink hairline), single columns faint; span ticks and the 7/8 mark | point at k | specimen `j_space` | static |
| `fig-j-occupancy` | How much of the Shakespeare stream lies in the J-space, and where does the lens frame stop beating chance? | nine depths as small multiples: residual share vs k (greedy, k = 1…65), J against random, the occupancy marked, k = 25 ticked, span ticks | point at k | shakespeare `j_space` | static |

## 5 · Reading and writing (`05-writing`)

| id | question | encoding | interaction | data | print |
|---|---|---|---|---|---|
| `fig-j-steer` | What does adding α v_a do to the output? | for one column: the V probabilities as bars across α = −4…4 (small multiples on a hairline α axis), the target a in bold; below, p_a against α as a curve; ablation as the α at which ⟨x, v̂_a⟩ is removed, marked | drag α; click a; click a column | specimen `interventions.steer`, `ablate` | α ∈ {−4, 0, 4} |
| `fig-j-swap` | What does swapping two concepts in lens coordinates do, here and later? | before/after bars at the edited position and at every later position of the window; the 2×2 Gram matrix and the coordinates c shown as numbers; the two characters chosen from the corpus strip | click s, u, the column | specimen `interventions.swap` | one swap |
| `fig-j-privileged` | Is the J-space part of a concept the part that drives the report? | per character: three bars (the mean change in log p_a when steering with v_a, with the J-component of the concept vector, with the remainder, equal size), per depth as small multiples | point | specimen `interventions.concept_mean_delta_logp` | static |
| `fig-j-swap-gremio` | What does one swap do in the Shakespeare model? | the final-position distribution of the GREMIO prompt before the swap and after it at each α, as bars; the two swapped characters named | point at α | shakespeare `interventions.swap` | static |

## 6 · Is it a workspace? (`06-workspace`)

| id | question | encoding | interaction | data | print |
|---|---|---|---|---|---|
| `fig-j-cka` | Do the lens vectors of different depths agree? | 1 − CKA between depths as a Hinton diagram (CKA itself is nearly uniform), both models side by side, depth labels on both axes | point | both exports' `cka` | static |
| `fig-j-band` | Do the paper's per-depth signatures show a band of depths here? | the per-depth statistics (agreement, accuracy, kurtosis, occupancy, span share) on one depth axis, each drifting to its final value | point | shakespeare `lenses.metrics`, `j_space` | static |
| `fig-j-gain` | Do the MLPs amplify J-lens directions more than other directions? | per block: the gain of J-lens directions (dots, value), neuron-output directions (rings), readout rows, with the random median at 1 as a hairline; quartiles as hairline bars | point | both exports' `broadcast.mlp_gain` | static |
| `fig-j-heads` | Do some attention heads relay J-lens directions? | scatter: gain (x) vs label preservation (y) per head, J-lens population filled, rotated/readout/random open; both models; heads labelled block.head | point | both exports' `broadcast.attention` | static |

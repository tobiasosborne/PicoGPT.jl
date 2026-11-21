# Brief: the J-lens analysis (Julia) for the J-space supplement

You are building the data layer of a supplement to the PicoGPT book. The supplement explains
Anthropic's paper *Verbalizable Representations Form a Global Workspace in Language Models*
(transformer-circuits.pub/2026/workspace) to a BSc physics/maths reader, by computing every object
in the paper's mathematics on two models we own completely:

- **the specimen** (V=7, D=8, H=2, L=1, T=8, 1007 parameters, trained on "to be or not to be"),
  from `artifacts/specimen.json` (`train.theta_final`, `batch`, `config`, `vocab`); and
- **the Shakespeare model** (V=65, D=128, H=4, L=4, T=256, F=512), from the checkpoint
  `artifacts/shakespeare/checkpoints/latest.bin` (loader: `scripts/common.jl` `load_checkpoint`;
  corpus and 90/10 split: `tiny_shakespeare`, `shakespeare_split`).

Read first: `BRIEF.md` §1–§3 and §12–13, `src/PicoGPT.jl`, `src/block.jl`, `src/model.jl`,
`src/parameters.jl`, `scripts/common.jl`, `scripts/export_specimen.jl`. The code you write will be
**quoted in the text** by the book's `::: {.code file="jspace/src/JLens.jl" fn="…"}` mechanism, so
write it in the style of `src/*.jl`: short functions, docstrings that state the mathematics in
index and matrix form, comments that say *why*. Every operation that needs a derivative uses the
book's pullbacks (`embed`, `layernorm`, `affine`, `attention`, `mix`, `relu`), never autodiff and
never finite differences except in tests.

## Paths you own

```
jspace/src/JLens.jl                      the analysis, one file (include'd by the scripts), `module JLens`
jspace/scripts/export_specimen_jlens.jl   → jspace/artifacts/specimen-jlens.json
jspace/scripts/export_shakespeare_jlens.jl→ jspace/artifacts/shakespeare-jlens.json
jspace/test/runtests.jl                   tests (run with `julia --project -t auto jspace/test/runtests.jl`)
jspace/artifacts/README.md                what each exported field is, with conventions
```

Do not touch anything outside `jspace/`. Do not commit. Use the main `Project.toml`
(`julia --project -t auto …`); stdlib only. Compute in Float64 (convert the Float32 checkpoint).

## 1. Depths

A model with L blocks has residual stream states at the **depths**
ℓ ∈ {0, ½, 1, 1½, …, L}: X_0 after the embedding; X_{ℓ−½} after the attention update of block ℓ
(the book's `Xmid`); X_ℓ after the MLP update of block ℓ. The final state X_L feeds the readout
Xf = LN(X_L; γf, βf), Z = Wout Xf + bout, p = softmax(Z). Specimen: depths 0, ½, 1. Shakespeare:
0, ½, 1, …, 4 (nine depths). Index depths in exports as `depth: [0, 0.5, 1, …]` with
`depth_kind: ["embed", "attention", "mlp", …]`.

You will need to run the model from any depth to X_L and pull back from X_L to any depth. Compose
it yourself from the book's primitives (the block is `layernorm → affine ×3 → attention → mix →
affine → residual add` then `layernorm → affine → relu → affine → residual add`; see `block` in
`src/block.jl` and reproduce exactly those semantics, splitting at the residual add). Keep the
pullback closures so that one forward pass serves many pullbacks.

## 2. The Jacobian and the averaged Jacobian (the heart of the paper)

For a sequence (window) b, source position t and target position t′ ≥ t, define the D×D block
K^{(ℓ)}_{t′t} = ∂X_L[:, t′] / ∂X_ℓ[:, t]. Compute it by pullbacks: row i of K_{t′t} is column t of the
pullback of the cotangent e_i placed at column t′ of X_L (and zero elsewhere). Blocks with t′ < t
are exactly zero (causality); assert this on the specimen.

The paper's averaged Jacobian is
    J_ℓ = mean over (b, t, t′ ≥ t) of K^{(ℓ)}_{t′t}   (uniform weight on every pair t′ ≥ t).
Because the pullback is linear, placing e_i in **every** column of X̄_L gives, at column t, the
sum over t′ ≥ t of row i of K_{t′t} in one pullback: D pullbacks per window give J. Use this.
Also compute the two limiting variants the paper discusses:
    J^self_ℓ  = mean over (b, t) of K_{tt}        (t′ = t only; "closer to the logit lens")
    J^future_ℓ = mean over (b, t, t′ > t) of K_{t′t} (strictly later targets; "the broadcast part")
For the specimen these are cheap and exact. For Shakespeare compute J exactly as above, and compute
J^self / J^future on a subsample of target positions only if the whole script stays under about
25 minutes on 12 cores; otherwise omit them and say so in the README.

**Sanity facts to check and export** (they appear in the text):
- At depth L, K_{t′t} = δ_{t′t} I, so J_L = (2/(T+1)) I exactly, J^self_L = I, J^future_L = 0.
- Hence the J-lens at depth L coincides with the model's own output (up to LayerNorm's ε).
- Finite-difference check of a few blocks K_{t′t} on the specimen (relative error < 1e-6).

Export for the specimen: J, J^self, J^future at every depth (D×D); **every** block K_{t′t} for all
t′ ≥ t for ONE window (choose `window = 1`, and say which) at every depth (36 blocks of 8×8 each);
and the per-window averages (10 windows × D×D) at depth ½, so a figure can show how much a single
context's Jacobian differs from the average. Export for Shakespeare: J (and the variants if done)
at every depth, from at least 64 windows of the **validation** text (never trained on), windows
spaced evenly; record how many windows and positions were averaged. Also export, per depth, the
singular values of J and of Wout·J.

Shakespeare averaging: average over all source positions t = 1…T (do not drop early positions;
if you find heavy-tailed outliers worth noting, note them in the README; do not silently filter).

## 3. The lenses

    lens_ℓ(x) = softmax(Wout · LN(J_ℓ x; γf, βf) + bout)         the Jacobian lens (J-lens)
    logitlens(x) = softmax(Wout · LN(x; γf, βf) + bout)           the logit lens (J = I)

The **J-lens vectors** at depth ℓ are the rows of Wout J_ℓ: v_a = (Wout J_ℓ)[a, :] ∈ ℝ^D, one per
character a. (They are covectors: v_a = J_ℓᵀ Woutᵀ e_a is the averaged pullback of logit a. The
text will say so.) The logit-lens vectors are the rows of Wout. Export both at every depth
(V×D), plus the cosine-similarity Gram matrix of the v_a at each depth.

Specimen: export, for all 80 token columns of the training batch, at every depth: the J-lens
probabilities (V×80), the logit-lens probabilities, and the model's own probabilities (once).
Shakespeare: over ≥ 2000 token positions of validation windows, per depth and per lens, export:
top-1 agreement with the model's own prediction; top-1 and top-3 accuracy against the actual next
character; mean KL(model ‖ lens); mean excess kurtosis of the lens logits. Also export the
"lens grid" for the prompt
`"GREMIO:\nGood morrow, neighbour Baptista.\n\nBAPTISTA:\nGood morrow, neighbour Gremio.\n"`
(the book's attention prompt): for every position and depth, the top-3 characters and their
probabilities under the J-lens and under the logit lens, and the model's top-1 at the end.

## 4. The J-space

Given the V J-lens vectors at a depth and a sparsity k, the J-space is the union over k-subsets S
of the cones {Σ_{a∈S} c_a v_a : c ≥ 0}. For an activation x define
    d_k(x) = min_{|S|=k} min_{c ≥ 0} ‖x − Σ_{a∈S} c_a v_a‖,
the J-component x_J = the minimiser, the remainder x − x_J. Implement nonnegative least squares
(Lawson–Hanson active set is fine; test it against brute force on small cases).

Specimen (exact): V=7 vectors in ℝ^8, so enumerate all C(7,k) subsets for k = 1…7. For every one
of the 80 columns, at every depth, for every k: the chosen subset, the coefficients, ‖x‖, ‖x−x_J‖.
Control: the same with V random unit vectors in ℝ^8 (fixed seed, 20 draws, report the mean of
‖x−x_J‖²/‖x‖² over draws, per k and depth). Note and export the fact that V < D here: the v_a
span at most a 7-dimensional subspace, so even k = V leaves a remainder (the orthogonal complement
of the span); report the fraction of ‖x‖² in span{v_a} vs in a random 7-dimensional subspace.

Shakespeare (greedy): 65 vectors in ℝ^128. Use nonnegative orthogonal matching pursuit (add the
vector with the largest positive inner product with the residual, refit NNLS on the active set,
repeat) for k = 1…65 on ≥ 2000 validation activations per depth. Export per depth: the mean
residual fraction ‖x−x_J‖²/‖x‖² as a function of k, for the J-lens frame and for a random
unit frame of 65 directions (fixed seed, mean of 5 draws); the **occupancy** = the smallest k at
which the J frame's marginal improvement (from k−1 to k) falls below the random frame's; and the
fraction of ‖x‖² in the full span{v_a} (plain least squares, k=65) vs a random 65-dimensional
subspace. Also export the median over positions of the number of nonzero coefficients at k=25.

## 5. Reading and writing (interventions), specimen first

Fix a depth ℓ (do all three) and a column (window b, position t). Replace x = X_ℓ[:, t] and run
the rest of the model forward, reporting the output distribution at position t and at every later
position of the window (an edit at t can change what later positions read through attention).

- **Steer**: x ← x + α v_a, α ∈ {−4, −3, …, 4} (plus 0). For every column, every a, every α:
  the probability vector at position t. (80 × 7 × 9 × 7 numbers.)
- **Ablate**: x ← x − ⟨x, v̂_a⟩ v̂_a (project out the unit direction). For every column and a.
- **Swap in lens coordinates**: for characters s ≠ u, V = [v_s v_u], c = V⁺x (pseudoinverse,
  write it as least squares: V⁺ = (VᵀV)⁻¹Vᵀ), x′ = x + V(σ(c) − c) with σ the swap of the two
  entries. For every column and every unordered pair: the probabilities after the swap at position
  t and at the later positions. Also export the 2×2 Gram matrix VᵀV and the coordinates c for the
  reader to see.
- **Privileged for report?** (the paper's Fig. 8, in miniature). For each character a, define a
  concept vector m_a = mean of X_ℓ[:, n] over the columns n whose *target* is a, minus the mean
  over all columns. Split m_a = m_a^J + m_a^⊥ with m_a^J the exact J-component at k = 3. For every
  column n with target ≠ a, add each of m_a^J, m_a^⊥ and v_a, each rescaled to the same norm
  (‖m_a‖), and record the change in log p_a at that position. Export the mean change per
  character and per component, per depth.

Shakespeare: one example only. On the prompt above, at depth 2 (the middle), swap the lens vectors
of two characters at every position (choose a pair where the model's top-1 at the last position
changes, and say how you chose) and sweep α ∈ {0, ¼, ½, 1, 2} as a scale on (σ(c) − c). Export the
final-position distributions before and after. If no pair changes the top-1, export the pair with
the largest change and say so.

## 6. Structure: is it broadcast?

- **MLP gain.** For the MLP of block ℓ, which reads X_{ℓ−½}: gain(v) = ‖MLP(v)‖ for a unit v
  (the MLP includes its LayerNorm, so scale does not matter), normalised by the median of
  ‖MLP(r)‖ over 2000 random unit r. Populations: J-lens vectors at depth ℓ−½; rows of Wout
  (logit-lens vectors); the columns of W2 of block ℓ−1 (neuron output directions; skip for ℓ=1);
  random. Export the median and quartiles per population and block.
- **Attention OV.** For head h of block ℓ, W_OV = Wo[:, rows of head h] · Wv[rows of head h, :]
  (D×D). For a population P of unit vectors at depth ℓ−1: gain = mean ‖W_OV v‖ / mean ‖W_OV r‖
  over random unit r; label preservation = mean over i of the reciprocal rank of cos(W_OV v_i, v_i)
  among {cos(W_OV v_i, v_j)}_j, minus the same statistic on random directions. Populations: J-lens
  vectors at depth ℓ−1; the same vectors under one fixed random orthogonal rotation; rows of
  Wout; random. Export per head.
- **Similarity of depths.** Linear CKA between the matrices Wout J_ℓ (V×D, columns centred) for
  every pair of depths; export the matrix per model. Also the top-1 agreement between the J-lens
  and the logit lens per depth (Shakespeare).

## 7. Export format and provenance

Follow `scripts/common.jl`: `{shape, data}` column-major tensors, 1-based ids, `writejson(…; sig = 6)`
for the specimen and `sig = 5` for Shakespeare. Each file carries `description`, `model` (config,
parameter count, source checkpoint/step), `conventions` (as in `artifacts/specimen.json`), the
vocabulary, the depths, how many windows/positions were averaged, seeds, and wall-clock time.
Every number a figure might show must be in the file; nothing is to be typed by hand later.
Keep `specimen-jlens.json` under ~3 MB and `shakespeare-jlens.json` under ~6 MB (reduce sample
counts, not precision, if needed; say what you reduced).

## 8. Tests

`jspace/test/runtests.jl` (Test stdlib): the finite-difference check of K blocks; causality (zero
blocks); J_L = 2/(T+1)·I, J^self_L = I; lens at depth L equals the model's probs to 1e-9;
NNLS against brute force; the pseudoinverse swap leaves the component orthogonal to span{v_s,v_u}
unchanged and exchanges the two coordinates; CKA(X, X) = 1 and invariance to orthogonal rotation.

## 9. Report back

When done, write `jspace/briefs/analysis-report.md`: what you built; how you verified it (test
output, timings); every number you think the author will want in the prose (with the JSON path it
lives at); anything unfinished; any decision others must know (e.g. which window/pair you chose and
why; whether J^self/J^future exist for Shakespeare). Be exact and dry. No hype.

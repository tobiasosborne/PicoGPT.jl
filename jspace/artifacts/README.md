# J-lens exports

Reproduce from the repository root with Julia 1.13 and stdlib only:

```sh
julia --project -t auto jspace/test/runtests.jl
julia --project -t auto jspace/scripts/export_specimen_jlens.jl
julia --project -t auto jspace/scripts/export_shakespeare_jlens.jl
julia --project -t auto jspace/scripts/export_specimen_jlens_training.jl
```

All three scripts include `scripts/common.jl` for `tensor`, `writejson`, and, where
applicable, the checkpoint loader and corpus split. The two static analyses
perform no training: the specimen uses `train.theta_final` and the stored batch.
The training export replays the canonical 200 updates from stored `theta0`.
The checkpoint is converted to Float64 before any forward pass. BLAS runs on
one thread; Julia parallelises independent windows or depths over 12 threads.

## Conventions and provenance

Every multidimensional numeric array is `{shape, data}`, in Julia column-major
order. Numeric vectors are plain JSON arrays. Indices and character ids are
1-based; array indices in the descriptions below are mathematical/Julia indices.
The `vocab` array maps character ids to strings. The depth index `j` corresponds
to `depth[j] = (j-1)/2`; `depth_kind[j]` names the preceding update.

Both files contain `description`, `model` (config, parameter count, source,
step, SHA-256 of source), `conventions`, `vocab`, `depth`, `depth_kind`, `sampling`,
`seeds`, and `wall_seconds`. Shakespeare also records the corpus hash and stage
`timings`. Timings start on entry to the export function and include subsequent computation
and compilation before JSON serialisation; process startup and compilation of
the export function itself are excluded.
The writer uses six significant digits for the specimen and five for Shakespeare.
Use the full-precision in-memory calculations, not rounded exports, for 1e-9 tests.

## Jacobians

`jacobians.averaged[i,q,j]` is the **mean over windows and sources of the sum
over causal targets** of `K = ∂X_L[i,target]/∂X_depth[j][q,source]`.
Its denominator is the number of source positions, BT, rather than the number
of causal pairs. This equals the literal uniform-pair average times (T+1)/2,
recorded as `conventions.jacobian_factor` (4.5 specimen; 128.5 Shakespeare).
All source positions, including the
first position in every window, are included. Each output coordinate needs one
pullback with that coordinate set to one at every output column. A cached
forward trace supplies all depths during the same reverse traversal.

`jacobians.self` averages diagonal K blocks over sources; `future` averages
the sum of strictly later K blocks over sources. Both use denominator BT, so
`averaged = self + future`, asserted before rounding. The previous export's
strict-pair mean for `future` is replaced by this source-normalised sum to make
the split additive; `self` is unchanged. They are exact for the specimen and
`null` for Shakespeare: target-by-target self pullbacks were omitted to keep the
64-window Jacobian, activation analyses and controls within the runtime budget.

The specimen additionally has:

- `blocks[i,q,target,source,j]`, **(T+1)/2 times** all causal K blocks of **window 1** of the source
  batch. Anti-causal entries are zero padding; separate tests actually evaluate
  those blocks. Divide by `conventions.jacobian_factor` to recover the raw
  derivative returned by `jacobian_block`. This export scaling follows the
  follow-up brief; the derivative itself has not changed. Window 1 inputs decode to `to be or`.
- `per_window_attention[i,q,b]`, the ten separate averages at depth 0.5.

`singular_values[rank,j]` contains all D singular values of J, descending;
`readout_singular_values[rank,j]` contains all min(V,D) singular values of Wout J.
Shakespeare's `per_window_frobenius[b,j]` and
`per_window_max_residual_column_norm[b,j]` expose variation and possible outliers
without discarding any early positions or windows.

At the final depth, J = I, Jself = I, Jfuture = 0. `sanity.final_j_error`
compares J_L to I; `final_lens_error` checks the final readout against the
model at ≤1e-9 before rounding. Specimen `self_future_error` checks additivity.
The existing `sanity.final_literal_lens_error` remains a compatibility alias
of `lenses.literal.final_max_probability_error`.
The specimen also includes `sanity.pullback_tests`, copied from the test-generated
`verification.json`: finite-difference step/error, number of anti-causal blocks
checked, and NNLS trial count. Run tests before exporting to populate this field.

## Lenses

`lenses.vectors[a,q,j] = (Wout J_j)[a,q]` are covectors represented as vectors;
`logit_vectors[a,q,j] = Wout[a,q]`. `cosine_gram[a,b,j]` compares character
covectors, including their sign. No centring or normalisation is applied to the
actual intervention vectors. Only the cone pursuit and explicitly unit-direction
statistics normalise columns; positive normalisation leaves a cone unchanged.

The J-lens is softmax(Wout LN(Jx; gamma_f,beta_f) + bout), using the
book's **fixed** epsilon=1e-5. The logit lens uses J=I. The specimen exports
`j_probs[a,n,j]`, `logit_probs[a,n,j]` and `model_probs[a,n]` at all 80 columns
(n=t+(b-1)T). `final_max_probability_error` measures the final discrepancy in
both static artifacts;
`j_logit_agreement[j]` is agreement between the two lenses' top predictions.

With the new J_L=I convention the final lens is exactly the model readout,
and the final covectors are the rows of Wout, without any epsilon adjustment.
The separate `lenses.literal` set retains only the old paper-weighted readouts,
using J/((T+1)/2) and the same fixed epsilon. For the specimen it contains
`j_probs[a,n,j]` and `final_max_probability_error`; for Shakespeare it contains
`metrics[j]` with exactly the same fields as the primary metrics, plus
`final_max_probability_error`. No literal vectors, grids or decompositions are
exported. For positive s, LN_epsilon(sx)=LN_(epsilon/s²)(x) before the learned
affine part; the tests document the resulting literal discrepancy and pass
the primary final-depth equality with no broken assertions.

Shakespeare has `metrics[j].j_lens` and `.logit_lens`, each with `positions`,
`model_agreement`, `top1_accuracy`, `top3_accuracy`, `kl_model_lens` (nats, model
as first argument), and `mean_excess_kurtosis` (population fourth moment divided
by squared population variance, minus three, across V logits then across tokens).
These use 32 evenly spaced positions from each of 64 validation windows, 2,048
activations per depth. `sampling` lists every start and within-window position;
starts are relative to the validation split. J itself uses all 16,384 positions
and 2,105,344 causal pairs. Overlapping windows are retained as distinct contexts.

`lenses.prompt` contains the exact prompt, ids, `ids[rank,position,j,lens]`, and
`probabilities` with the same axes; lens=1 is J and lens=2 is logit. It also gives
`model_top1[position]` and the complete `model_final[a]` distribution.

## J-space

Each `j_space[j]` describes one depth. NNLS uses a Lawson–Hanson passive set;
`exact_decomposition` enumerates all subsets, with zero coefficients allowed.

Specimen fields:

- `selected[slot,k,n]`, chosen character ids, and `coefficients[slot,k,n]`, using
  the original unnormalised J vectors; slots after k are zero padding.
- `norm[n]` and `distance[k,n]`, respectively ‖x‖ and ‖x−x_J‖.
- `random_fraction[k,n]`, mean squared residual fraction over 20 fixed-seed
  independent random unit frames, each with V=7 directions.
- `span_fraction[n]` and `random_span_fraction[n]`, squared orthogonal projection
  fractions for the full J span and the mean of those 20 random spans. The latter
  are random seven-dimensional subspaces of R^8, not complete cones.
- `frame_rank`, `ambient_dimension`, `vocabulary_size`, explicitly recording V<D.

Shakespeare fields:

- `k = [0,...,65]`, `residual_fraction[k+1]`, and `random_residual_fraction[k+1]`,
  position means for the J frame and five fixed-seed random unit frames.
- `improvement[k]` and `random_improvement[k]`, the decreases from k−1 to k.
- `occupancy`, the first k whose J improvement is strictly below the random
  improvement, or null if none; no smoothing or interpolation is applied.
- `median_nonzero_at_25`, median count of coefficients >1e-10 after 25 pursuit
  steps, and `positions` (2,048).
- `span_fraction` and `random_span_fraction`, position-mean orthogonal projection
  fractions using a rank-revealing SVD; the random value also averages five draws.

NNOMP selects the largest positive residual correlation among unselected unit
directions and refits NNLS; if no such direction exists it carries forward the
last residual. `pursuit_curves` implements the same algorithm with cached Gram
matrices and feasible warm starts, tested against the direct `nnomp` function.
The five random frames are shared across depths to reduce control noise. Random
spans are full spans, not NNLS cones. Cone residuals need not equal span residuals,
even at k=V; nonnegativity can leave an additional in-span remainder.

## Interventions

Specimen `interventions` contains `alphas = [-4,...,4]`, lexicographic unordered
`pairs`, `concept_components = ["j_k3","remainder","lens_vector"]`, a shared
`probability_table[a,id]`, and `by_depth[j]`. Distribution ids allow overlapping
contexts and unchanged later positions to share probability columns. Columns are
interned at the same six significant digits as the JSON writer; no further
precision reduction is used. A zero id pads past the window boundary.

Each depth contains:

- `steer[n,a,alpha_index]`, probability-table ids at every edited position for
  x+alpha*v_a, using the raw lens vector.
- `steer_later_window1[offset,t,a,alpha_index]`, all later readouts for steering
  in window 1. Offset index 1 is the edited token. Later steering is subsampled
  to one window to fit the ~3 MiB size budget; all 80 edited-token readouts remain.
- `ablate[offset,n,a]`, all later readouts after projecting x perpendicular to
  the unit character direction.
- `swap[offset,n,pair]`, all later readouts for every pair and every column;
  `swap_coordinates[coordinate,n,pair]` and `swap_gram[row,column,pair]` expose
  the pre-swap least-squares coordinates and the 2x2 Gram matrix.
- `concept_vectors[coordinate,a,component]`, the unscaled exact k=3 cone
  component, remainder, and raw lens direction; `concept_norm[a] = ‖m_a‖`.
- `concept_mean_delta_logp[a,component]`, mean change in log p_a after adding the
  component rescaled to ‖m_a‖, restricted to positions whose target is not a.
  A direction of numerical zero is written as zero rather than divided by zero.

A swap uses c=V⁺x and x'=x+V(reverse(c)-c); `pinv` computes the SVD pseudoinverse
so nearly parallel character pairs remain defined. It agrees with
(V'V)^(-1)V' on independent pairs. Orthogonal components are preserved.

Shakespeare `interventions.swap` contains depth=2, selected pair, `alphas`,
`baseline[a]`, and `probabilities[a,alpha_index]` at the final prompt position.
The pair is chosen by exhaustive testing of all 2,080 unordered pairs at alpha=1,
preferring pairs that change final top-1, then maximising total variation.
`changed_top1_at_one`, `total_variation_at_one`, and `selection` record the choice.
The same replacement is made at every prompt position before continuing forward.

## Structure and depth similarity

`structure` records its seed and random counts. `mlp[b]` contains the block,
`random_median` (the unnormalised reference gain), and each population's
`q25`, `median`, `q75` of normalised gain. Populations are J vectors at the MLP's
input depth, logit vectors, 2,000 random unit directions, and columns of the
previous block's W2 (absent at block 1). The MLP branch includes its input LN and
all biases; it excludes the identity residual path.

`attention_ov` has one object per block/head with `random_gain` (mean over 2,000
unit directions), `random_label_baseline` (mean reciprocal rank over V independent
random labels), and population statistics `gain`, `label_preservation` (random
baseline subtracted), `raw_reciprocal_rank`. Populations are J vectors at the
attention input depth, one fixed random orthogonal rotation of the same vectors,
logit vectors, and V random label directions. Tied cosine ranks use average rank.

`cka[i,j]` is linear CKA of Wout J_i and Wout J_j, with each coordinate column
centred over character observations. The identity, orthogonal rotation and
translation/scale invariances are tested.

`analysis-report.md` in `jspace/briefs/` records numerical findings, actual
sizes/timings, test output, and limits. Logs in this directory retain the runs.

## Specimen training trajectory

`specimen-jlens-training.json` records all 201 states k=0,…,200, including the
initial state and the state after every update. It uses the stored Float64
`artifacts/specimen.json:theta0`, full training batch, AdamW hyperparameters,
matrix-only decay, clipping, and warmup/cosine schedule from
`scripts/export_specimen.jl`. Step 200 parameters and all losses are asserted
to agree with the source within 1e-12 before rounding.

- `steps[k+1] = k`; `depth` and `depth_kind` retain the static-export ordering.
- `vectors[a,q,j,k+1]`, shape [7,8,3,201], is Wout(θ_k) J_j(θ_k). Every J uses
  all ten windows and all 80 source positions under the new convention.
- `last_probs[a,j,k+1]`, shape [7,3,201], is the J-lens distribution at position
  8 of window 1, whose inputs are `to be or`.
- `loss[k+1]` is the full-batch cross-entropy after k updates.
- `model`, `conventions`, `vocab`, `sampling`, `provenance`, `sanity`, and
  `wall_seconds` record the source hash, recipe, hyperparameters, Julia/BLAS
  threads, axis meanings, and unrounded reproduction errors.

The file uses the same column-major tensor packaging, five significant digits,
and a strict size limit of 1,000,000 bytes. At every step the final-depth
vectors are the current Wout rows and the final-depth probabilities are the
current model readout. There is no subsampling of training time or depth.

# J-lens analysis implementation report

Current results and conventions are in the dated follow-up at the end of this
report. Earlier sections retain the superseded literal-pair analysis.

## Implementation

Implemented `jspace/src/JLens.jl`, the two exporters, tests, tensor-field
conventions in `jspace/artifacts/README.md`, and both requested JSON artifacts.
All changes are confined to `jspace/`; no commit was made. The module uses
PicoGPT's `embed`, `layernorm`, `affine`, `attention`, `mix`, and `relu` and their
hand-written pullbacks. There is no autodiff. Finite differences occur only in
the tests. Both scripts include `scripts/common.jl` rather than duplicating its
checkpoint loader, corpus split, tensor packaging or JSON writer.

Book-excerpt entry points include `attention_update`, `mlp_update`,
`residual_trace`, `continue_from`, `pull_to_depth`, `jacobian_block`,
`averaged_jacobians`, `lens_logits`, `lens_frame`, `nnls`,
`exact_decomposition`, `nnomp`, `pursuit_curves`, `span_fraction`,
`swap_coordinates`, `linear_cka`, `specimen_interventions`,
`structure_statistics`, and `prompt_swap`. Their docstrings state the maps or
optimisation problems. Passive-set NNLS is used both for exhaustive subset
search and for greedy nonnegative pursuit. The fast pursuit caches A'A and A'X
and starts each NNLS refit from the preceding feasible coefficients.

## Decisions and limits

1. The literal final-depth J-lens equality at tolerance 1e-9 is false with the
   brief's fixed LayerNorm epsilon. J_L=sI with s=2/(T+1), but
   LN_epsilon(sx)=LN_(epsilon/s²)(x), before the learned affine map. The exports
   retain the literal requested lens, without rescaling J or changing epsilon.
   Tests pass the identity at 1e-9 after scaling epsilon by s², and mark the
   requested unadjusted identity explicitly broken. This is a conflict in §8,
   not an unexplained numerical failure. Exact discrepancies are listed below.
2. Specimen K blocks use source batch window 1, whose inputs decode to
   `to be or`. The 10 per-window attention-depth Jacobians use all 10 windows.
3. The first size-budget trial included all later steering readouts and measured
   4,242,647 bytes. The final file retains all 80 edited-position steering
   results, every alpha and character, every later ablation and swap result,
   and later steering for window 1. This is the only specimen sampling
   reduction. Full K blocks and all exact cone fits remain present.
4. Specimen intervention probability columns share a six-significant-digit
   lookup table. The table uses exactly the requested writer precision; ids
   reconstruct the distributions. See the README for axes and padding.
5. Shakespeare J averages all source positions and all causal targets from 64
   evenly spaced validation windows of length 256: 16,384 positions and
   2,105,344 pairs. No early position or outlier is filtered. Jself/Jfuture
   are omitted (`null`) to reserve runtime for the required computations.
6. Shakespeare lens metrics and NNOMP use 32 evenly spaced positions per window,
   including positions 1 and 256: 2,048 activations per depth. The five random
   frames are shared across depths. The sample ids, seeds and corpus/checkpoint
   hashes are in the JSON. No training text is used for these samples.
7. The Shakespeare swap exhaustively tests all 2,080 unordered pairs at alpha=1
   and prefers changed top-1, then greatest total variation. The chosen pair is
   then swept at alpha=[0,0.25,0.5,1,2], with every prompt position edited at depth 2.
8. Span-energy fractions are orthogonal least-squares projections, not cone
   fits. V<D guarantees a possible orthogonal remainder; nonnegativity can add
   an additional in-span remainder, so a full-k cone residual is not in general
   the orthogonal-complement energy alone.
9. All arithmetic is Float64; the source Shakespeare checkpoint is Float32.
   Julia 1.13.3, 12 Julia threads, BLAS=1. The metadata timings start
   on entry to the exporter, include later compilation, and exclude startup,
   compilation of the export function itself, and serialisation. Rounded JSON is not a
   suitable input for numerical identities at 1e-9.

## Verification and results

### Export sizes and runtime

| Model | Bytes | Computation seconds | Checkpoint step |
|---|---|---|---|
| specimen | 2816783 | 187.158 | 200 |
| shakespeare | 3439732 | 751.05 | 5000 |

Paths: `$.wall_seconds`, `$.model.step`; file sizes measured on disk.

Shakespeare stage timings (`$.timings`): jacobian_seconds=537.39, space_seconds=146.57, swap_seconds=58.256.

The Shakespeare process took approximately 803 seconds (13.38 minutes), measured from export-log creation to its final write, including startup and output. This is below the 25-minute budget.

### Specimen

Parameter count: 1007 (`$.model.parameter_count`). All 80 activation columns and 360 causal pairs contribute.

Final-depth checks (`$.sanity`): final_j_error=5.55112e-17, final_self_error=0, final_future_norm=0, final_literal_lens_error=1.72324e-05.

| Depth | J/logit top-1 agreement | Mean full J span energy | Mean random span energy | Mean k=7 J cone residual | Mean k=7 random cone residual |
|---|---|---|---|---|---|
| 0 | 0.4 | 0.8894853 | 0.8776025 | 0.5056189 | 0.5176389 |
| 0.5 | 0.45 | 0.81298 | 0.8713875 | 0.5081652 | 0.5529944 |
| 1 | 1 | 0.9905193 | 0.8599236 | 0.06381523 | 0.5424455 |

Paths: `$.lenses.j_logit_agreement[j]`, means of `$.j_space[j].span_fraction`, `random_span_fraction`, `distance[k,n]^2/norm[n]^2`, and `random_fraction[k,n]`; k=7.

Full mean cone-residual curves, k=1,...,7 (derived from those same arrays):

- Depth 0: J=[0.8286006, 0.7370965, 0.6038478, 0.5151329, 0.5077318, 0.5056189, 0.5056189]; random=[0.7248036, 0.5989677, 0.5453568, 0.523997, 0.5184802, 0.5176393, 0.5176389].
- Depth 0.5: J=[0.803077, 0.6904913, 0.6036265, 0.5460555, 0.5226612, 0.5083081, 0.5081652]; random=[0.7412354, 0.6293186, 0.5798449, 0.5585399, 0.5535904, 0.5530109, 0.5529944].
- Depth 1: J=[0.1779043, 0.1159769, 0.09327913, 0.07625979, 0.06889109, 0.06434645, 0.06381523]; random=[0.7392552, 0.6205105, 0.5727649, 0.5498253, 0.5436437, 0.5425974, 0.5424455].

Concept interventions: mean across characters of the per-character mean delta log p, ordered [exact k=3 J component, remainder, raw lens direction], always rescaled to ||m_a||. Path: `$.interventions.by_depth[j].concept_mean_delta_logp[a,component]`.

- Depth 0: [5.103279, 2.017762, 6.828241].
- Depth 0.5: [8.073726, 2.059628, 8.514746].
- Depth 1: [9.435144, 2.612527, 9.490324].

The effects are changes in nats at non-target positions, often starting from very small p_a; they are not accuracy improvements. Full per-character values are in the cited tensor.

MLP normalised median gains (`$.structure.mlp[1].populations`): logit_lens=0.727064, random=1, j_lens=1.50391.

| Head | J gain | Rotated J gain | Logit gain | J label preservation | Rotated J label preservation |
|---|---|---|---|---|---|
| 1 | 1.34852 | 0.975606 | 0.899214 | 0.272109 | 0.256803 |
| 2 | 0.871444 | 1.4098 | 0.994757 | -0.047619 | -0.0833333 |

Paths: `$.structure.attention_ov[h].populations.{j_lens,rotated_j_lens,logit_lens}.{gain,label_preservation}`.

CKA (`$.cka`, rows/columns depths 0,0.5,1):

```text
1 0.909709 0.82894
0.909709 1 0.762601
0.82894 0.762601 1
```

### Shakespeare

The literal J-lens has lower next-character top-1 accuracy than the logit lens at every intermediate depth in this checkpoint. The J cone beats the random cone in mean residual only at depths 3.5 and 4. These are measured results, not evidence for an unconditional advantage of J directions. The final-depth maximum probability discrepancy 0.18907 is material; prose must not describe the two final readouts as equal at fixed epsilon.

Parameter count 810049; step 5000; validation tokens 111539; 64 windows, 16384 Jacobian source positions and 2105344 pairs, 2048 analysed activations per depth. Paths: `$.model`, `$.sampling`.

Final-depth checks (`$.sanity`): final_j_error=1.3878e-17, final_literal_lens_error=0.18907.

| Depth | J model agreement | Logit model agreement | J top-1 accuracy | Logit top-1 accuracy | J top-3 accuracy | Logit top-3 accuracy | J KL | Logit KL | J excess kurtosis | Logit excess kurtosis |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.094727 | 0.16699 | 0.089844 | 0.12158 | 0.21533 | 0.25 | 2.7139 | 2.7077 | -0.5975 | -0.4252 |
| 0.5 | 0.080078 | 0.14941 | 0.079102 | 0.11035 | 0.22363 | 0.23926 | 2.6076 | 2.5692 | -0.52662 | -0.4912 |
| 1 | 0.088379 | 0.15625 | 0.086426 | 0.1167 | 0.2334 | 0.24561 | 2.7146 | 2.4847 | -0.33915 | -0.19029 |
| 1.5 | 0.09375 | 0.15625 | 0.096191 | 0.13525 | 0.22021 | 0.30908 | 2.5837 | 2.2034 | -0.36544 | -0.21066 |
| 2 | 0.11133 | 0.18848 | 0.11572 | 0.16504 | 0.23779 | 0.33887 | 2.6303 | 2.0633 | -0.34076 | -0.15531 |
| 2.5 | 0.15625 | 0.22119 | 0.15234 | 0.19141 | 0.30273 | 0.38184 | 2.3882 | 2.0247 | -0.24932 | -0.10164 |
| 3 | 0.27832 | 0.40723 | 0.24756 | 0.32324 | 0.44189 | 0.56055 | 1.552 | 1.1353 | -0.30334 | -0.01249 |
| 3.5 | 0.4043 | 0.4624 | 0.3252 | 0.36719 | 0.54492 | 0.59814 | 1.1101 | 0.89961 | -0.12366 | -0.16114 |
| 4 | 0.99414 | 1 | 0.54785 | 0.54785 | 0.75244 | 0.75049 | 0.0054144 | 0 | 0.087693 | 0.10963 |

Paths: `$.lenses.metrics[j].{j_lens,logit_lens}`; accuracies and agreements are fractions, KL is nats.

| Depth | Occupancy | Median nonzero at k=25 | J residual k=25 | Random residual k=25 | J residual k=65 | Random residual k=65 | J span energy | Random span energy |
|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 25 | 0.82482 | 0.76071 | 0.81849 | 0.75773 | 0.40241 | 0.50425 |
| 0.5 | 1 | 25 | 0.87557 | 0.74356 | 0.87063 | 0.73983 | 0.28722 | 0.51892 |
| 1 | 1 | 25 | 0.84612 | 0.72869 | 0.8425 | 0.72433 | 0.3011 | 0.51516 |
| 1.5 | 1 | 25 | 0.83292 | 0.73264 | 0.82901 | 0.72828 | 0.35886 | 0.50574 |
| 2 | 1 | 25 | 0.82895 | 0.73347 | 0.82542 | 0.72932 | 0.34399 | 0.50694 |
| 2.5 | 2 | 25 | 0.77537 | 0.73514 | 0.77026 | 0.73089 | 0.4432 | 0.51078 |
| 3 | 3 | 25 | 0.75683 | 0.74383 | 0.75221 | 0.73992 | 0.45971 | 0.50622 |
| 3.5 | 5 | 25 | 0.69965 | 0.7494 | 0.69222 | 0.746 | 0.52087 | 0.50589 |
| 4 | 37 | 25 | 0.40515 | 0.74764 | 0.39843 | 0.7441 | 0.78395 | 0.50526 |

Paths: `$.j_space[j].{occupancy,median_nonzero_at_25,residual_fraction,random_residual_fraction,span_fraction,random_span_fraction}`. The curves include k=0, so k=25 is entry 26 with Julia indexing.

Swap pair ids [21, 57], characters ['H', 'r']. At alpha=1 final top-1 changes from '\n' (id 1) to 'a' (id 40); changed flag=True, total variation=0.99596. Paths: `$.interventions.swap.{pair,baseline,probabilities,changed_top1_at_one,total_variation_at_one}`.

| Block | J MLP median | Logit MLP median | Previous neurons median |
|---|---|---|---|
| 1 | 1.0688 | 1.0306 | null |
| 2 | 0.90866 | 1.088 | 1.0538 |
| 3 | 0.90398 | 1.1635 | 1.0224 |
| 4 | 1.3376 | 1.4771 | 1.1227 |

Path: `$.structure.mlp[b].populations`. Quartiles are also exported.

| Block | Head | J OV gain | Rotated J gain | Logit gain | J label preservation | Rotated J label preservation |
|---|---|---|---|---|---|---|
| 1 | 1 | 1.2594 | 1.1726 | 0.99807 | 0.31168 | -0.049744 |
| 1 | 2 | 1.5543 | 0.99288 | 1.0032 | 0.31501 | -0.016147 |
| 1 | 3 | 1.3815 | 1.0444 | 0.938 | 0.3096 | -0.011142 |
| 1 | 4 | 1.4444 | 0.90301 | 1.07 | 0.26608 | 0.0037311 |
| 2 | 1 | 1.4208 | 0.97478 | 1.1445 | 0.10667 | -0.0022477 |
| 2 | 2 | 1.8261 | 1.0151 | 1.222 | 0.090836 | -0.0023292 |
| 2 | 3 | 1.2847 | 1.0196 | 1.1178 | 0.13089 | 0.013248 |
| 2 | 4 | 1.2822 | 1.0362 | 1.0859 | 0.12124 | 0.00081669 |
| 3 | 1 | 1.2417 | 1.0715 | 1.2341 | 0.071668 | 0.0008479 |
| 3 | 2 | 1.1462 | 1.0461 | 1.0588 | 0.14803 | -0.00087612 |
| 3 | 3 | 1.1644 | 1.0479 | 1.0776 | 0.12249 | 0.018421 |
| 3 | 4 | 1.1828 | 0.99964 | 1.0971 | 0.1252 | 0.00024082 |
| 4 | 1 | 1.0513 | 1.0445 | 1.1965 | 0.10099 | 0.0049793 |
| 4 | 2 | 0.97596 | 0.99339 | 1.1074 | 0.097464 | -0.00054624 |
| 4 | 3 | 1.275 | 0.9844 | 1.1617 | 0.1492 | -0.0018153 |
| 4 | 4 | 1.1065 | 1.0534 | 1.2961 | 0.046976 | -0.0008069 |

Paths: `$.structure.attention_ov[index].{block,head,populations}`. Gains divide by the random mean; label preservation subtracts the random baseline.

| Depth | Min per-window J Frobenius norm | Median | Max |
|---|---|---|---|
| 0 | 0.063544 | 0.0651685 | 0.067299 |
| 0.5 | 0.060033 | 0.061773 | 0.063945 |
| 1 | 0.059952 | 0.0615945 | 0.063489 |
| 1.5 | 0.058957 | 0.0602565 | 0.061776 |
| 2 | 0.065705 | 0.066968 | 0.068476 |
| 2.5 | 0.066491 | 0.067779 | 0.069258 |
| 3 | 0.077843 | 0.078496 | 0.079352 |
| 3.5 | 0.079874 | 0.08036 | 0.080991 |
| 4 | 0.088044 | 0.088044 | 0.088044 |

Path: `$.jacobians.per_window_frobenius[b,j]`. All windows, including the largest norms, remain in the average.

J/logit top-1 agreement by depth (`$.lenses.j_logit_agreement`): [0.104, 0.12842, 0.10107, 0.15088, 0.16113, 0.20312, 0.29736, 0.43604, 0.99414].

CKA of embed vs final (`$.cka[1,9]`): 0.77205.

### Test output

Command: `julia --project -t auto jspace/test/runtests.jl`. Exit status 0: 381 passing assertions, one explicitly broken assertion. Finite-difference maximum relative error is 3.4559228746713074e-10; trained-frame fast/direct NNOMP maximum discrepancy is 8.881784197001252e-16. The explicitly broken assertion is the fixed-epsilon contradiction described above.

```text
Maximum finite-difference relative block error: 3.4559228746713074e-10
Test Summary:                                 | Pass  Total   Time
Residual semantics and hand-written Jacobians |  101    101  14.2s
Literal final lens epsilon discrepancy: 2.3232230184078162e-7
Test Summary:                          | Pass  Broken  Total  Time
Final-depth lens and LayerNorm epsilon |    2       1      3  3.2s
Test Summary:                          | Pass  Total  Time
NNLS versus exhaustive active supports |  125    125  5.3s
Test Summary:                         | Pass  Total  Time
Gram pursuit agrees with direct NNOMP |   40     40  4.4s
Test Summary:          | Pass  Total  Time
Two-block continuation |    6      6  1.2s
Test Summary:                 | Pass  Total  Time
Pseudoinverse coordinate swap |    4      4  1.1s
Test Summary: | Pass  Total  Time
Linear CKA    |    3      3  1.0s
Maximum trained-frame Gram/direct NNOMP discrepancy: 8.881784197001252e-16
Test Summary:                                               | Pass  Total  Time
Export schema and numeric invariants (when artifacts exist) |  100    100  8.3s
```

## Unfinished work

No required analysis family is absent. The explicit limits are Shakespeare Jself/Jfuture (permitted omission), specimen later steering beyond window 1 (size-budget reduction), and the literal final-depth 1e-9 equality (mathematically incompatible with fixed epsilon). No prose or figures are implemented by this task.

## 2026-11-06 — scale-free J and the specimen training trajectory

This section supersedes the earlier convention, final-depth caveat, numerical tables, and test summary. The earlier section is retained as the record of the literal-pair export.

The exported Jacobian is now J = (BT)⁻¹ Σ_(b,t) Σ_(t′≥t) K_(t′t), the mean over windows and sources of the sum over causal targets. It is (T+1)/2 times the former uniform-pair average: 4.5 for the specimen and 128.5 for Shakespeare. Thus J_L=I, the final covectors are exactly the rows of Wout, and the final J-lens equals the model readout with the original fixed LayerNorm epsilon. Only the clearly labelled `lenses.literal` readouts retain the paper-weighted convention; their nonzero final errors demonstrate the epsilon lesson. No epsilon adjustment is used by the primary lens.

`self` is unchanged. To make `averaged = self + future` exact, `future` now uses the same BT source denominator; its old strict-pair denominator was incompatible with the requested additive split. The exported specimen `blocks` are multiplied by 4.5 as explicitly requested; divide by `conventions.jacobian_factor` to recover raw K. The `jacobian_block` function and its finite-difference tests still return/check the raw derivative. Per-window matrices and singular values use the new scale. All existing field names and tensor axes remain available.

All readouts, prompt grids, exact/greedy decompositions, interventions and structure statistics were recomputed, not patched by rescaling stored results. Cone geometry, span energy, unit-direction MLP/OV statistics and CKA are invariant to this positive global scale, apart from numerical rounding. Exact-fit coefficients and swap coordinates have inverse scale; swap Gram matrices have squared scale. Raw steering uses the same alpha grid with the larger vectors. Ablations, swaps and equal-norm concept writes retain their scale-invariant effects.

### Export sizes, timings and sanity

| Export | Bytes | Computation seconds | Process seconds | Final step |
|---|---|---|---|---|
| specimen | 2834032 | 177.837 | 209.19 | 200 |
| shakespeare | 3083825 | 653.79 | 694.37 | 5000 |
| specimen training | 322404 | 69.803 | 84.54 | 200 |

Computation timings are `$.wall_seconds`; process timings include startup and serialisation and are retained in the corresponding `*-export-time.txt` files. Runs overlapped on this machine. Shakespeare stage timings (`$.timings`): jacobian_seconds=507.27, space_seconds=88.172, swap_seconds=51.196.

| Model | J_L−I max error | Self error | Future norm | J−self−future max error | Final lens max probability error | Literal paper-weighted final error |
|---|---|---|---|---|---|---|
| specimen | 0 | 0 | 0 | 3.60822e-16 | 0 | 1.72324e-05 |
| shakespeare | 0 | null | null | null | 0 | 0.18907 |

Paths: `$.sanity`, `$.lenses.final_max_probability_error`, `$.lenses.literal.final_max_probability_error`. Shakespeare self/future remain omitted to preserve the runtime budget. Final identities were asserted before JSON rounding. `sanity.final_literal_lens_error` is retained as a compatibility alias of the labelled literal error.

### Specimen: refreshed tables and invariant quantities

| Depth | J/logit top-1 agreement | Mean J span energy | Mean random span energy | Mean k=7 J cone residual | Mean k=7 random cone residual |
|---|---|---|---|---|---|
| 0 | 0.4 | 0.88948525 | 0.87760255 | 0.50561892 | 0.51763891 |
| 0.5 | 0.45 | 0.81298 | 0.8713875 | 0.50816517 | 0.55299444 |
| 1 | 1 | 0.99051926 | 0.85992362 | 0.063815233 | 0.54244546 |

Paths: `$.lenses.j_logit_agreement[j]` and means of `$.j_space[j].{span_fraction,random_span_fraction,distance[k,n]^2/norm[n]^2,random_fraction[k,n]}`. Residuals and span fractions are unchanged to the original report precision; small differences below that precision arise from rounding the regenerated distances.

Recomputed mean cone-residual curves for k=1,…,7:

- Depth 0: J=[0.82860062, 0.73709654, 0.60384776, 0.51513292, 0.50773182, 0.50561892, 0.50561892]; random=[0.72480356, 0.59896769, 0.54535683, 0.52399695, 0.51848023, 0.51763927, 0.51763891].
- Depth 0.5: J=[0.80307698, 0.69049129, 0.60362648, 0.54605547, 0.52266122, 0.50830815, 0.50816517]; random=[0.74123539, 0.62931857, 0.57984494, 0.55853988, 0.55359044, 0.55301091, 0.55299444].
- Depth 1.0: J=[0.17790427, 0.11597688, 0.093279132, 0.07625979, 0.06889109, 0.064346451, 0.063815233]; random=[0.73925518, 0.62051051, 0.5727649, 0.54982533, 0.54364367, 0.5425974, 0.54244546].

Equal-norm concept writes, means across characters of per-character mean delta log p (`$.interventions.by_depth[j].concept_mean_delta_logp`), in nats at non-target positions:

| Depth | Exact k=3 J component | Remainder | Lens vector |
|---|---|---|---|
| 0 | 5.103279 | 2.0177617 | 6.8282414 |
| 0.5 | 8.0737257 | 2.059628 | 8.5147457 |
| 1 | 9.4351443 | 2.6125271 | 9.4903243 |

Specimen MLP median gains remain logit_lens=0.727064, random=1, j_lens=1.50391.

| Head | J gain | Rotated J gain | Logit gain | J label preservation | Rotated J label preservation |
|---|---|---|---|---|---|
| 1 | 1.34852 | 0.975606 | 0.899214 | 0.272109 | 0.256803 |
| 2 | 0.871444 | 1.4098 | 0.994757 | -0.047619 | -0.0833333 |

The MLP and OV tables are unchanged. Paths: `$.structure.mlp[1].populations` and `$.structure.attention_ov[h].populations`. Specimen CKA (`$.cka`) is also unchanged:

```text
1 0.909709 0.82894
0.909709 1 0.762601
0.82894 0.762601 1
```

### Shakespeare: primary lens metrics

Under the new scale the J-lens still trails the logit lens at every intermediate depth in next-character top-1 accuracy. The measured gaps range from 3.418 to 7.666 percentage points; the final-depth metrics coincide. The new table replaces the earlier literal-pair table. Literal metrics remain separately available at `$.lenses.literal.metrics[j]` with the same fields.

| Depth | J model agreement | Logit model agreement | J top-1 accuracy | Logit top-1 accuracy | J top-3 accuracy | Logit top-3 accuracy | J KL | Logit KL | J excess kurtosis | Logit excess kurtosis |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.088379 | 0.16699 | 0.083496 | 0.12158 | 0.19922 | 0.25 | 3.0662 | 2.7077 | -0.56498 | -0.4252 |
| 0.5 | 0.074219 | 0.14941 | 0.072266 | 0.11035 | 0.21533 | 0.23926 | 2.9213 | 2.5692 | -0.49007 | -0.4912 |
| 1 | 0.085938 | 0.15625 | 0.08252 | 0.1167 | 0.21875 | 0.24561 | 3.0823 | 2.4847 | -0.30861 | -0.19029 |
| 1.5 | 0.087891 | 0.15625 | 0.088867 | 0.13525 | 0.20605 | 0.30908 | 2.9782 | 2.2034 | -0.34824 | -0.21066 |
| 2 | 0.10107 | 0.18848 | 0.10889 | 0.16504 | 0.23291 | 0.33887 | 2.9923 | 2.0633 | -0.33009 | -0.15531 |
| 2.5 | 0.15088 | 0.22119 | 0.15039 | 0.19141 | 0.29346 | 0.38184 | 2.6932 | 2.0247 | -0.24202 | -0.10164 |
| 3 | 0.27344 | 0.40723 | 0.24658 | 0.32324 | 0.43604 | 0.56055 | 1.7017 | 1.1353 | -0.28206 | -0.01249 |
| 3.5 | 0.39746 | 0.4624 | 0.31934 | 0.36719 | 0.53613 | 0.59814 | 1.2105 | 0.89961 | -0.11336 | -0.16114 |
| 4 | 1 | 1 | 0.54785 | 0.54785 | 0.75049 | 0.75049 | 0 | 0 | 0.10963 | 0.10963 |

| Depth | Logit minus J top-1 accuracy (percentage points) |
|---|---|
| 0 | 3.8084 |
| 0.5 | 3.8084 |
| 1 | 3.418 |
| 1.5 | 4.6383 |
| 2 | 5.615 |
| 2.5 | 4.102 |
| 3 | 7.666 |
| 3.5 | 4.785 |

Paths: `$.lenses.metrics[j].{j_lens,logit_lens}`; accuracies/agreements are fractions, KL is model-to-lens in nats. Counts and validation sampling are unchanged: 64 windows, 16,384 Jacobian source positions, 2,105,344 causal pairs, and 2,048 activation/metric positions per depth.

### Shakespeare: recomputed J-space and structure

| Depth | Occupancy | Median nonzero at k=25 | J residual k=25 | Random residual k=25 | J residual k=65 | Random residual k=65 | J span energy | Random span energy |
|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 25 | 0.82482 | 0.76071 | 0.81849 | 0.75773 | 0.40241 | 0.50425 |
| 0.5 | 1 | 25 | 0.87557 | 0.74356 | 0.87063 | 0.73983 | 0.28722 | 0.51892 |
| 1 | 1 | 25 | 0.84612 | 0.72869 | 0.8425 | 0.72433 | 0.3011 | 0.51516 |
| 1.5 | 1 | 25 | 0.83292 | 0.73264 | 0.82901 | 0.72828 | 0.35886 | 0.50574 |
| 2 | 1 | 25 | 0.82895 | 0.73347 | 0.82542 | 0.72932 | 0.34399 | 0.50694 |
| 2.5 | 2 | 25 | 0.77537 | 0.73514 | 0.77026 | 0.73089 | 0.4432 | 0.51078 |
| 3 | 3 | 25 | 0.75683 | 0.74383 | 0.75221 | 0.73992 | 0.45971 | 0.50622 |
| 3.5 | 5 | 25 | 0.69965 | 0.7494 | 0.69222 | 0.746 | 0.52087 | 0.50589 |
| 4 | 37 | 25 | 0.40515 | 0.74764 | 0.39843 | 0.7441 | 0.78395 | 0.50526 |

The J-space table is unchanged at exported precision; J cones beat the random control in mean residual at depths 3.5 and 4. Paths: `$.j_space[j]`, curves include k=0.

The recomputed depth-2 swap chooses ids [21, 57], characters ['H', 'r']; at alpha=1 it changes the final top-1 from '\n' (id 1) to 'a' (id 40), total variation 0.99596. The pair and reported effects are unchanged. Path: `$.interventions.swap`.

| Block | J MLP median | Logit MLP median | Previous neurons median |
|---|---|---|---|
| 1 | 1.0688 | 1.0306 | null |
| 2 | 0.90866 | 1.088 | 1.0538 |
| 3 | 0.90398 | 1.1635 | 1.0224 |
| 4 | 1.3376 | 1.4771 | 1.1227 |

| Block | Head | J OV gain | Rotated J gain | Logit gain | J label preservation | Rotated J label preservation |
|---|---|---|---|---|---|---|
| 1 | 1 | 1.2594 | 1.1726 | 0.99807 | 0.31168 | -0.049744 |
| 1 | 2 | 1.5543 | 0.99288 | 1.0032 | 0.31501 | -0.016147 |
| 1 | 3 | 1.3815 | 1.0444 | 0.938 | 0.3096 | -0.011142 |
| 1 | 4 | 1.4444 | 0.90301 | 1.07 | 0.26608 | 0.0037311 |
| 2 | 1 | 1.4208 | 0.97478 | 1.1445 | 0.10667 | -0.0022477 |
| 2 | 2 | 1.8261 | 1.0151 | 1.222 | 0.090836 | -0.0023292 |
| 2 | 3 | 1.2847 | 1.0196 | 1.1178 | 0.13089 | 0.013248 |
| 2 | 4 | 1.2822 | 1.0362 | 1.0859 | 0.12124 | 0.00081669 |
| 3 | 1 | 1.2417 | 1.0715 | 1.2341 | 0.071668 | 0.0008479 |
| 3 | 2 | 1.1462 | 1.0461 | 1.0588 | 0.14803 | -0.00087612 |
| 3 | 3 | 1.1644 | 1.0479 | 1.0776 | 0.12249 | 0.018421 |
| 3 | 4 | 1.1828 | 0.99964 | 1.0971 | 0.1252 | 0.00024082 |
| 4 | 1 | 1.0513 | 1.0445 | 1.1965 | 0.10099 | 0.0049793 |
| 4 | 2 | 0.97596 | 0.99339 | 1.1074 | 0.097464 | -0.00054624 |
| 4 | 3 | 1.275 | 0.9844 | 1.1617 | 0.1492 | -0.0018153 |
| 4 | 4 | 1.1065 | 1.0534 | 1.2961 | 0.046976 | -0.0008069 |

MLP and OV tables are unchanged. Paths: `$.structure.mlp[b].populations`, `$.structure.attention_ov[index].populations`.

| Depth | Min per-window J Frobenius norm | Median | Max |
|---|---|---|---|
| 0 | 8.1654 | 8.37415 | 8.6479 |
| 0.5 | 7.7142 | 7.9378 | 8.2169 |
| 1 | 7.7039 | 7.9149 | 8.1584 |
| 1.5 | 7.5759 | 7.74295 | 7.9382 |
| 2 | 8.4431 | 8.6054 | 8.7992 |
| 2.5 | 8.5441 | 8.7096 | 8.8996 |
| 3 | 10.003 | 10.087 | 10.197 |
| 3.5 | 10.264 | 10.326 | 10.407 |
| 4 | 11.314 | 11.314 | 11.314 |

The Frobenius norms replace the earlier table and have the 128.5 scale factor. Path: `$.jacobians.per_window_frobenius[b,j]`. No outlier was filtered.

New J/logit top-1 agreements (`$.lenses.j_logit_agreement`): [0.094238, 0.12695, 0.095703, 0.14746, 0.15479, 0.20117, 0.29395, 0.4292, 1].

Embed/final CKA remains 0.77205 (`$.cka[1,9]`). The prompt grid was recomputed under the new convention at `$.lenses.prompt.{ids,probabilities}`; the prompt and tensor axes are unchanged.

### Training-time export

`jspace/artifacts/specimen-jlens-training.json` contains every k=0,…,200. `vectors[a,q,j,k+1]` has shape [7,8,3,201], `last_probs[a,j,k+1]` has shape [7,3,201], and `loss[k+1]` has 201 entries. Last probabilities read position 8 of window 1. J averages all ten training windows and all 80 source positions at every update. Depth, vocabulary, explicit axis conventions, source SHA-256, hyperparameters, recipe path, thread counts and timing are exported. The writer uses five significant digits; the file is below 1 MB.

| Training check (before rounding) | Maximum error |
|---|---|
| theta_final_max_error | 0 |
| loss_trajectory_max_error | 6.6613e-16 |
| final_depth_max_probability_error | 3.3307e-16 |

The step-200 parameters reproduce `artifacts/specimen.json:train.theta_final` exactly. The complete loss trajectory agrees to roundoff. Paths: `$.sanity`, `$.loss`, `$.provenance`. At all snapshots, J_L=I and the final vectors are the current readout rows.

### Verification

Command: `julia --project -t auto jspace/test/runtests.jl`. Exit status 0; 433 passing assertions; process time 71.35 seconds (`jspace/artifacts/test-time.txt`). There are no broken assertions. Tests cover the explicit source-normalised sum, self/future additivity (one window and full batch), exact final identity and vectors, the retained literal epsilon discrepancy, finite differences and causality, NNLS/NNOMP, swaps and CKA, static schemas, and training endpoints/trajectory.

```text
Maximum finite-difference relative block error: 3.4559228746713074e-10
Test Summary:                                 | Pass  Total   Time
Residual semantics and hand-written Jacobians |  107    107  16.3s
Scale-free final lens probability discrepancy: 0.0
Literal paper-weighted final lens epsilon discrepancy (80 columns): 1.7232390429833266e-5
Test Summary:                          | Pass  Total  Time
Final-depth lens and LayerNorm epsilon |    5      5  5.8s
Test Summary:                          | Pass  Total  Time
NNLS versus exhaustive active supports |  125    125  8.8s
Test Summary:                         | Pass  Total  Time
Gram pursuit agrees with direct NNOMP |   40     40  4.5s
Test Summary:          | Pass  Total  Time
Two-block continuation |    6      6  0.7s
Test Summary:                 | Pass  Total  Time
Pseudoinverse coordinate swap |    4      4  0.9s
Test Summary: | Pass  Total  Time
Linear CKA    |    3      3  1.5s
Maximum trained-frame Gram/direct NNOMP discrepancy: 1.1102230246251565e-15
Test Summary:                                               | Pass  Total  Time
Export schema and numeric invariants (when artifacts exist) |  118    118  7.2s
Test Summary:                                     | Pass  Total  Time
Specimen training export endpoints and trajectory |   25     25  1.5s
```

The specimen’s embedded `sanity.pullback_tests` was refreshed from the final `verification.json` after the full suite; the file-size table reflects this metadata update.

Schema comparison against the pre-follow-up files and numeric comparison notes:

```text
specimen: every existing key, list length and tensor shape preserved.
specimen: Jacobians, blocks/per-window matrices, singular values and vectors have the requested 4.5 scale (within export precision).
specimen: structure statistics and CKA are unchanged at exported precision.
specimen: self unchanged; future uses BT denominator (3.5 times the former strict-pair mean).
specimen: labelled literal probabilities exactly reproduce the previous export.
specimen: primary J/logit top-1 agreement is unchanged.
specimen: cone distances, activation norms, span fractions and random controls unchanged at exported precision.
specimen: reconstructed ablate probabilities differ by at most 0.
specimen: reconstructed swap probabilities differ by at most 0.
shakespeare: every existing key, list length and tensor shape preserved.
shakespeare: Jacobians, blocks/per-window matrices, singular values and vectors have the requested 128.5 scale (within export precision).
shakespeare: structure statistics and CKA are unchanged at exported precision.
shakespeare: labelled literal metrics exactly reproduce the previous export.
shakespeare: J-space table/curves and prompt swap are unchanged at exported precision.
```

No commits were made; implementation changes are confined to `jspace/`. The earlier permitted omissions (Shakespeare self/future and later specimen steering outside window 1) remain. The former final-depth equality limitation is resolved.

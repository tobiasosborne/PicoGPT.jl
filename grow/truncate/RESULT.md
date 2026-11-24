# Coarse-graining back: truncating the residual stream

Idea 3 of `grow/README.md`. Given a trained model of stream width D, find a model of width r < D
that it is the finer version of, by inserting a rank-r projector on the stream and choosing the
projector by a rule. Julia 1.13, the repository's own `src/` (used, not changed), 3 threads.
Work stopped early at the owner's deadline; what was not done is listed at the end.

## What was built

| File | What it is |
|---|---|
| `truncate.jl` | the cut model with a hand-written pullback, the statistics C and M, the rules, retraining, the cut model written in r coordinates, the intertwiner defect |
| `test.jl` | 136 checks, all passing (about 1.5 min alone) |
| `specimen.jl` → `specimen.log`, `specimen.tsv`, `spectra.tsv`, `defects.tsv` | steps 1–3 on D = 4, 8, 16 (about 4 min on a loaded machine) |
| `shakespeare.jl` → `shakespeare.log`, `shakespeare.tsv` | step 4, on the 810,049-parameter checkpoint (6.6 min on a loaded machine) |

**The cut.** The stream is visited at K = 2L + 1 sites (after the embedding, after each attention
write, after each MLP write). At every site k the stream is replaced by

    x ↦ T S (x − μₖ) + μₖ,      T ∈ ℝ^{D×r}, S ∈ ℝ^{r×D}, S T = 1,

one projector for all sites, μₖ the mean of the stream at site k over the data. The discarded
directions are frozen at their mean, not at zero. Everything else in the model is unchanged.

**What happens to LayerNorm (the choice made).** LayerNorm keeps acting on the lifted stream
T z + μₖ in ℝᴰ, where z = S (x − μₖ) ∈ ℝʳ is the small model's stream. In z coordinates a reader
W LN(x) + b is then exactly

    (W Γ Π T) z/σ(z) + (W Γ Π μ)/σ(z) + (W β + b),     σ(z)² = (zᵀG z + 2hᵀz + c₀)/D + ε,
    G = TᵀΠT,  h = TᵀΠμ,  c₀ = μᵀΠμ,  Π = 1 − 11ᵀ/D.

So the cut model has r-column weights, but its normalisation is by a general quadratic form in z
with a constant term, not by the round sphere of a width-r LayerNorm, and its position code is the
fixed table S (P − μ₁), not sinusoids. It is a width-r stream with the parent's head width D/H and
MLP width F. **It is not a member of the PicoGPT family**: coarse-graining the width generates a
coupling outside the family, as the README's table says blocking does. `smallgpt` computes the
model entirely in r coordinates from these formulas and agrees with the projector form to 1e-10.
(Unchecked remark: when h = 0 and c₀ = 0 a change of basis by G^{1/2} turns σ into a standard
LayerNorm of width r + 1, so the obstruction is the frozen mean, which costs a constant coordinate.)

**The rules.** C = covariance of the stream, M = E Σ JᵀG J, the Gram of the Jacobian J of the
logits with respect to the stream, both pooled over sites. G is a metric on logits: `logit`
(centred Euclidean) or `fisher` (diag p − ppᵀ, the KL to second order; this is the expected
Gram of the loss gradient). M is exact for the specimen (one backward pass per output position
and symbol).

| Rule | Subspace |
|---|---|
| `pca` (a) | top eigenvectors of C |
| `pcaperp` | the same after removing the direction 1 |
| `jac` (b, sensitivity alone) | top eigenvectors of M |
| `balanced` (b, White's rule) | top eigenvectors of C^{1/2} M C^{1/2}; T = C^{1/2}V, S = VᵀC^{-1/2}, an oblique projector. Its eigenvalues are the squared singular values of G^{1/2} J C^{1/2}, the map from what the past sends to what the output sees: the analogue of the Schmidt spectrum. It is the optimal rank-r truncation of a linear model (reduced-rank regression, balanced truncation). |
| `random` (c) | Haar-random subspace, mean of 5 |

## Numbers

All three models reach the floor (2 ln 2 + 3 ln 3)/80 = 0.058527: excess 3.9e-4 (D = 4, needed
400 steps), 4.1e-4 (D = 8, the specimen, 200 steps), 1.2e-4 (D = 16, 200 steps).

### Spectra (relative to the largest; `hankel` = √eig(CM))

    D = 8   C              1  .67  .23  .20  .11  .062 .050 .037
            M/logit        1  .50  .37  .15  .12  .088 .041 1e-16
            hankel/logit   1  .74  .70  .45  .37  .24  .10  0
            hankel/fisher  1  .60  .24  .18  .093 .064 .013 0
    D = 16  hankel/logit   1  .85  .46  .28  .16  .15  .11  .082 .049 .038 .032 .019 .016 .016 .0052 0
            hankel/fisher  1  .54  .18  .11  .054 .044 .023 .021 .016 .011 .0056 .0028 .0023 .0017 .00063 0

There is a spectrum. It has **one exact zero**, the direction 1 that no LayerNorm reads (M 1 = 0
to 1e-16; cutting it changes the logits by 1e-15). Otherwise it falls slowly and without a gap:
about a factor 0.7 per value. At D = 8 the seven visible values span one decade.

### Loss above the floor against kept rank r: straight after the cut → after 50 AdamW steps

Specimen, D = 8 (uniform guessing is 1.89 above the floor):

| r | pca | jac/logit | balanced/logit | balanced/fisher | random |
|---|---|---|---|---|---|
| 1 | 5.54 → 1.97 | 6.82 → 1.54 | 7.32 → 2.14 | 5.83 → 2.08 | 6.32 → 1.59 |
| 2 | 5.57 → 1.35 | 4.48 → 0.80 | 6.39 → 1.35 | 7.03 → 1.25 | 5.77 → 1.17 |
| 3 | 4.98 → 0.55 | 3.88 → 0.51 | 4.26 → 0.26 | 5.16 → 0.40 | 4.84 → 0.55 |
| 4 | 3.42 → 0.046 | 2.56 → 0.097 | 2.92 → 0.017 | 3.41 → 0.068 | 3.79 → 0.25 |
| 5 | 2.83 → 0.020 | 0.85 → 0.019 | 0.66 → 0.0047 | 2.18 → 0.015 | 2.78 → 0.071 |
| 6 | 1.21 → 0.0046 | 0.028 → 0.0007 | 0.028 → 0.0005 | 0.12 → 0.0009 | 2.02 → 0.036 |
| 7 | 0.71 → 0.0021 | 0.0004 → 0.0004 | 0.0004 → 0.0004 | 0.0004 → 0.0004 | 0.83 → 0.0038 |
| 8 | 0.0004 | 0.0004 | 0.0004 | 0.0004 | 0.0004 |

D = 16 (selected rows; all 16 are in `specimen.log`):

| r | pca | jac/logit | balanced/logit | balanced/fisher | random |
|---|---|---|---|---|---|
| 3 | 3.65 → 0.29 | 5.23 → 0.58 | 4.39 → 0.24 | 4.23 → 0.21 | 6.67 → 0.70 |
| 4 | 3.21 → 0.099 | 4.75 → 0.30 | 4.01 → 0.057 | 3.56 → 0.099 | 6.09 → 0.43 |
| 5 | 1.65 → 0.036 | 3.21 → 0.14 | 3.38 → 0.011 | 3.49 → 0.011 | 5.46 → 0.28 |
| 6 | 1.78 → 0.0092 | 1.59 → 0.0073 | 2.67 → 0.0028 | 3.06 → 0.0083 | 4.82 → 0.15 |
| 7 | 1.08 → 0.0028 | 1.07 → 0.0027 | 0.17 → 0.0005 | 1.98 → 0.0020 | 4.25 → 0.094 |
| 8 | 0.93 → 0.0007 | 0.84 → 0.0017 | 0.14 → 0.0003 | 0.98 → 0.0009 | 3.89 → 0.084 |
| 10 | 0.62 → 0.0011 | 0.050 → 0.0007 | 0.090 → 0.0002 | 0.14 → 0.0004 | 2.30 → 0.0054 |
| 12 | 0.0039 → 0.0001 | 0.018 → 0.0004 | 0.0010 → 0.0001 | 0.041 → 0.0002 | 1.76 → 0.0039 |
| 14 | 0.0003 | 0.0001 | 0.0001 | 0.0001 | 0.82 → 0.0009 |

D = 4: any rule that knows about the direction 1 is lossless at r = 3; every cut to r ≤ 2 is
ruinous (8 to 17 nats before retraining, 1.6 to 9.6 after).

Native models trained from scratch for 200 steps, best of 3 seeds, loss above the floor:
D = 2: 1.56; D = 4: 0.075 (0.0004 after 400 steps); D = 6: 0.0007; D = 8: 0.0003.

What the tables say:

1. **Without retraining, a cut is lossless only if it removes almost nothing.** Loss within 0.03
   of the floor needs r = 6 of 7 visible directions at D = 8 and r ≈ 11 of 15 at D = 16. Below
   that the cut model is worse than uniform guessing: it is confident and wrong. The trained
   model is not a width-r model plus small corrections, and the rank needed grows with D: the
   wider model spreads the same computation over more directions.
2. **After 50 steps the picture changes, and the needed rank no longer depends on D.** At r = 5
   the best rule stands at 0.005 (D = 8) and 0.011 (D = 16) above the floor, and both are
   within 0.001 at r = 6 to 7. The world needs about five stream directions with this head and MLP width,
   whatever width the parent had.
3. **Rule (b) beats rule (a), in its balanced form.** Straight after the cut, in the nearly
   lossless range, `balanced/logit` is 40 times better than `pca` at D = 8, r = 6 (0.028 against
   1.21) and 6 times at D = 16, r = 7 to 8. After retraining it is better at every r from 3 up to
   where both reach the floor: by a factor 2 to 9 at D = 8 (0.017 against 0.046 at r = 4,
   0.0005 against 0.0046 at r = 6) and 1.2 to 6 at D = 16 (0.011 against 0.036 at r = 5). Deep cuts of D = 16 before retraining go the other
   way (r = 5: pca 1.65, balanced 3.38), where every number is far above uniform anyway.
   Sensitivity alone (`jac`) is good near full rank and poor for deep cuts; the Fisher metric
   is no better overall than the plain logit metric (predictions at the floor are saturated, so the
   Fisher metric sees little).
4. **Random is far worse.** After retraining at D = 16, r = 6: random 0.15, the four rules
   0.003 to 0.009.
5. **A cut parent is not cheaper than a child trained directly**, as far as this goes: a native
   D = 4 model (three visible directions, 315 parameters) reaches 0.0004 in 400 steps from
   scratch, while the specimen cut to r = 3 or 4 and retrained for 50 steps stands at 0.26 and
   0.017, with 574 weights at r = 4 because it keeps the parent's heads and MLP. Equal-budget
   comparison was not run.

### The intertwiner R∘F₂ ≈ F₁∘R

R is the restriction x ↦ S (x − μₖ); F₁ is the cut model's step. Local defect: one step from
the full model's true state, |R F₂(x) − F₁(R x)| / |R F₂(x)| over the data. Drift: the cut model
run from the start. Readout compared by the mean KL between predictions (nats).

| D, r, rule | weights | attention step | MLP step | readout KL (local) | drift after MLP | KL of the whole model |
|---|---|---|---|---|---|---|
| 8, 6, balanced/logit | cut | 0.10 | 0.12 | 0.0007 | 0.18 | 0.027 |
| 8, 6, balanced/logit | retrained | 0.18 | 0.30 | 0.0011 | 0.37 | 0.0003 |
| 8, 6, pca | cut | 0.26 | 0.31 | 0.0001 | 0.45 | 1.21 |
| 8, 6, pca | retrained | 0.35 | 0.45 | 0.0048 | 0.48 | 0.0039 |
| 8, 4, balanced/logit | cut | 0.53 | 0.50 | 0.030 | 0.68 | 2.92 |
| 8, 4, balanced/logit | retrained | 0.52 | 0.59 | 0.011 | 0.62 | 0.016 |
| 16, 8, balanced/logit | cut | 0.32 | 0.22 | 0.0020 | 0.36 | 0.14 |
| 16, 8, balanced/logit | retrained | 0.37 | 0.45 | 0.0022 | 0.52 | 0.0003 |
| 8, 6, random | cut | 0.19 | 0.31 | 0.042 | 0.38 | 0.69 |

The embedding step is exact by construction before retraining (defect 1e-15) and 0.03 to 0.15
after. **The internal condition fails.** Per half-block the defect is 10 % at best and 20 to 60 % at
half rank. Retraining makes the predictions agree (KL down by a factor 100, to 3e-4 nats) while
the internal defect grows (0.12 → 0.30). So the retrained cut model is a coarse-graining of the
parent in the weak sense of the README (same predictions, here π is the identity on histories)
and not in the strong sense with this R. No rule is clearly best on the internal defect
(all rows are in `defects.tsv`).

### Shakespeare (810,049 parameters, D = 128, L = 4)

The checkpoint `best.bin` (step 5000) read from the author's main checkout, not retrained from
scratch. C and M from 16 training windows (M by 6 Gaussian probes per metric, a Hutchinson
estimate, not exact); losses on 24 validation windows of 256 characters. Full model 1.5447,
uniform guessing 4.174. The run took 6.6 min on a machine shared with other agents, longer
than the README's limit.

Spectra, relative to the largest, at ranks 1, 2, 4, 8, 16, 32, 64, 96, 127, 128:

    C              1  .67  .31  .15  .088 .046 .020 .0098 .0040 .0038
    M/logit        1  .71  .045 .029 .019 .013 .0075 .0047 .0022 2e-10
    hankel/logit   1  .58  .097 .054 .037 .025 .013 .0082 .0049 2e-6

Two large values, then a long flat tail: a factor 20 between rank 4 and rank 127, and again one
exact zero (the direction 1).

Validation loss of the cut model, no retraining:

| r | pca | jac/logit | jac/fisher | balanced/logit | balanced/fisher | random |
|---|---|---|---|---|---|---|
| 8 | 5.34 | 5.25 | 5.31 | 4.75 | 4.94 | 6.11 |
| 16 | 5.16 | 4.63 | 4.93 | 4.50 | 4.51 | 6.03 |
| 32 | 4.18 | 3.96 | 4.37 | 3.60 | 3.65 | 5.49 |
| 48 | 3.51 | 3.49 | 3.65 | 2.86 | 2.95 | 5.15 |
| 64 | 2.87 | 2.94 | 3.02 | 2.28 | 2.31 | 4.96 |
| 96 | 1.87 | 2.09 | 2.09 | 1.72 | 1.73 | 3.93 |
| 112 | 1.636 | 1.716 | 1.713 | 1.599 | 1.597 | 3.53 |
| 127 | 1.549 | 1.5447 | 1.5447 | 1.5447 | 1.5447 | 1.620 |

After 30 AdamW steps (batch 16, η = 3e-4, cut fixed), validation loss before → after:

| r | pca | balanced/fisher | random |
|---|---|---|---|
| 32 | 4.18 → 2.51 | 3.65 → 2.30 | 5.49 → 3.13 |
| 64 | 2.87 → 2.08 | 2.31 → 1.92 | 4.96 → 2.72 |

The same picture at 800 times the size. The balanced rule beats PCA at every rank, and the
loss above the full model is roughly halved (r = 64: 0.74 against 1.32; r = 96: 0.17 against
0.33; r = 112: 0.054 against 0.091); the two metrics agree. The loss does not fall fast: half
the stream costs 0.74 nats, a quarter is near uniform guessing, and staying within 0.05 needs
r ≈ 112 of 127. Thirty steps recover much of it and keep the order of the rules (r = 64:
1.92, 2.08, 2.72), but 30 steps is far too few to say what rank this model needs after
retraining, and the intertwiner was not measured here.

## Verified, and how

`test.jl`, 136 checks on two random models (D = 8, L = 1 and D = 4, L = 2):

- with no cut, `tgpt` equals `gpt` and its gradient, bit for bit;
- the gradient of the cut loss against central differences, relative error below 1e-6, for
  orthogonal, oblique and random cuts at two ranks; the stream adjoints at every site likewise,
  with and without a cut;
- the cut is idempotent; a full-rank cut changes the logits by less than 1e-8 and has zero
  intertwiner defect; the rank D − 1 cut by `jac` changes nothing (the direction 1 is free);
- the cut model in r coordinates (`smallgpt`, the LayerNorm formula above) equals the projector
  form to 1e-10, logits and streams;
- M 1 = 0; M from basis probes equals M from explicit token-by-token Jacobians, and agrees
  with 4000 Gaussian probes to 10 %.

A bug was found by these tests and fixed before any number here was produced: a closure variable
shared between threads made M wrong by a factor of the number of thread groups.

Not verified: seeds (one training seed per width; the random rule alone is averaged), and the
retraining recipe (50 steps, η = 1e-2 → 1e-3, one setting, not tuned; at D = 4 it lifts the
uncut model from 0.0004 to 0.0012 above the floor, so it is a little too hot there).

## What failed, and what was not done

- The strong, internal coarse-graining fails for every rule (table above).
- No rule gives a usable small model without retraining unless r is within one or two of the
  visible rank.
- Not done for lack of time: a projector per site (each "bond" with its own basis, as in DMRG);
  optimising the projector directly against the loss (an upper bound on what any rule can do);
  refitting R after retraining; cutting head width and MLP width by the same rule; an
  equal-budget comparison against training the small model from scratch; the linear cut
  (μ = 0) as a variant.

## Verdict

**Needs more**: a relevant spectrum exists and White's rule (balanced, output-weighted) beats PCA (by a factor 1.2 to 40 on the specimen family, about 2 in excess loss on Shakespeare), but the spectrum falls slowly, a cut is lossless only next to full rank, the cut model leaves the family through LayerNorm, and the internal intertwiner fails (10 to 50 % per half-block); what survives is a good starting point for a short retraining, whose required rank on the specimen's world (about 5) does not grow with the parent's width; the Shakespeare model shows the same ordering of rules and the same slow fall.

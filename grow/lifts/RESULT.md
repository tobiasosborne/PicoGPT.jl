# Exact lifts on every axis — result

24 November 2026. Julia 1.13.3, one thread, PicoGPT's `src/` used as a library and unchanged.
Work stopped at a deadline (15:55 UTC); what was not reached is listed at the end.

**In one paragraph.** Every axis has an exact lift, tested to rounding (largest error 2.4e-15),
except two: width in PicoGPT *as written*, because its sinusoidal positional code changes with D,
and the number of heads at fixed D, which is a rank restriction and not a growth axis. With the
positional code duplicated along with the coordinates, the width lift is exact. On the specimen,
growing D 4 → 8 did **not** reduce the work to reach within 1 % of the floor: the best grown arm
equals training at full size from scratch (126 against 124 thousand parameter-updates), the others
cost 1.2 to 2.4 times as much, and wall time is worse in every grown arm. From a D = 2 parent,
which cannot reach the floor by itself, every grown arm costs 1.2 times scratch or more. Plain duplication is a
symmetry trap: the copies stay equal to the last bit for as long as we trained.

## Files

| File | What it is | Run time |
|---|---|---|
| `Lifts.jl` | the lifts, the model with its positional code as data, a trainer | – |
| `test_lifts.jl` → `out/test_lifts.txt` | 34 tests: exactness of each lift, finite-difference checks | 1 to 2 min (mostly compiling) |
| `grow_experiment.jl` → `out/growth.txt`, `out/growth.tsv` | grown against scratch from a D = 4 parent, 8 seeds | about 2 min |
| `grow_experiment.jl 2` → `out/growth_D2.txt`, `out/growth_D2.tsv` | the same from a D = 2 parent | about 2 min |
| `symmetry.jl` → `out/symmetry.txt` | what training does to the duplicated coordinates, 6 seeds | about 1 min |

Run each as `julia --project=. -t 1 grow/lifts/<file>` from the repository root.

## 1. The lifts

Notation: U = I_D ⊗ 1_k duplicates each coordinate k times (UᵀU = k I); Uᵀ/k averages the copies.
"Error" is the largest absolute difference of a logit between child and parent, on random
parameters (PicoGPT's initial draw plus 0.3·N(0,1) on every entry, so that gains, offsets and
biases are all nontrivial) and random token sequences (6 sequences of length 8, V = 7, L = 2).

| Axis | Lift | Error | Exact? |
|---|---|---|---|
| Width D → kD, H kept (d → kd) | duplicate coordinates; k = 2, 3 | ≤ 2.4e-15 | yes, with the positional code duplicated |
| Width D → kD with heads H → kH (d kept) | each head copied k times, Wo averages them | ≤ 1.3e-15 | yes, same condition, no constant |
| Heads, new ones | as above, but the k − 1 new heads per old head are random with zero columns in Wo | ≤ 1.7e-15 | yes |
| Heads H → 2H at fixed D | best rank-d/2 score form per head | 0.95 | **no** (exact only for low-rank parents: 1.0e-15) |
| MLP width F → F′ | new units: random W1 rows, zero W2 columns | 0 | yes |
| MLP width F → kF | units duplicated, W2 averages | 4.4e-16 | yes |
| Depth L → L + 1 | new block with Wo = bo = W2 = b2 = 0, at any of the 3 positions | 0 | yes |
| Context T → T′ | θ unchanged | 0 | yes, on positions t ≤ T |
| Vocabulary | split token v into m | 4.4e-16 | yes, in the sense below |
| All five composed (p 559 → 2768) | | 8.3e-16 | yes |

### Width: what each piece requires

- **LayerNorm**: nothing. The mean and the variance of U x are those of x, so LN′(U x) = U LN(x)
  with γ′ = U γ, β′ = U β; the ε inside the square root is the same too. Zero-padding instead of
  duplicating does not commute with LayerNorm (error 0.97), because padding changes both the mean
  and the variance.
- **ReLU**: nothing; it acts in the MLP's hidden layer, which the width lift does not touch.
- **The √d in the scores**: with H kept, q′⋅k′ = k q⋅k and the divisor is √(kd), so the scores
  come out √k too large. The cure is one constant, k^(−1/2), put as k^(−1/4) on each of Wq, bq, Wk,
  bk. Without it the error is 0.27. With the heads multiplied instead (d kept) no constant is
  needed.
- **The readout**, and every other reader of the stream (Wq, Wk, Wv, W1, Wout): W′ = W Uᵀ/k, the
  average of the copies. Writers (E, Wo, bo, W2, b2) are forced: W′ = U W.
- **The positional code: this is the obstruction.** PicoGPT's code is `positions(D, T)`, a fixed
  function of D, and positions(2D, T) ≠ U positions(D, T). It contains positions(D, T) exactly
  (rows 4i−3, 4i−2 of the wide code are rows 2i−1, 2i of the narrow one; tested), but its other
  D rows carry a new position signal that no parameter can cancel and that enters the mean and
  variance of every LayerNorm. The duplication lift evaluated with PicoGPT's own code is off by
  1.45 in the logits. So **within PicoGPT as written I have no exact width lift**; the exact lift
  needs the child's code to be U P, which `Lifts.Net` carries as data. I did not prove that no
  other exact lift exists, and did not compute the smallest achievable error.
- **The lift is not unique.** A reader needs only W′ U = W, so W′ = W Uᵀ/k + Ξ (I − UUᵀ/k) is
  exact for every matrix Ξ: the second term reads the antisymmetric part of the stream, which is
  zero at birth (tested with Ξ as large as a fresh initial draw). This freedom is what breaks the
  symmetry in section 3.

### The other axes

- **Heads at fixed D are not a growth axis.** In PicoGPT D = H·d, so more heads at fixed D means
  narrower heads. Head h's scores are the bilinear form [x̂ᵢ; 1]ᵀ M x̂ⱼ / √d with
  M = [Wq bq]ᵀ Wk of rank ≤ d; a sub-head of width d/m carries rank ≤ d/m. Splitting is exact when
  rank M ≤ d/m (tested, 1.0e-15) and not otherwise: for random parameters the best rank-d/2
  forms discard 27 to 56 % of ‖M‖ and move the logits by 0.95. Merging heads fails for the
  opposite reason (one head has one attention pattern). New heads must come with new width.
- **bk is a gauge direction.** q⋅bk is the same for every key, softmax ignores it, and the
  gradient of the loss with respect to bk is zero to rounding (tested, < 1e-14).
- **MLP**: a third exact lift, zero W1 rows with random W2 columns, is stillborn: relu(0) = 0
  with backward value 0, so the new units get exactly zero gradient (tested).
- **Depth** needs no gate parameter: the writing matrices Wo and W2 of the new block are the gate.
  At birth only Wo, bo, W2, b2 have a nonzero gradient.
- **Context**: no parameter depends on T, and column t of the code depends only on t. The child's
  logits at positions t ≤ T of a longer sequence are the parent's on the prefix, bit for bit.
  What the child does at t > T was not measured.
- **Vocabulary**: with every copy of v given v's embedding and readout row and
  bout′(v, b) = bout(v) + log q_b, the child's logits are z′(v, b) = z(v) + log q_b and
  z′(u) = z(u), so Σ_b p′(v, b | h′) = p(v | πh′) (1.7e-16) and the loss is the parent's loss on
  the coarse text plus the mean of −log q_b (1.7e-16). The partition function is unchanged.

## 2. Growing the specimen, D 4 → 8

Corpus `to be or not to be`, all 10 windows of length 8, full batch. Floor 0.058527 nats, target
1.01 × floor = 0.059112. AdamW as for the specimen (β = 0.9, 0.95; weight decay 0.01 on matrices;
clipping at 1) with a 20-update warm-up to a constant step 2e-2, restarted with fresh moments
after the lift. Parent: D = 4, H = 2, F = 16, 315 parameters. Child: the specimen's shape, D = 8,
H = 2, F = 32, 1007 parameters, with the duplicated positional code. The lift is width
duplication followed by 16 fresh MLP units; the loss at birth equals the parent's to 4e-16 in all
72 lifts. Work = Σ over updates of the number of parameters of the model being updated,
in thousands. 8 seeds; every run reached the target. Median [min, max].

| Arm | parent updates | child updates | work / 1000 | wall, ms |
|---|---|---|---|---|
| scratch, D = 8, PicoGPT's code (the specimen) | – | 124 [109, 135] | 124 [110, 136] | 106 |
| scratch, D = 8, duplicated code | – | 121 [102, 142] | 122 [103, 143] | 72 |
| parent D = 4 alone, never lifted | 386 | – | 122 [100, 177] | 255 |
| grown, asymmetric readers | 50 | 109 [98, 179] | 126 [114, 196] | 133 |
| grown, asymmetric readers | 100 | 118 [71, 255] | 150 [103, 288] | 150 |
| grown, asymmetric readers | 200 | 128 [87, 162] | 192 [151, 226] | 236 |
| grown, plain duplication | 50 | 166 [117, 263] | 183 [134, 281] | 163 |
| grown, plain duplication | 100 | 174 [127, 287] | 206 [159, 321] | 208 |
| grown, plain duplication | 200 | 164 [125, 242] | 229 [189, 307] | 270 |
| grown, parent frozen, only the 692 new directions trained | 50 | 278 [162, 346] | 295 (208 counting 692) | 3131 |
| grown, parent frozen | 100 | 222 [147, 328] | 255 (185) | 2476 |
| grown, parent frozen | 200 | 124 [80, 267] | 188 (149) | 1870 |

What this says:

- **Growing is not cheaper here.** After the lift the child needs about as many updates as a
  model trained from scratch (109 to 174 against 124), whatever the parent had already learned,
  so the parent's updates are pure extra cost. The best arm (50 parent updates, asymmetric
  readers) ties with scratch; training the parent longer makes the total worse.
- **The control spoils the premise.** The 315-parameter parent reaches the target by itself, in
  about three times as many updates, for the same work (122) as the full-size model. On this
  specimen width 8 buys speed per update, not a lower loss, so there is nothing for growth to add.
- **Frozen parent**: with the parent's 315 directions held fixed (the child moves only in the
  orthogonal complement of the lift's image), the target was still reached in 24 of 24 runs, in 80
  to 346 child updates: slower than training everything, except after the longest parent run.
  Its wall time is an artefact of multiplying by a dense 1007 × 692 basis at every update.
- Wall time follows the number of updates, not the parameter count (an update costs about 0.6 to
  1 ms at either width), so by the clock every grown arm loses to scratch.
- The duplicated positional code trains as fast as PicoGPT's own (121 against 124 updates).

### The same from a parent that cannot reach the floor, D 2 → 8

Because the D = 4 parent reaches the target alone, the experiment was repeated with a parent
that does not: D = 2, H = 2, F = 8, 113 parameters, lifted by fourfold duplication (plus 24 fresh
MLP units) to the same child. Alone, this parent stalls at 1.37 to 1.49 nats (the floor is 0.0585)
in all 8 seeds after 3000 updates. Same protocol, 8 seeds.

| Arm | parent updates | child updates | work / 1000 | reached |
|---|---|---|---|---|
| scratch, D = 8, PicoGPT's code | – | 124 [109, 135] | 124 [110, 136] | 8/8 |
| scratch, D = 8, fourfold-duplicated code | – | 127 [112, 151] | 128 [113, 152] | 8/8 |
| parent D = 2 alone | 3000 (cap) | – | – | 0/8 |
| grown, asymmetric readers | 50 | 148 [118, 254] | 154 [124, 261] | 8/8 |
| grown, asymmetric readers | 100 | 142 [104, 196] | 155 [116, 209] | 8/8 |
| grown, asymmetric readers | 200 | 145 [112, 230] | 169 [135, 254] | 8/8 |
| grown, plain duplication | 50 | 276 [224, 478] | 283 [231, 487] | 8/8 |
| grown, plain duplication | 100 | 242 [157, 3000] | 254 [169, 3032] | 7/8 |
| grown, plain duplication | 200 | 199 [151, 296] | 223 [175, 321] | 8/8 |
| grown, parent frozen | 50 | 196 [182, 216] | 204 [189, 223] | 8/8 |
| grown, parent frozen | 100 | 212 [172, 315] | 225 [185, 329] | 8/8 |
| grown, parent frozen | 200 | 232 [160, 307] | 257 [184, 332] | 8/8 |

Here the lift is doing real work (the child reaches a floor its parent cannot), but it is still
no cheaper: the child alone needs more updates after the lift (142 to 148) than a fresh model needs
in total (124 to 127). One plain-duplication run stalled at 0.0754 for 3000 updates. Freezing the
parent's 113 directions costs about 1.4 times the child updates of training everything.

## 3. The symmetric and the new directions

Parent trained 100 updates, lifted, trained 400 more; medians over 6 seeds. "Stream asymmetry" is
the fraction of the stream's norm, after the block, outside the duplicated subspace range(U).

| Lift | after 1 | 5 | 20 | 100 | 400 updates |
|---|---|---|---|---|---|
| width duplicated, F kept | 0 | 0 | 0 | 0 | 0 |
| width duplicated, MLP units duplicated too | 0 | 0 | 0 | 0 | 0 |
| width duplicated, 16 fresh MLP units | 0 | 9.2e-5 | 3.4e-3 | 6.5e-2 | 0.22 |
| width duplicated, noise 1e-3 on θ′ (inexact) | 1.4e-3 | 1.3e-3 | 1.6e-3 | 1.1e-2 | 0.15 |
| asymmetric readers ×0.1, fresh MLP units | 2.8e-4 | 2.5e-3 | 1.5e-2 | 0.11 | 0.24 |
| asymmetric readers ×1, fresh MLP units | 1.9e-3 | 1.7e-2 | 9.8e-2 | 0.23 | 0.36 |

- **Plain duplication never separates.** Permuting the copies is a symmetry of the loss (it needs
  the duplicated positional code), the lifted point is fixed by it, so the gradient is fixed too,
  and AdamW, clipping and weight decay all respect the symmetry. The copies stayed equal exactly,
  zeros in both the stream and the parameters, for 400 updates in all 6 seeds; rounding did not
  break it. The child is then a width-4 stream with wider heads. It still learns, more slowly:
  after 400 updates it is 1.9 % above the floor (F kept) or 13 % (MLP units duplicated), against
  0.7 to 0.9 % for the three exact symmetry-broken lifts (which also have 16 new MLP units).
- **The symmetry has to be broken, and it can be broken exactly.** Asymmetric readers give a
  first-order asymmetric gradient: the stream is 1 % asymmetric after 3 to 5 updates and 10 %
  after 20. Fresh MLP units also break it, but at second order (their writing columns start at
  zero): nothing after 1 update, 0.3 % after 20, 6.5 % after 100. Noise of 1e-3 sits still for
  about 20 updates and then grows.
- In the growth experiment this is the difference between the two "grown" rows: asymmetric
  readers need 109 to 128 child updates, plain duplication with fresh MLP units 164 to 174.

## Verified, and how

- Each exact lift: logits compared on random parameters and inputs, tolerance 1e-11, observed
  ≤ 2.4e-15 (`test_lifts.jl`, 34 tests pass).
- The model copy `Lifts.logits` equals `PicoGPT.gpt` bit for bit when U = I. Its pullback (the
  only gradient written here, with the mean-squared-error loss) was checked by central
  differences on a duplicated-code model: relative error 1.7e-8 (loss) and 3.4e-9 (squared error).
- Negative controls in the same test file: each broken variant (no score constant, PicoGPT's
  code, zero-padding, generic head split) is tested to be wrong by more than 1e-3.
- Loss at birth equals the parent's loss in every lift of the growth experiment (≤ 4e-16).

## Failed, or not reached

- No exact width lift inside PicoGPT as written (positional code). The smallest achievable
  error was not computed; 1.45 is the error of the plain duplication lift, not a minimum.
- Not reached before the deadline: a growth experiment on any axis but width (with MLP width); a
  sweep of the step size, or of the warm-up after the lift; lifting Adam's moments instead of
  restarting them; what the child does at positions t > T; a task where the full-size model is
  itself slow to train, which is where a head start could matter.
- Derived but not measured: the duplicated readers W Uᵀ/k hold each parent weight as k copies of
  w/k, so the same step on every copy moves the effective weight k times as far; the lift
  multiplies the effective step size of the readers by k.
- One trainer configuration only (η = 2e-2). The comparison could change with tuning.

**Verdict: needs more** — the lifts are exact and in hand (width needs a duplicated positional
code and a symmetry-breaking reader), but on the specimen growing by width costs at least as much
as training at full size (1.0 to 2.4 times), from a parent that reaches the floor and from one
that does not; as a way to save training work it is dead at this scale, and untested at any other.

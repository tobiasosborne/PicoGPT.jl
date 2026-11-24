# A synthetic tower of worlds with dials — result

**Verdict: needs more, leaning negative.** The lift is exact, the entropy bookkeeping is exact, and
the answer to "does staging pay?" at this scale is: only when the two children of a class are
interchangeable *as inputs*. As soon as the fine symbol matters for prediction, the lifted child is
no faster than a model trained from scratch (1.2–2.4 times slower with the exact lift, about equal
with a symmetry-breaking lift), and the parent's own cost rises steeply with the defect.

Everything here reruns with `julia exact.jl` (seconds) and
`OPENBLAS_NUM_THREADS=1 julia -t2 --project=../.. stage.jl 5` (about 4 minutes), Julia 1.13,
stock PicoGPT from `src/` (unchanged).

## 1. The world (`world.jl`, `exact.jl`)

Three classes, two children each, v = 2(c−1)+b. Fine transition p(v'|v) = Pc(c'|v)·q(b'|c',v) with

- Pc(c'|v) ∝ P(c'|c)·exp(±δ·U[c,c']), sign by child; P = [.7 .2 .1; .1 .6 .3; .45 .05 .5];
- q(b'=1|c',v) = σ(κ·(S[c',c] ± ε·R[c',c])), sign by child.

**A correction to the brief.** With the form p(v'|v) = P(c'|c)·q(v'|c',v) the class transition never
depends on the child, so the chain is exactly lumpable and the defect I(c';v|c) is zero *for every
ε*. ε only changes the split entropy and makes the child matter for the split. To get a nonzero
defect a third dial δ was added, which lets the class transition depend on the child. So the dials
are: κ (how peaked the split is), ε (the split depends on the child; defect stays 0), δ (the defect).

Exact values in nats (T = 8; "coarse" is the floor for a model that sees t = 1…8 classes, computed
by enumerating all coarse histories with a forward filter; for δ = 0 it equals H(c'|c)):

| ε | κ | δ | coarse floor | H(c'\|c) | H(c'\|v) | split H(v'\|c',v) | fine floor | defect |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0.84151 | 0.84151 | 0.84151 | 0.69315 | 1.53466 | 0 |
| 0 | 1 | 0 | 0.84151 | 0.84151 | 0.84151 | 0.60567 | 1.44718 | 0 |
| 1 | 1 | 0 | 0.84151 | 0.84151 | 0.84151 | 0.51581 | 1.35733 | 0 |
| 1 | 2 | 0 | 0.84151 | 0.84151 | 0.84151 | 0.32250 | 1.16401 | 0 |
| 1 | 1 | 0.25 | 0.81225 | 0.81445 | 0.79292 | 0.50805 | 1.30096 | 0.02153 |
| 1 | 1 | 0.5 | 0.78408 | 0.79383 | 0.71333 | 0.50111 | 1.21444 | 0.08050 |
| 1 | 1 | 1 | 0.72824 | 0.76942 | 0.51515 | 0.49211 | 1.00726 | 0.25427 |
| 1 | 1 | 2 | 0.61628 | 0.71610 | 0.20077 | 0.48491 | 0.68568 | 0.51533 |
| 0 | 1 | 1 | 0.77391 | 0.79467 | 0.53039 | 0.60744 | 1.13783 | 0.26428 |

Verified (asserted in `exact.jl`): the chain rule fine floor = H(c'|v) + split holds with residual
≤ 4.4e-16; the defect is 0 to 1e-12 and coarse floor = H(c'|c) whenever δ = 0. Note that with δ > 0
the coarse process is no longer Markov: a longer coarse history recovers part of the defect
(coarse(T=8) < H(c'|c)), but never all of it (coarse > H(c'|v)).

## 2. The lift (`stage.jl`, function `lift`)

Parent: PicoGPT with V=3, D=8, H=2, F=32, L=1, T=8 (939 parameters). Child: the same with V=6 (990).
Child embedding = class embedding; child readout row and bias = the class's. The logits are then
z(v) = z_parent(class v) + 0, so the softmax over six gives p_parent(c)/2 for each child. This is
the stock V=6 head, not a separate two-stage (class, then within-class) head: it is exact at birth,
but after birth the class marginal is free to move. No new gradient code was written, so there is
no new finite-difference check (the stock pullbacks are checked in `test/`).

Tested on 4,000 test tokens, every setting and seed (50 runs):

- max |Σ_{v∈c} p_child(v|h) − p_parent(c|πh)| = **2.2e-16**;
- max |child loss at birth − (parent loss + ln 2)| = **3.0e-14**.

Against the exact floors, the child at birth sits above the fine floor by (ln 2 − split) + defect
+ the parent's own excess: measured birth excess 0.007 (κ=0: nothing to learn, the child is born
within 1 %), 0.097, 0.181, 0.380 for the δ = 0 rows (predicted ln 2 − split = 0, 0.087, 0.177,
0.371), and 0.20, 0.27, 0.43, 0.64 for δ = 0.25…2 at ε=κ=1.

## 3. Staged against scratch (`results.tsv`, `run.log`)

Same optimiser for every arm (AdamW, η = 0.01, clip 1, batch 32 fresh windows of 8, so 256 tokens
per step). A run stops when its loss on a fixed 4,000-token test set is within 1 % of the floor
(measured as excess over the true process's surprise on the same tokens), checked every 5 steps.
Work = parameters × tokens × steps. Five seeds; medians.

| ε | κ | δ | defect | parent steps | child steps | scratch steps | W child / W scratch | W (parent+child) / W scratch |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 30 | 0 | 35 | 0 | 0.81 |
| 0 | 1 | 0 | 0 | 30 | 10 | 30 | 0.33 | 1.45 |
| 1 | 1 | 0 | 0 | 30 | 190 | 80 | 2.38 | 2.91 |
| 1 | 2 | 0 | 0 | 30 | 265 | 175 | 1.51 | 1.62 |
| 1 | 1 | 0.25 | 0.022 | 35 | 140 | 70 | 2.00 | 2.62 |
| 1 | 1 | 0.5 | 0.080 | 165 | 115 | 100 | 1.15 | 3.32 |
| 1 | 1 | 1 | 0.254 | 685 | 145 | 95 | 1.53 | 8.10 |
| 1 | 1 | 2 | 0.515 | 1155 | 240 | 140 | 1.71 | 9.72 |
| 0 | 1 | 1 | 0.264 | 660 | 155 | 90 | 1.72 | 8.73 |
| 0 | 1 | 2 | 0.541 | 790 | 130 | 90 | 1.44 | 9.71 |

Every run reached the target within the 1,500-step cap.

What this says:

1. **The child alone beats scratch only when the fine symbol is irrelevant as an input** (ε = 0 and
   δ = 0: the child needs only new readout biases; 0–10 steps against 30–35). Even then the whole
   lineage is cheaper only in the trivial κ = 0 world.
2. **Once the child matters as an input (ε > 0 or δ > 0) the lifted child is slower than scratch**,
   by 1.2–2.4 times, at zero defect as much as at large defect. So the quantity that kills the
   child's advantage here is not the defect but whether prediction depends on the child at all.
3. **The defect prices the parent, not the child.** Parent steps go 30 → 35 → 165 → 685 → 1155 as
   the defect goes 0 → 0.02 → 0.08 → 0.25 → 0.52, because the coarse world stops being Markov and
   its floor needs inference over the hidden child, while the fine world stays first-order Markov
   and stays easy (scratch 70–140 steps). Whole-lineage work over scratch therefore grows
   monotonically with the defect at fixed ε = κ = 1 from δ = 0.25 on: 2.6, 3.3, 8.1, 9.7. In this
   sense the advantage of staging does shrink as the defect grows, but it was already gone
   at defect zero. A non-closed coarse world is *harder* than the fine world it came from.

### Diagnostic: is the slow child a symmetry problem? (`run_noise0.3.log`, `run_noise1.0.log`)

The exact lift gives the two children identical embeddings. Splitting them at birth by ±n,
n ~ N(0, noise²) per coordinate (no longer exact: the class marginal moves by up to 0.4–0.8),
`NOISE=0.3 julia … stage.jl 5 sub`:

| ε | κ | δ | child/scratch, exact | noise 0.3 | noise 1.0 |
|---|---|---|---|---|---|
| 0 | 1 | 0 | 0.33 | 0.30 | 0.57 |
| 1 | 1 | 0 | 2.38 | 1.27 | 0.90 |
| 1 | 2 | 0 | 1.51 | 1.73 | 0.89 |
| 1 | 1 | 0.5 | 1.15 | 1.24 | 1.06 |
| 1 | 1 | 2 | 1.71 | 1.39 | 1.32 |

(The scratch arm draws different random numbers in these runs, so its medians differ a little from
the table above.) Symmetry explains part of the handicap: with embeddings as far apart as a fresh
initialisation the child is level with scratch, not ahead. The inherited parent is then worth
roughly nothing in steps.

## What is not established

- **Third level (12 symbols): not reached.**
- One model size, one learning rate, one threshold. With stopping times of 30–250 steps checked
  every 5 on a 4,000-token test set, single ratios are noisy (seed ranges are in `run.log`); the
  ordering child-slower-than-scratch for ε > 0 holds in the medians of all eight such settings but
  no interval was computed.
- The world is first-order Markov and a one-block model learns it in about 100 steps, so there is
  little for a parent to pass on. Staging might pay where the coarse structure is expensive to
  learn and the fine addition is cheap; this world is the opposite case for δ > 0. A fair next test
  needs a coarse level that is hard (long-range) with a fine level that is closed over it.
- A true two-stage head (class softmax times within-class softmax) with the class part frozen or
  slowed was not tried; nor freezing the parent's block, nor a lower learning rate for inherited
  weights.
- Why the exact-lift child is slower than scratch beyond the symmetry effect was not analysed.

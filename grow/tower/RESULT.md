# A tower of worlds: result

24 November 2026. Julia 1.13.3, PicoGPT's `src/` used unchanged.

**Scope.** Only the specimen's text was done: the exact floors, T1 on the coarse text, the
vocabulary lift, and staged training against scratch. The synthetic world with a dial was
dropped on instruction before it produced a result (another agent has it); no code for it
is kept here.

## What was tried

The fine world is the specimen's: the 80 predictions of the ten windows of length 8 of
`to be or not to be`, 7 symbols. The coarse world is its image under π: vowel (e, o),
consonant (b, n, r, t), space. The coarse text is `CVSCVSVCSCVCSCVSCV`.

| File | What it does | Runs in |
|---|---|---|
| `floors.jl` | the floors, by enumeration, as exact integer combinations of ln 2 and ln 3 | 2 s |
| `specimen_tower.jl` | T1, the lift, the split loss, staged against scratch, 5 seeds | 87 s on 3 threads |
| `common.jl` | `splitloss`, `lift`, the training loop with a freezing mask | |
| `out/*.txt` | the printed output of the two scripts, as quoted below | |

Run from the repository root: `julia --project=. -t 3 grow/tower/specimen_tower.jl`.

## 1. The exact floors

h is the fine history (the window so far), πh its coarse image, a the next symbol, v its
class. All in nats per prediction; the numerators are over 80.

| Floor | Exact | Value | Undetermined predictions |
|---|---|---|---|
| H(a \| h), the fine text | (2 ln 2 + 3 ln 3)/80 | 0.058527 | 5 of 80 |
| H(v \| πh), the coarse text read coarsely | (8 ln 2 + 6 ln 3)/80 | 0.151711 | 19 of 80 |
| H(v \| h), the class given the fine history | (3 ln 3)/80 | 0.041198 | 5 of 80 |
| H(a \| v, h), the refinement term | (2 ln 2)/80 | 0.017329 | 2 of 80 |
| H(a \| v, πh), a refinement blind to the fine past | (2 ln 2 + 3 ln 3)/80 | 0.058527 | 7 of 80 |
| H(a \| πh), the fine text read coarsely | (10 ln 2 + 9 ln 3)/80 | 0.210237 | 21 of 80 |

- The chain rule holds exactly, as an identity of the integer coefficients:
  H(a | h) = H(v | h) + H(a | v, h), that is (2 ln 2 + 3 ln 3) = (3 ln 3) + (2 ln 2); and
  likewise H(a | πh) = H(v | πh) + H(a | v, πh).
- **The coarse floor depends on which history the model sees, by a factor of 3.7.** Seen
  coarsely the classes cost 0.1517; seen finely, 0.0412. The difference,
  I(v; h | πh) = (8 ln 2 + 3 ln 3)/80 = 0.1105 nats, is what the coarse history hides about the
  next class. The coarse text is the *harder* text: 19 undetermined predictions against 5.
  Merging letters makes histories collide (`to ` and `be ` are both CVS) while the next class
  still depends on which one it was.
- So this coarse-graining is not exact (not lumpable). A T2 whose parent stays frozen on the
  coarse history has the floor H(v | πh) + H(a | v, h) = (10 ln 2 + 6 ln 3)/80 = 0.169039,
  which is 2.89 times the fine floor, whatever q does.

## 2. T1, the lift, and the split loss

T1 is PicoGPT with V = 3, D = 8, H = 2, F = 32, L = 1, T = 8 (939 parameters), trained on the
coarse windows with the specimen's recipe (AdamW, full batch, 200 updates, warmup and cosine).

Two constructions of T2, p(v, b | h) = p(v | ·) q(b | v, h):

- **Two streams.** p(v | πh) is T1 itself on the coarse history, frozen. q is a new small
  PicoGPT reading the fine history; its 7 logits are normalised inside each class
  (q(a | v, h) = exp z_a / Σ_{a′ ∈ v} exp z_a′). Only q is trained, on the refinement term.
- **One stream (`lift`).** A specimen-shaped model (1,007 parameters) born from T1: each fine
  symbol inherits the embedding column and readout row of its class, and its readout bias is
  the class bias minus ln |class|. At birth it reads a fine history as its coarse image and
  predicts p(v | πh)/|v|. Its ordinary cross-entropy splits, for any weights, into a class
  term −log Σ_{a ∈ v} p_a and a refinement term (`splitloss`).

### Verified to rounding (seed 1)

| Statement | Error |
|---|---|
| lift: max \|Σ_b T2(v, b \| h) − T1(v \| πh)\| over 3 classes × 80 contexts | 4.4e-16 |
| lift: max \|T2(a \| h) − T1(cls a \| πh)/\|cls a\|\| | 4.4e-16 |
| lift: class term at birth − T1's loss | −1.7e-16 |
| lift: refinement term at birth − mean ln \|class\| (predicted 0.762462 nats) | +1.2e-15 |
| lift: class term + refinement term − `loss(θ2)` | +2.2e-16 |
| two streams, random q: max \|Σ_b T2(v, b \| h) − T1(v \| πh)\| | 2.2e-16 |
| two streams: −mean log p(a \| h) − (T1's loss + q's refinement term) | 0.0 |
| ℓc + ℓr − `crossentropy` on random logits | −4.4e-16 |

So the loss at birth is the parent's loss plus mean ln |class| = (88 ln 2)/80 = 1.1 ln 2 =
0.762462 nats, as predicted: each prediction pays ln of the size of its target's class
(ln 2 for a vowel, ln 4 for a consonant, 0 for the space).

Pullbacks against central differences (relative error): `splitloss` class term 3.7e-9,
refinement term 1.6e-9; θ̄ of each term through `gpt`, all 1,007 coordinates, 9.4e-9 and 5.5e-9.

### Training (median over seeds 1–5, [min–max]; 200 updates each)

"To ε" is the number of updates after which the loss is within ε of its floor.

| Phase | Trained | Floor | Final | To ε = 1e-2 | To ε = 1e-3 |
|---|---|---|---|---|---|
| Scratch specimen, fine text | 1007 | 0.05853 | 0.05886 | 64 [55–73] | 108 [103–123] |
| T1, coarse text | 939 | 0.15171 | 0.15258 | 137 [112–181] | never in 2 of 5 |
| q alone, D = 4, F = 8 | 243 | 0.01733 | 0.01767 | 90 [79–114] | never in 1 of 5 |
| q alone, D = 4, F = 16 | 315 | 0.01733 | 0.01827 | 113 [64–140] | never in 2 of 5 |
| q alone, D = 8, F = 32 | 1007 | 0.01733 | 0.01742 | 46 [45–59] | 66 [64–82] |
| Lifted parent, class term only, fine history | 1007 | 0.04120 | 0.04123 | 36 [25–37] | 49 [42–67] |
| Lift, all trained | 1007 | 0.05853 | 0.05895 | 75 [73–99] | 126 [104–142] |
| Lift, only E, Wout, bout trained | 119 | 0.05853 | 0.12194 | never | never |

### Whole lineages against scratch

Work = Σ over phases of (parameters trained) × (updates); every update is one full-batch
pass over the same 80 predictions. Unit: the specimen's 1,007 × 200.

| Lineage | Ends at | Its floor | Work, 200 updates per phase |
|---|---|---|---|
| Scratch specimen | 0.05886 | 0.05853 | 1.00 |
| T1 → frozen parent × q (243) | 0.17027 | 0.16904 | 1.17 |
| T1 → frozen parent × q (1007) | 0.16999 | 0.16904 | 1.93 |
| T1 → lifted parent thawed (class term) × q (315) | 0.05950 | 0.05853 | 2.25 |
| T1 → lift → all trained | 0.05895 | 0.05853 | 1.93 |
| T1 → lift → only E, Wout, bout | 0.12194 | 0.05853 | 1.05 |

Counting only the updates needed to come within 1e-2 of the fine floor (T1 charged the
updates it needed to come within 1e-2 of its own): scratch 64 × 1007 = 64,448; the lineage
T1 → lift 137 × 939 + 75 × 1007 = 218,007. **The staged lineage costs 3.4 times scratch.**
Per seed: scratch 64, 68, 55, 61, 73; T1 168, 137, 126, 181, 112; lifted 75, 85, 99, 75, 73.

### What the numbers say

- **Frozen parent.** Training only q works as designed: the total is T1's loss plus q's, and
  it lands on its own floor (0.170 against 0.16904). That floor is 2.89 times the fine
  floor. The fine floor is unreachable with the parent frozen; this is section 1's gap, not a
  training failure.
- **Parent reading the fine history.** The lifted parent, trained on the class term alone,
  drops from 0.1517 to 0.0412 in about 36 updates, and the lineage then reaches the fine
  floor. Two things change. The lineage now trains a full-size model after T1, so its work
  is about twice scratch at equal budgets. And the consistency condition breaks: after
  training, max |Σ_b T2(v, b | h) − T1(v | πh)| = 0.747. What survives is the averaged
  statement, T1(v | c) ≈ the mean of Σ_b T2(v, b | h) over the fine histories h with πh = c
  (worst deviation 0.011). On a world that is not lumpable, "summing over b returns the
  parent" can only hold in that averaged sense.
- **The lift gives no head start here.** Born from T1, the full model needs 75 [73–99]
  updates to come within 1e-2 of the fine floor; from a random start, 64 [55–73]. The
  parent's cost (137 updates on the harder coarse text) comes on top.
- **Re-embedding is not enough.** With the block frozen and only E, Wout, bout trained
  (119 parameters), the loss stalls at 0.122. A block trained on coarse symbols does not
  carry the fine distinctions the class needs.

## What failed, and limits

- The hope "the coarse world is easier, so the parent is cheap" is false for this pair of
  worlds: the coarse text has the higher floor and takes T1 twice as many updates.
- The small q networks (243 and 315 parameters) reach 1e-2 but not reliably 1e-3 in 200
  updates; only the full-size q does.
- One recipe throughout (the specimen's 200-update schedule). "Updates to ε" depends on the
  schedule; a shorter schedule for the warm-started phases was not tried.
- Work is the proxy parameters × updates, not wall-clock or flops.
- Five seeds; the scratch and lifted ranges touch at 73.
- Not done: the synthetic world with a dial (dropped on instruction), so the question
  "when does staging win, tie or lose?" is answered here for one world only, and that world
  is far from lumpable.

## Verdict

**Needs more**: the lift and the split loss are exact to rounding, but on the specimen's text staging loses (3.4 times the work of scratch, and a frozen parent stops at 2.89 times the fine floor) because vowel/consonant/space hides 0.11 nats of what the next class depends on; the untested case is a world whose coarse-graining is lumpable.

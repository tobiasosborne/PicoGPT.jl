# grow/doors — a world refined by adding a hop

**Verdict: needs more.** The exact part holds (birth equality to the last bit, a nonzero gate
gradient). Children of a one-pass parent learn the two-hop world more often and faster than a
two-pass model from scratch (7 of 18 runs reach 99 % against 1 of 9), but outcomes are bimodal
across seeds and no parent hands over its answer in the form of a query: opening the gate
without training gives chance or worse.

24 November 2026, Julia 1.13, PicoGPT's own operations (`block`, `layernorm`, `affine`,
`crossentropy`). About 30 minutes of work, nine seeds, nothing tuned; read accordingly.

## Setup

- Symbols: 4 names, 4 keys, 4 doors, a blank. A fact `a → b` is **one position** whose column is
  `Ek[a] + Ev[b]` (two embedding tables); the query is `Ek[name] + Ev[blank]`, placed last, and
  the answer is read there. No position code: the facts are a set, in random order. Every
  example draws new random bijections, so nothing can be memorised.
- **Coarse world**: 4 facts `name → door`, query a name, answer its door (one lookup).
  **Fine world**: 8 facts, `name → key` and `key → door`, query a name, answer its door (two
  lookups). **Hop world** (a second parent world, my addition): the fine context, but a single
  lookup, either `name ↦ key` or `key ↦ door`.
- Chance on the fine world is 1/4 (loss ln 4 = 1.386) once the model knows the answer is a
  door; perfect is 1 (loss 0), since the world is deterministic.
- Model: one block, D = 24, **one head**, F = 48, 6,195 parameters, applied k times with shared
  weights. Child: X₂ = X₁ + g·(block(X₁) − X₁), gate g = 0 at birth; scratch two-pass model
  starts at g = 1 (g trainable in both). AdamW, η = 3·10⁻³, batch 64, clip 1, test on 1,000
  fixed examples every 25 steps, stop at 99.5 %.
- Work = block applications × examples = steps × 64 × k.

## Numbers

Steps to reach a test accuracy on the fine world (`./summary.sh`, from `curves*.tsv`); "-" means
not reached within the cap. Seed 1 was capped at 2,500 steps, seeds 2–3 at 8,000, seeds 4–9 at
4,000.

```
seed arm                         →40% →45% →50% →90% →99%  steps  final   best
1 scratch-2pass                     925   1025   1400      -      -   2500  0.546  0.548
1 scratch-1pass                       -      -      -      -      -   2500  0.270  0.315
1 child-of-coarse                   250    275    400      -      -   2500  0.497  0.526
1 child-of-coarse-gate-only           -      -      -      -      -    300  0.240  0.273
1 child-of-hops                    1650   2150      -      -      -   2500  0.452  0.476
1 child-of-hops-gate-only             -      -      -      -      -    300  0.000  0.000
2 scratch-2pass                     500    625   1700      -      -   8000  0.604  0.623
2 child-of-coarse                   375    950   1675      -      -   8000  0.770  0.790
2 child-of-hops                     275    300    325    500    550    550  0.999  0.999
3 scratch-2pass                    1375   2025   5050      -      -   8000  0.504  0.544
3 child-of-coarse                   150    150    175    500    600    625  1.000  1.000
3 child-of-hops                     175    250    425      -      -   8000  0.763  0.774
4 scratch-2pass                       -      -      -      -      -   4000  0.304  0.348
4 child-of-coarse                   475    700   2275      -      -   4000  0.479  0.512
4 child-of-hops                     225    225    275   1000   1525   1525  0.995  0.995
5 scratch-2pass                       -      -      -      -      -   4000  0.340  0.356
5 child-of-coarse                     -      -      -      -      -   4000  0.363  0.382
5 child-of-hops                     100    250    450      -      -   4000  0.459  0.526
6 scratch-2pass                     975   1300   1650   3325   3925   4000  0.994  0.994
6 child-of-coarse                    50     75    100    225    325    350  1.000  1.000
6 child-of-hops                     875    900    950      -      -   4000  0.490  0.535
7 scratch-2pass                    2625   3700   3900      -      -   4000  0.512  0.521
7 child-of-coarse                   125    150    175    525    675    675  0.995  0.995
7 child-of-hops                     900   1025   1150   3550   3725   3725  0.997  0.997
8 scratch-2pass                     650   1050   1350      -      -   4000  0.715  0.742
8 child-of-coarse                   250   1500   1550      -      -   4000  0.506  0.517
8 child-of-hops                     400    475    500    825    875    875  1.000  1.000
9 scratch-2pass                    1025   1850      -      -      -   4000  0.477  0.497
9 child-of-coarse                   175    200    325      -      -   4000  0.516  0.538
9 child-of-hops                    1400   2725   3550      -      -   4000  0.558  0.558
```

Reaching 99 % on the fine world within the cap:

| Arm | Runs reaching 99 % | Steps of the fine stage when reached |
|---|---|---|
| scratch, two passes | 1 of 9 (seed 6) | 3,925 |
| child of the coarse parent | 3 of 9 (seeds 3, 6, 7) | 600, 325, 675 |
| child of the hop parent | 4 of 9 (seeds 2, 4, 7, 8) | 550, 1,525, 3,725, 875 |
| scratch, one pass | 0 of 1 (seed 1) | stays at chance, 27.0 % |

The caps differ between seeds (2,500, 8,000, 4,000), so these are counts, not rates to compare
finely. Parents need 50–2,550 one-pass steps when they converge; four did not reach 99.5 % within
their cap (coarse, seed 5: 88.5 %; hop, seeds 1, 6, 9: 89.3 %, 94.7 %, 87.3 %) and none of
their children succeeded. The table below lists every parent.

What this says:

- **The only seed where both sides reached 99 %** is seed 6: scratch 3,925 steps × 64 × 2 =
  502,400 block applications; child of the coarse parent 325 × 64 × 2 + 50 × 64 (parent) =
  44,800. That is 11 times less work, on one seed.
- Elsewhere scratch sits on a plateau (30–72 % at the cap), and so do the children that fail
  (36–77 %). Everything is bimodal: a run either breaks through within a few hundred steps or
  stalls. Children break through more often (7 of 18 against 1 of 9), which with these counts
  is suggestive and no more (Fisher's exact test was not run).
- Early on the child of the coarse parent is usually ahead even when it stalls later: 45 %
  reached after 75–1,500 steps, median about 240 (8 of 9 runs), against 625–3,700 for scratch (7 of 9 runs).
- With one head, one pass cannot do the fine world (chance after 2,500 steps), so the second
  pass is doing real work in every run above chance.

## The crux: does the first lookup come out as a query?

**No, not from either parent.**

- Opening the gate fully on the trained parent, with no further training (g = 1, two passes):
  fine accuracy 20–27 % for the coarse parent (chance) and 0–12 % for the hop parent (below
  chance, because it answers with a key: its pass-1 readout on fine queries names a key
  100 % of the time, seed 1). Nine seeds each.
- Training the gate alone for 300 steps (all other weights frozen): 24.0 % (coarse parent),
  0.0 % (hop parent), seed 1.
- So the hop parent has both lookups and still cannot chain them: the key arrives in the form
  the readout wants, not in the form the query matrix wants. The coarse parent has never seen a
  key at all (its key embeddings are at their random start).
- Yet children do learn, sometimes fast. What they inherit is therefore the lookup machinery
  (matching a query to a fact and copying its value), not a ready-made interface; the interface
  is learned after birth, and whether it is found is a matter of luck here. The hop parent,
  which already has both lookups, does not do clearly better than the coarse parent (4 against
  3 of 9), so having the two lookups is not what is missing: the form of the intermediate
  result is.
- The gate often ends **negative** (all three successful children of the coarse parent: −0.64,
  −0.73, −0.78; two of the four successful children of the hop parent), so the second pass is
  used as X₁ − |g|·(update), which the scratch model starting at g = 1 never arrives at (its
  gates end at 0.9–2.6). Failed children also have negative gates, so the sign does not
  explain success. Not analysed further.
- A coarse world that *demands* the query form was not built. The natural candidate: tie the
  readout to the query embedding (answer scored by Ek), or a parent world whose answers must be
  fed back as queries. This is the open design question, and these numbers say the form does
  not appear unasked.

## The gate gradient at birth

Not zero. On 1,000 fine examples, central differences (h = 10⁻⁵) against the pullback:

```
seed 1 child-of-coarse  parent steps   400 acc 1.000 | birth max|dZ| 0.0 | dl/dg FD +1.223347 pullback +1.223347 | zero-shot g=1 0.257 | final acc 0.497 gate -0.623
seed 1 child-of-hops    parent steps  2500 acc 0.893 | birth max|dZ| 0.0 | dl/dg FD +22.552606 pullback +22.552607 | zero-shot g=1 0.072 | final acc 0.452 gate +0.671
seed 2 child-of-coarse  parent steps   500 acc 1.000 | birth max|dZ| 0.0 | dl/dg FD +0.133368 pullback +0.133368 | zero-shot g=1 0.246 | final acc 0.770 gate -0.743
seed 2 child-of-hops    parent steps   450 acc 0.996 | birth max|dZ| 0.0 | dl/dg FD +1.355438 pullback +1.355438 | zero-shot g=1 0.041 | final acc 0.999 gate -0.497
seed 3 child-of-coarse  parent steps    75 acc 1.000 | birth max|dZ| 0.0 | dl/dg FD -0.040766 pullback -0.040766 | zero-shot g=1 0.225 | final acc 1.000 gate -0.639
seed 3 child-of-hops    parent steps   400 acc 1.000 | birth max|dZ| 0.0 | dl/dg FD -3.605023 pullback -3.605023 | zero-shot g=1 0.092 | final acc 0.763 gate +1.138
seed 4 child-of-coarse  parent steps  1275 acc 1.000 | birth max|dZ| 0.0 | dl/dg FD +1.866618 pullback +1.866618 | zero-shot g=1 0.232 | final acc 0.479 gate -0.744
seed 4 child-of-hops    parent steps   975 acc 1.000 | birth max|dZ| 0.0 | dl/dg FD -1.300212 pullback -1.300212 | zero-shot g=1 0.081 | final acc 0.995 gate +1.127
seed 5 child-of-coarse  parent steps  4000 acc 0.885 | birth max|dZ| 0.0 | dl/dg FD +0.298411 pullback +0.298411 | zero-shot g=1 0.204 | final acc 0.363 gate -0.288
seed 5 child-of-hops    parent steps  2250 acc 1.000 | birth max|dZ| 0.0 | dl/dg FD +7.324775 pullback +7.324775 | zero-shot g=1 0.001 | final acc 0.459 gate -0.681
seed 6 child-of-coarse  parent steps    50 acc 1.000 | birth max|dZ| 0.0 | dl/dg FD +0.109097 pullback +0.109097 | zero-shot g=1 0.267 | final acc 1.000 gate -0.731
seed 6 child-of-hops    parent steps  4000 acc 0.947 | birth max|dZ| 0.0 | dl/dg FD -15.875925 pullback -15.875924 | zero-shot g=1 0.117 | final acc 0.490 gate +0.083
seed 7 child-of-coarse  parent steps   100 acc 1.000 | birth max|dZ| 0.0 | dl/dg FD +0.357377 pullback +0.357377 | zero-shot g=1 0.211 | final acc 0.995 gate -0.777
seed 7 child-of-hops    parent steps  2525 acc 1.000 | birth max|dZ| 0.0 | dl/dg FD +1.562873 pullback +1.562873 | zero-shot g=1 0.011 | final acc 0.997 gate +2.122
seed 8 child-of-coarse  parent steps  2550 acc 1.000 | birth max|dZ| 0.0 | dl/dg FD -1.481371 pullback -1.481371 | zero-shot g=1 0.236 | final acc 0.506 gate +0.874
seed 8 child-of-hops    parent steps   225 acc 1.000 | birth max|dZ| 0.0 | dl/dg FD +2.035402 pullback +2.035402 | zero-shot g=1 0.000 | final acc 1.000 gate -0.828
seed 9 child-of-coarse  parent steps   200 acc 1.000 | birth max|dZ| 0.0 | dl/dg FD +1.014427 pullback +1.014427 | zero-shot g=1 0.212 | final acc 0.516 gate -0.825
seed 9 child-of-hops    parent steps  4000 acc 0.873 | birth max|dZ| 0.0 | dl/dg FD +0.598891 pullback +0.598891 | zero-shot g=1 0.042 | final acc 0.558 gate -0.061
```

At g = 0 the second pass contributes nothing to the gradient of the other weights (it enters as
g·X̄), so at birth only the gate, and the weights through pass 1, move.

## Verified, and how

- Gradient of the whole looped model (all of θ and the gate; k = 1 and k = 2 at g = 0, 0.37, 1)
  against central differences: relative error ≤ 3.1·10⁻⁹ (`check.jl`, D = 8, 1,059 parameters).
- Birth: max |Z_child − Z_parent| over 1,000 examples of the parent's world and 1,000 of the
  fine world is **0.0** exactly, in every run (column `birth_maxdiff`).
- Gate gradient: finite differences agree with the pullback to 6–7 digits, in every run.

## Not reached, and caveats

- Outcomes are bimodal and the seed count is small; no ratio of work can be quoted.
- Nobody was tuned: one learning rate, one width. Scratch may only need a different
  initialisation, a learning-rate schedule or more heads. A first run with D = 32, **4 heads**
  (not saved; rerun with `DOORS_D=32 DOORS_H=4 DOORS_F=64`) had scratch reach 99 % in 2,600
  steps, and there *one pass alone reached 95.8 %*: with four heads the world is small enough
  to solve without chaining, which is why everything above uses one head.
- Attention is causal (PicoGPT's), so a `key → door` fact can itself look up an earlier
  `name → key` fact in pass 1. That shortcut works only when the order is favourable, and may be
  the 50–60 % plateau. Not checked.
- No look inside the successful children (which fact the query attends to in each pass).
- The hop world is not a coarse-graining of the fine world; the coarse world is.

## Rerun

    OPENBLAS_NUM_THREADS=1 julia --project -t2 grow/doors/check.jl            # seconds
    OPENBLAS_NUM_THREADS=1 julia --project -t2 grow/doors/run.jl 1 2500       # ~4 min, all arms
    DOORS_TAG=_long2 DOORS_LEAN=1 julia --project -t1 grow/doors/run.jl 2 8000
    DOORS_TAG=_s4 DOORS_LEAN=1 julia --project -t1 grow/doors/run.jl 4:6 4000   # likewise 7:9, tag _s7
    grow/doors/fill.sh        # the tables above (also in tables.txt)

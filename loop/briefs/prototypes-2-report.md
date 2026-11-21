# Prototypes round 2 report

**The 17-parameter predecessor-register model meets the specimen criterion for both tasks.**
The block is a fixed predecessor head and one learned two-layer ReLU MLP, whose scalar output
is clipped to `[-1,1]` before it becomes the next state. Prefix parity supplies the original
input bit as the MLP's other input. Rule 60 supplies the current state. This is an architectural
simplification of the PicoGPT block. The head's routing is fixed.

| Task and objective | Qualifying seeds | Every training length, 2–12 | Length 48 through pass 96 | Timed updates per seed |
|---|---:|---:|---:|---:|
| Prefix parity, final loss | 5/5 | 100% at `K = n` | 100% for passes 48–96 | 0.80–0.94 s |
| Rule 60, deep loss at random horizons 1–12 | 5/5 | 100% at every pass | 100% at every pass | 0.78–0.96 s |

Prefix parity has a linear front beginning at position two after pass one. Its register
changes by exactly zero from pass 48 on at length 48. Rule 60 matched 96 complete 193-position
impulse rows in each seed. Its register keeps changing as the target evolves. Both fitted
blocks wrote the correct bit exactly for all four possible pairs of binary inputs.
[17-parameter prefix parity](../artifacts/round2-prefix-compact-tied.json)
[17-parameter Rule 60](../artifacts/round2-rule60-compact-tied.json)

A 13-parameter version met the threshold in four of five seeds on each task. The
nine-parameter prefix pilot had one passing seed. Use 17 parameters for the live specimen
for its observed 5/5 reliability on both tasks.
[13-parameter prefix parity](../artifacts/round2-prefix-three-tied.json)
[13-parameter Rule 60](../artifacts/round2-rule60-three-tied.json)
[Nine-parameter prefix screen](../artifacts/round2-screen-register-two-final.json)

In the 33-parameter ablations, removing clipping from Rule 60 kept trained-depth accuracy
above 99% in every seed and brought pass-48 accuracy down to 64.45–89.70%. Deep supervision
raised Rule 60 from four passing seeds to five. Prefix parity passed in all five seeds
with final-only, window, and cone loss, and the unclipped prefix ablation retained 5/5.
[33-parameter Rule 60 final](../artifacts/round2-rule60-final.json)
[Rule 60 unbounded](../artifacts/round2-rule60-unbounded.json)

The strongest stock-style block used a fixed lag-one bias and refresh. With 650 parameters
it scored 80.21% at length 12 and 58.63% at pass 48, with its front stalled at position three.
Its head chose the previous token in 97.87% of sampled rows at pass 48.
[Adjacent block diagnostics](../artifacts/round2-adjacent-five.json)

For prefix parity, the 17- and 205-parameter one-pass controls both scored 58.46% at length 12
and 52.10% at length 48. A single pass can read two bits. The 204-parameter untied model got
54.09–59.44% at length 48 by layer 12, where its stack ends. For Rule 60, the one-pass
controls scored 52.77–56.43% and 54.47–56.43% on length 48 when averaging over requested
passes 1–12. The untied 12-layer model had 49.84–50.07% at pass 12 on the same length.
[Results, controls, and reproduction](../notes/prototypes.md)

Use prefix parity to show what the extra passes compute and when the state settles.
Use Rule 60 to show how the same cell can evolve autonomously.

The package and loop suites passed. All new Float64 pullback checks were below `8e-10`.
Timings cover optimizer updates. Julia startup, compilation, and evaluation add wall time.

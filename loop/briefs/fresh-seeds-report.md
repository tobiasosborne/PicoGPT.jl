# Fresh seeds for the frozen 270-parameter specimen

At lag bias +4, 25 of 30 fresh seeds met the complete target. The two-sided 95% Clopper–Pearson interval is 65.3%–94.4%.

| Lag bias | Qualified | Within 1% of floor | Exact on all four rollouts | 95% Clopper–Pearson interval | Updates, median (range) |
|---:|---:|---:|---:|---:|---:|
| +4 | 25/30 (83.3%) | 27/30 | 26/30 | 65.3%–94.4% | 10.17 s (7.17–12.33 s) |
| 0 | 20/30 (66.7%) | 20/30 | 27/30 | 47.2%–82.7% | 9.92 s (7.98–12.78 s) |
| +2 | 21/30 (70.0%) | 22/30 | 25/30 | 50.6%–85.3% | 10.50 s (8.14–12.21 s) |

Qualification requires exhaustive mixed cross-entropy within 1% of the exact 0.161350952-nat floor and 100% accuracy at lengths 32 and 64, each at `K=T` and `K=2T`.
Every arm used initialization keys 20001–20030 and minibatch keys 21001–21030. Earlier screens had used 1001–1005 as minibatch keys.
The 256 held-out words per length came from Xoshiro(642901) and were shared by all arms.

At +4, two seeds recovered from errors at length 64 and pass 64 by pass 128. One settled to a wrong fixed point near the boundary, one showed continuing stream motion and lost its front, and one reached exact endpoints with weak predictions at first arrival.
At 0, seven seeds had exact endpoints with weak arriving predictions. Three stalled near the boundary with RMS stream motion still present at pass 128.
At +2, four seeds had exact endpoints with weak arriving predictions. One lost a correct front between passes 64 and 128, two approached wrong fixed points, and two showed delayed or stalled transport.
At zero bias, the per-seed mean predecessor weights at length 64 and pass 128 spanned 0.644–0.942 in qualifiers and 0.437–0.868 in failures.
The notes include every seed’s timing, failure diagnostics, and the zero-bias attention means, minima, and maxima. [Fresh seeds in the window notes](../notes/window.md#fresh-seeds)

Commit `9d526ef` froze the recipe before training. The first invocation stopped at an ambiguous optimizer binding, which `8190e19` resolved by narrowing the SHA import before any completed update. [Frozen configuration](../scripts/window_fresh.toml) · [Runner](../scripts/window_fresh.jl)

[+4 artifact](../artifacts/window-fresh-bias-4.json) · [0 artifact](../artifacts/window-fresh-bias-0.json) · [+2 artifact](../artifacts/window-fresh-bias-2.json)

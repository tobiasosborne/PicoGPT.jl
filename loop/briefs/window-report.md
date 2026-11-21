# Two-token window experiment

The masked, post-injection PicoGPT loop approached the exact loss floor on length 8, and every tested model fell short of the five-seed and long-rollout criteria. Exhaustive enumeration of all 256 words confirmed that pass one sees the predecessor. The deep floor at length and depth 8 is `21 × log(2) / 64 = 0.227438919` nats.

| Near-floor run | Parameters | Deep CE and gap | Accuracy at T8 | T32 at 32 / 64 passes | T64 at 64 / 128 passes |
|---|---:|---|---:|---:|---:|
| 600 steps, seed 2 | 582 | 0.229095, +0.001656 (0.73%) | 100% | 65.09 / 64.90% | 57.28 / 57.32% |
| 1,600 steps, seed 1 | 582 | 0.227447, +0.000008433 (0.0037%) | 100% | 71.85 / 75.23% | 62.42 / 63.73% |

Each recipe came within 1% of the floor in one of five seeds. The 340- and 162-parameter shapes missed it in all five seeds of the width screen. The best checkpoint was exactly correct inside its cone at length 8 and its largest mean deviation from 0.5 outside the cone was 0.00460. It retained exact length-8 predictions through pass 16, with continued movement in the centered state. On longer words its 99%-correct front reached only position 15. The terminal covector obeyed the same locality bound while most of its norm stayed at the seeded site. The best checkpoint's repeat took 21.81 seconds of timed updates.

Both narrow and parameter-matched wide one-pass controls reached the 62.5% ceiling in some seeds and approached the 0.519860385-nat floor. The untied deep control uses 4,306 parameters and shares the tied deep floor. Its best loss was 0.260761 nats. The best tied run with final-pass loss reached 0.413276 nats against a zero floor.

[Full notes and reproduction commands](../notes/window.md) · [Five-seed width screen](../artifacts/window-screen.json) · [Control runs](../artifacts/window-controls.json) · [Best checkpoint diagnostics](../artifacts/window-probe-tuned.json)

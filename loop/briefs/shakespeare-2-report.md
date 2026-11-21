# Tiny Shakespeare follow-up

Both 215,233-parameter runs completed 5,000 Float32 updates with 12 Julia threads and one seed each, 1234. Both start with token embeddings and sinusoidal positions at every column. The 29 Float64 checks passed before training.

| Run | Training passes | Lowest held-out loss | Pass 16 loss | Training wall time |
| --- | --- | ---: | ---: | ---: |
| A · Full causal, full-input start | Uniform 3–5 | 1.7102 at pass 3 | 1.7820 | 89.44 min |
| B · Window of two | Uniform 1–8 | 1.6256 at pass 6 | 1.6387 | 71.74 min |

**A.** Full-input initialization improves the original loop by 0.0698 nats at pass 4. Its pass-4 result of 1.7145 is 0.0604 above stock tied and 0.0162 above one pass. At A's best pass the gaps shrink to +0.0561 and +0.0120, respectively.

**B.** The run averaged 4.4998 block applications per update. Its best pass beats stock tied by 0.0286 nats and trails Book 1 by 0.0943. It lags the interpolated Kneser–Ney reference at every pass. The gap is +0.0187 nats at pass 1, +0.0650 at pass 2 and +0.0905–0.1022 at passes 3–16. Each n-gram model uses one shared discount tuned on the last 5% of the training split. Its counts are refit on all training text. Pass `k` matches order `k+2`.

In three held-out windows, both tested positions beyond the cone gave exactly unchanged logits through pass 16. All 48 changes at the cone's inside edge affected the logits. On “Good morrow, neighbour,” A and B settled their argmax by pass 8 at 81 and 74 of 83 positions. Their centred state steps at pass 16 were 0.289 and 1.906.

The training wall timer records awake time. A crossed a 4 h 54 min suspend and resumed in the same process. The host reported Power Saver at both ends of A. B began in Power Saver and ended in Performance. The timings reflect those system conditions.

[Full notes and samples](../notes/shakespeare.md) · [Run A data](../artifacts/shakespeare-fullstart.json) · [Run B data](../artifacts/shakespeare-window.json) · [N-gram ladder](../artifacts/shakespeare-ngram.json)

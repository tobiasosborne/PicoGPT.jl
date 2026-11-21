# Tiny Shakespeare trial

All three Float32 runs finished 5,000 updates of 32 sequences with 12 Julia threads. Each model has 215,233 parameters. The data, split and 256 held-out windows match Book 1, whose four untied blocks reached 1.5313 nats per character with 810,049 parameters.

| Model | Best pass among 1–16 | Held-out loss | Pass 16 loss | Wall time under Power Saver |
| --- | ---: | ---: | ---: | ---: |
| One block, one pass | 1 | 1.6983 | 6.8989 | 24.91 min |
| Looped, full-width specimen recipe | 4 | 1.7843 | 1.8914 | 83.07 min |
| Stock tied | 4 | 1.6541 | 5.3124 | 75.14 min |

The specimen-style loop trained on a uniform 3–5 passes and averaged 4.0022 block applications per batch. On the 83-character held-out passage, its centred state step fell to 0.83 by pass 16, compared with 34.02 for stock tied. It made its final measured argmax change by pass 8 at 78 positions, compared with 26 for stock tied. Its loss rose at every pass after 4 even as its state steps shrank. A Shakespeare chapter can show the same tied cell learning character language and retaining useful predictions when rolled beyond training. The quality result favors stock tied at its trained depth, and every model trails Book 1. There was one seed per model.

The Float64 checks passed before training, with relative errors below 1.5 × 10⁻⁹ for both stream maps. The specimen tests passed, and the three training histories remained finite. The wall times were measured under Power Saver. Book 1 reports 36.2 minutes of update time, with no power profile on record. The optional width-two run and n-gram comparison were skipped.

[Full notes](../notes/shakespeare.md) · [One-pass data](../artifacts/shakespeare-one.json) · [Looped data](../artifacts/shakespeare-looped.json) · [Stock tied data](../artifacts/shakespeare-stock.json)

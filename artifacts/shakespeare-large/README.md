# The large Shakespeare run (an exhibit of overfitting)

This directory is *not* the book's Shakespeare model; that model's data is in `../shakespeare/`. It
holds the loss history and samples of a larger model trained on the same Tiny-Shakespeare text,
on "monolith" (a Xeon E5-2699 v3, 18 cores), with the canonical AdamW recipe (BRIEF §13):

- model: D = 256, H = 4, L = 6, T = 256, F = 1024, 4,772,417 parameters, Float32;
- training: 12,000 steps of 32 × 256 tokens, 98.3 M tokens, about 90 passes over the corpus;
- run length: 3 h 20 min, at ≈ 998 ms/step and 8.2 k tokens/s;
- outcome:
  - best validation loss 1.560 at step 2,500;
  - final validation loss 4.371 against training loss 0.152.

It memorised its training text. Exported by `scripts/export_book_data.jl` from the final (step-12,000)
checkpoint; branch `monolith-run`, commit db7b949.

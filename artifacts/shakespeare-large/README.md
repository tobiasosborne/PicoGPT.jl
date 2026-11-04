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

## Exact-match exhibit for chapter 9

`memorised.json` annotates both `samples.json` and `samples-extra.json`. The latter
contains extra draws preserved from the run report, rather than the regular exporter.
Regenerate with `node scripts/export_memorised.mjs`; verify byte-for-byte with
`node scripts/export_memorised.mjs --check` (both require the gitignored corpus at
`artifacts/data/tinyshakespeare.txt`). The book and CI read the generated JSON only.

A match is an inclusion-maximal substring of a sample, at least 20 characters long,
which occurs exactly in the training split: the first `round(0.9 * length)` characters,
as in `scripts/common.jl`. Case, punctuation, spaces and newlines are significant.
All maximal matches are retained, including overlaps; underlining shows their union.
Offsets are zero-based, half-open character ranges; each match also records one source
offset. Corpus and sample-file SHA-256 hashes document the inputs. The script checks
that the corpus is ASCII, so character indices and byte offsets agree.

The token total corresponds to about 88 whole-corpus equivalents (about 90 above),
or 98 passes over the 90% training prefix; no held-out tokens were used for updates.

# Does the cheap block replacement cost language-model loss?

**Verdict: needs more.** On Shakespeare characters the cheap replacement never beats a plain
recency window with the same number of sources. In a world whose evidence is spread over the
whole context it does beat it, and blocks chosen by key spread are then almost free: 7.9 sources
instead of 24.5 for +0.006 nats, where a window of the same size costs +0.050.

Follow-up to `grow/notes/blocking.md` (branch `grow/notes-blocking`), which proved that a block
of attention sources is replaced exactly by the pair (Z_B, N_B) and measured head-read errors
only. Written 24 November 2026 under a 40-minute limit on a heavily loaded laptop.

## What was done

Everything is in `run.jl` (about 2.5 minutes per world on two threads):

    OPENBLAS_NUM_THREADS=1 julia -t2 --project grow/blocking-loss/run.jl              # results.txt
    WORLD=topic OPENBLAS_NUM_THREADS=1 julia -t2 --project grow/blocking-loss/run.jl  # results-topic.txt

**The cheap replacement as score averaging.** One source with the block's mean key, a bias
log|B| and the mean value gets weight ∝ |B|·exp(mean score) and contributes that weight times the
mean value. This equals ordinary attention in which every member of the block is given the
block's mean score. So blocked attention is A = softmax(P·S + mask), with P the averaging
projection over the groups of each query row. P is symmetric, so the pullback is S̄ = P·S̄′. The
script uses this form in every layer and head.

**Model.** PicoGPT with D=32, H=4, L=2, T=32, F=64 (about 21,000 parameters), trained for 1,500
steps of 16 windows, about one minute.

**Two worlds.**
- *Shakespeare*: the first 60,000 bytes of Tiny Shakespeare (`corpus.txt`), 54,000 for training,
  6,000 held out.
- *Topic*: a synthetic stream in segments of 96 tokens, each segment drawn independently from one
  of 8 hidden distributions over 16 tokens. The best predictor pools all it has seen.

**Schemes.** b is the block size and r the number of most recent positions left untouched.
- *blk-contig*: blocks on a fixed grid of size b. A block is merged for query i only if it lies
  entirely among the sources older than the r most recent. Every merged block is then complete,
  so no partial block arises and the causal mask holds.
- *blk-keys*, *blk-key+val*: the same number of sources per query, but the sources older than
  the r most recent are grouped by Ward agglomeration (smallest spread) of the projected keys,
  or of keys and values together. Row i clusters only sources j ≤ i − r.
- *drop-same*: drop exactly the positions blk-contig merges.
- *drop-match*: keep the n most recent positions, n being the number of sources blk-contig
  keeps for that query. This is the fair opponent: equal cost.

All losses are held-out, in nats per token, averaged over positions 17 to 32 of 597 windows
(stride 10). Full attention has 24.5 sources per query there on average.

## Numbers

### Shakespeare (full attention 1.8932)

| b | r | sources | blk-contig | blk-keys | blk-key+val | drop-same | drop-match | blk-contig − drop-match |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 13.50 | 1.9976 | 1.9835 | 2.0165 | 2.2357 | 1.9008 | +0.097 ± 0.007 |
| 4 | 2 | 8.75 | 1.9992 | 2.0807 | 2.1050 | 2.1126 | 1.9161 | +0.083 ± 0.007 |
| 8 | 2 | 7.88 | 1.9728 | 2.1210 | 2.1328 | 2.0259 | 1.9340 | +0.039 ± 0.005 |
| 2 | 4 | 14.50 | 1.9126 | 1.9102 | 1.9144 | 1.9994 | 1.8993 | +0.013 ± 0.003 |
| 4 | 4 | 10.25 | 1.9219 | 1.9260 | 1.9273 | 1.9692 | 1.9089 | +0.013 ± 0.003 |
| 8 | 4 | 9.62 | 1.9157 | 1.9316 | 1.9320 | 1.9390 | 1.9136 | +0.002 ± 0.002 |

Plain window of the m most recent sources in every layer: m=4 2.0173, m=8 1.9203, m=12 1.9020,
m=16 1.8971, m=24 1.8936.

- Blocking beats dropping the same positions in every row (by 0.02 to 0.24).
- Blocking loses to a recency window of equal size in every row; at b=8, r=4 the two are level.
- This model hardly uses distant context: a window of 12 costs 0.009 nats. There is almost
  nothing for blocking to save, and averaging the scores disturbs heads that pick single
  positions.
- Blocks chosen by key spread are no better than contiguous ones here, and clearly worse at
  r=2 with b=4 or 8. Not explained in the time available.

### Topic world (full attention 1.9166)

| b | r | sources | blk-contig | blk-keys | blk-key+val | drop-same | drop-match | blk-contig − drop-match |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 13.50 | 1.9285 | 1.9167 | 1.9166 | 2.2149 | 1.9212 | +0.007 ± 0.002 |
| 4 | 2 | 8.75 | 1.9393 | 1.9195 | 1.9190 | 2.1356 | 1.9447 | −0.006 ± 0.003 |
| 8 | 2 | 7.88 | 1.9412 | 1.9225 | 1.9218 | 2.0492 | 1.9671 | −0.026 ± 0.003 |
| 2 | 4 | 14.50 | 1.9259 | 1.9168 | 1.9169 | 2.0518 | 1.9195 | +0.006 ± 0.002 |
| 4 | 4 | 10.25 | 1.9318 | 1.9187 | 1.9192 | 2.0134 | 1.9309 | +0.001 ± 0.002 |
| 8 | 4 | 9.62 | 1.9321 | 1.9207 | 1.9207 | 1.9777 | 1.9404 | −0.008 ± 0.002 |

Plain window: m=4 2.0790, m=8 1.9458, m=12 1.9212, m=16 1.9167, m=24 1.9164.

- Contiguous blocks beat the equal-size window once the compression is strong (b=8), and lose
  slightly at b=2.
- Blocks chosen by key spread beat the equal-size window in every row. At b=8, r=2 they cost
  +0.006 nats with 7.9 sources; the window costs +0.050 and contiguous blocks +0.025.
- This is the case the derivation predicts. There are only 16 tokens, so many sources share
  nearly the same projected key, and for coinciding keys the cheap replacement is exact.
- The trained model itself gains nothing beyond 16 sources (m=16 equals full attention), so
  even here the room is small.

### Training with the blocking in place (250 further steps, η = 10⁻³)

| world | scheme | b | r | sources | before | after | full attention, same extra steps |
|---|---|---|---|---:|---:|---:|---:|
| Shakespeare | blk-contig | 4 | 4 | 10.25 | 1.9219 | 1.8849 | 1.8753 |
| Shakespeare | drop-match | 4 | 4 | 10.25 | 1.9089 | 1.8785 | 1.8753 |
| Shakespeare | blk-contig | 8 | 2 | 7.88 | 1.9728 | 1.9219 | 1.8753 |
| Shakespeare | drop-match | 8 | 2 | 7.88 | 1.9340 | 1.8910 | 1.8753 |
| Topic | blk-contig | 4 | 4 | 10.25 | 1.9318 | 1.9220 | 1.9176 |
| Topic | drop-match | 4 | 4 | 10.25 | 1.9309 | 1.9224 | 1.9176 |
| Topic | blk-contig | 8 | 2 | 7.88 | 1.9412 | 1.9240 | 1.9176 |
| Topic | drop-match | 8 | 2 | 7.88 | 1.9671 | 1.9377 | 1.9176 |

The loss recovers partly, not fully, and the ranking does not change: on Shakespeare the window
stays ahead after training, in the topic world contiguous blocks stay ahead at b=8. The
Shakespeare model was still improving (the control gained 0.018 from the same extra steps), so
much of the "recovery" there is ordinary training. Spread-chosen blocks were not trained.

## What is verified, and how

Asserted at the start of every run of `run.jl`:
- The explicit cheap replacement (mean key, bias log|B|, mean value) equals score averaging on a
  random head: difference 3.2×10⁻¹⁶.
- With every source in its own group, the copied forward pass and gradient equal PicoGPT's
  `loss` exactly (difference 0).
- The gradient of the blocked loss agrees with central finite differences along 20 random
  directions: largest relative error 2.2×10⁻⁸.
- The spread labeler keeps exactly as many sources per query as the contiguous one.

The standard errors are paired over 597 windows that overlap (stride 10, length 32), so they are
too small by an unknown factor, perhaps up to 2. Differences below about 0.01 should not be
trusted. Everything comes from one training seed per world.

## Not reached, and open

- The saved specimen (T=8) was not used: too short to block.
- One seed, one model size, a one-minute model. The Shakespeare model is far from the 810,049
  parameter model's quality and uses little context; the comparison should be repeated on a
  model that uses 100 or more positions, where a window cannot win cheaply.
- "Sources kept" counts sources per query. Spread-chosen blocks are recomputed for every query
  row, head and layer, so they save no computation as implemented. A scheme that reuses
  clusters across queries (the merge rule in the derivation allows it) was not built.
- Why spread-chosen blocks hurt on Shakespeare at r=2 is unexplained. A guess, unchecked: the
  keys there carry position, and the heads read single recent positions that Ward merges away.
- The exact pair (Z_B, N_B) and the degree-two correction were not put through the loss; only
  the cheapest replacement was.
- Blocking the queries as well as the sources (a truly shorter sequence) was not tried.

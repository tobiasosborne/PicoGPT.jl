# Prefix parity with a two-token attention window

The two-token mask gives an exact training loss floor. In the five-seed width screen, one of fifteen random initializations came within 1% of it. That checkpoint reached 100% accuracy at length 8, then fell to 65.09% at length 32 and 57.28% at length 64. The best tuned run improved those longer scores to 71.85% and 62.42%. Every tested recipe fell short of the length and pass criteria.

## What the mask establishes

Each query reads itself and, except at position one, its predecessor. The stream starts at the usual token embedding plus the sinusoidal code. The block is applied with post-addition, `X[k+1] = B(X[k]) + X[0]`. The model uses the stock block LayerNorms and one shared readout. With these choices the only path between different positions is through attention. Pass `k` can read tokens from `max(1,t-k)` through `t`. Pass one already sees token `t-1`.

The verifier enumerates all 256 length-8 words and groups their prefix labels by the visible bits for each pass and query. Every group inside the cone had a single label. Every group outside it had equal counts of the two labels. Thus the per-pass floor is `max(T-k-1, 0) × log(2) / T`. At `T = K = 8` the deep floor is `21 × log(2) / 64 = 0.227438919` nats. The one-pass floor is `6 × log(2) / 8 = 0.519860385` and its exhaustive accuracy ceiling is 62.5%. The final-pass floor at `K = 8` is zero. The untied control has the same deep floor as the tied loop. These are bounds on loss imposed by information access. Correct deterministic predictions approach zero loss as their finite margins grow.

## Width screen

Every row uses one head, float64 BRIEF §13 initialization, and AdamW with matrix-only decay and global-norm clipping. The main screen used 600 updates, fresh batches of 32 uniform words, a peak learning rate of 0.02, a cosine finish at 0.002, and 20 warmup steps. Beyond the cone, the training estimator averages the cross-entropy of both labels. Its expectation is exactly the ordinary prefix loss because an unseen independent fair bit makes the target fair. All reported training-set losses score the ordinary hard labels over the whole enumeration. The test sets at lengths 32 and 64 each have 256 independently generated words.

Cells list seeds 1–5 in order. Loss and train accuracy are scored at length 8, and the rollout columns give accuracy at pass `K = T`.

| D, F | Parameters | Deep CE | T8 accuracy at pass 8 | T32 accuracy | T64 accuracy |
|---|---:|---|---|---|---|
| 8, 12 | 582 | .46418 / .22910 / .53249 / .43490 / .45548 | 67.19 / 100 / 62.26 / 73.05 / 70.31% | 53.87 / 65.09 / 52.69 / 53.71 / 54.31% | 51.82 / 57.28 / 51.48 / 51.56 / 51.90% |
| 6, 8 | 340 | .53461 / .54899 / .49883 / .43485 / .58735 | 62.11 / 63.87 / 65.63 / 77.34 / 65.04% | 53.48 / 52.44 / 53.99 / 56.95 / 54.39% | 50.61 / 51.83 / 51.81 / 52.87 / 52.44% |
| 4, 4 | 162 | .58737 / .58512 / .58923 / .49063 / .54346 | 63.67 / 61.72 / 62.11 / 70.51 / 61.72% | 52.89 / 53.58 / 52.81 / 54.41 / 51.94% | 51.96 / 51.82 / 51.50 / 49.58 / 51.46% |

Seed 2 at `D=8, F=12` was 0.001656 nats (0.73%) above the floor. The other four seeds of that shape and all ten narrower runs missed the 1% loss threshold. At twice the length in passes, seed 2 scored 64.90% at length 32 and 57.32% at length 64. The timed width-screen update loops took 16.41–43.20 seconds per run with concurrent jobs. The times exclude compilation, the fixed exhaustive loss checkpoints, and rollout evaluation.

## Controls

All controls used the screen schedule and five seeds. The one-pass models are evaluated at pass 1, the untied model at pass 8, and the tied final-loss model at pass 8 at length 8 and at pass `T` on the longer sets.

| Control | Parameters | Exhaustive loss floor | Observed loss range | T8 accuracy range | Timed updates |
|---|---:|---:|---:|---:|---:|
| Tied, final loss | 582 | 0 | 0.41328–0.54315 | 60.94–71.68% | 16.25–26.14 s |
| One pass | 582 | 0.519860385 | 0.519864–0.563376 | 59.38–62.50% | 3.19–4.62 s |
| Wide one pass | 4,305 | 0.519860385 | 0.519863–0.563309 | 59.38–62.50% | 11.54–14.83 s |
| Untied, eight blocks, deep loss | 4,306 | 0.227438919 | 0.26076–0.37952 | 77.83–98.24% | 23.52–31.51 s |

Three narrow one-pass seeds and four wide seeds came within 1% of the one-pass floor. Extra width left the theoretical ceiling at 62.5%. The eight untied blocks each have 532 parameters and share the same 50 embedding and readout parameters. Their best loss was 0.03332 nats above the floor. The tied final-loss runs reached at most 54.71% at length 32 and 51.98% at length 64.

## Schedule and batch tuning

The strongest run used the same 582-parameter shape with 1,600 updates and a fresh batch of 16. Seed 1 reached 0.227447351 nats, a gap of 0.000008433 nats (0.0037%). Its length-8 accuracy stayed at 100% through pass 16. At length 32, accuracy was 71.85% at pass 32 and 75.23% at pass 64. At length 64, accuracy was 62.42% at pass 64 and 63.73% at pass 128. The contiguous front with at least 99% accuracy reached position 15 at the end of both longer rollouts. The remaining four seeds missed the 1% loss threshold, with losses of 0.53084, 0.48220, 0.37292, and 0.46435. The five timed training runs for the longer recipe took 27.46–37.98 seconds. The standalone probe reproduced the seed-1 parameters with 21.81 seconds of timed updates.

The lower-rate recipe (1,200 steps, batch 24, peak rate 0.005) reached losses of 0.48355 and 0.45675 in seeds 1 and 2. Using ordinary sampled labels (800 steps, batch 24, peak rate 0.02) gave 0.50457 and 0.53838. Training against all 256 hard-labeled words (400 steps, peak rate 0.01) gave 0.57385 in seed 1. The full-batch run took 103.78 seconds of timed updates. Several settings varied at once, so those comparisons measure the recipes as a whole.

## The best checkpoint

At length 8 the tuned seed got every label inside the cone right at every trained pass. Outside the cone, accuracy was exactly 50%, and position-wise mean predicted probabilities for class 2 ranged from 0.49904 to 0.50460. The largest mean absolute deviation from 0.5 at a pass and position was 0.00460. The same checkpoint was much more confident at some unseen positions on longer inputs. At length 64 in the first eight passes, the maximum of that mean deviation outside the cone rose to 0.47263. The gap appears on unseen positions even at trained pass counts.

| Optimizer updates | 0 | 400 | 800 | 1,000 | 1,600 |
|---:|---:|---:|---:|---:|---:|
| Exhaustive deep CE | 0.77711307 | 0.44607183 | 0.24788711 | 0.22787565 | 0.22744735 |
| Gap to the exact floor | 0.54967415 | 0.21863291 | 0.02044819 | 0.00043673 | 0.00000843 |

- **Attention.** At length 8, pass 8, predecessor weights after the singleton first query ranged from 0.038 to almost 1. Half of the query/word pairs put at least 0.95 on the predecessor, and 0.50% put at least 0.95 on self. The head mixes the two sources with input- and position-dependent weights.
- **Stream state.** At length 8, decoded labels stayed fixed from pass 7 through pass 16. At pass 16 the largest absolute stream increment was 19.82, the largest absolute increment after column centering was 21.19, and the largest column-mean increment was 1.93. At length 64, pass 128, the corresponding values were 23.85, 25.33, and 2.23. The centered stream was still moving even where its predicted bits had settled.
- **Backward light cone.** The probe seeds `Z[2,16] - Z[1,16]` at pass 16 on a length-16 word and stores the recurrent stream cotangent at every pass. Direct contributions through repeated input injection accumulate separately when forming the embedding gradient. The cotangent is exactly zero beyond the allowed cone and its left edge moves left one position per backward pass until it reaches position 1. The norm peaks at the seeded position throughout. From pass 16 to pass 15, the norm there was 0.0891 and 0.0887, while the new left-edge norm at position 15 was 0.00106. In the separate digital probe, flipping input position 8 changed predictions at positions 8–11 by pass 16.

The prefix experiment missed the rollout target that gates the Rule 60 follow-up.

## Reproduction

Run from the repository root with Julia 1.13.3.

~~~sh
julia --project -O1 loop/test/window.jl
julia --project -O1 -t 3 loop/scripts/window_parity.jl screen
julia --project -O1 -t 2 loop/scripts/window_parity.jl controls
julia --project -O1 -t 2 loop/scripts/window_parity.jl tuning
julia --project -O1 -t 2 loop/scripts/window_parity.jl long-extra
julia --project -O1 loop/scripts/window_parity.jl probe
julia --project -O1 loop/scripts/window_parity.jl probe-tuned
~~~

The standalone test compares the compact forward operation and its pullback against dense additive-mask attention, checks finite differences for both implementations and for the tied and untied loops, and verifies the offset, loss, and gradient support. The data store the recipes, parameters, update losses, exhaustive training curves, and each rollout pass.

[Window implementation](../src/window.jl) · [Experiment script](../scripts/window_parity.jl) · [Tests](../test/window.jl) · [Width screen](../artifacts/window-screen.json) · [Controls](../artifacts/window-controls.json) · [Tuning](../artifacts/window-tuning.json) · [Long-schedule seeds 4–5](../artifacts/window-long-extra.json) · [Screen probe](../artifacts/window-probe.json) · [Best checkpoint probe](../artifacts/window-probe-tuned.json)

## Round 2

The smallest tested model meeting every target uses two learned gains and one scalar predecessor register. All five random initializations finished within 0.183% of the mixed training floor, and all five scored 100% on the length-32 and length-64 evaluations at both one and two times the sequence length in passes. Its fixed predecessor head, scalar state and initialization are architectural choices. The two gains determine independently whether each symbol copies or flips the predecessor's state.

### Exact floors and the position ablation

Each update draws a length uniformly from 4–12, draws a pass count uniformly from `T−1` through `2T`, and uses 12 fresh iid binary words of that length. Loss averages over all positions and over every pass from 1 through the drawn pass count. Outside the cone, the training estimator scores the fair posterior. Evaluation enumerates every word at every training length, scores ordinary hard labels, and averages with the same weights over lengths and pass counts.

At pass `k` the floor is still `max(T−k−1, 0) × ln(2) / T`. Exhaustive grouping by visible context confirms it for every relevant length, pass and position. The distribution average is exactly `(384975233676617 / 1653814216454400) × ln(2) = 0.161350951729293` nats. The one-pass model has `(89539 / 124740) × ln(2) = 0.497544535835794` nats. The untied variable-depth control shares the mixed floor.

Ordinary singleton attention averages identical values at both the boundary and the interior when all its visible tokens have identical states. With no position signal, the recurrence is equivalent to padding the word on the left with its first bit. Prefixes whose leading run of negations differs in parity can then present the same state context with opposite labels. Grouping words by this padded context gives a mixed deep-loss lower bound of 0.432345180 nats. At full receptive field, the accuracy upper bound is

```text
1 − (1/T) Σ_{r even, 2 ≤ r ≤ T} 2^(−r) [1 + (T−r)/2].
```

The subtraction counts the smaller posterior mass, from even-length leading runs of bit 1 for each distinguishable suffix. The bounds are 84.72%, 83.68% and 83.51% at lengths 8, 32 and 64. The fixed-length positional ablation reached at most 53.77% at length 32 for plain NoPE, 56.25% for sinusoids and 53.96% for a fixed marker at position 1. The marker adds `(+1, −1, …)` to the first embedding. Interior embeddings contain only token identities. [Enumeration](../artifacts/window-round2-floor.json) · [Stock ablations](../artifacts/window-round2-diagnosis.json)

### The two-gain register

Use `x_t = ±1` for the two symbols, with −1 denoting the neutral symbol. Keep one scalar `s` per token. Start from `s[0,1] = x₁` and `s[0,t] = 0` for later positions. The fixed predecessor head reads `s[k−1,t−1]` with weight 1 and uses a constant −1 to the left of position 1. The zero state means zero log odds. At each pass, use the immutable input symbol to select one of two independently learned gains:

```text
u[k,t] = gain[x_t] × s[k−1,t−1]
s[k,t] = tanh(u[k,t])
logits[k,t] = (0, u[k,t])
```

Both gains start from independent standard Gaussians. Use 650 AdamW updates, a 20-step warmup to 0.035, a cosine finish at 0.0035, decay 0.001 on the two gains, global norm clipping at 1, betas 0.9 and 0.95, and epsilon `1e−8`. Timings cover optimizer updates, including new batches. Compilation, exhaustive validation and rollout diagnostics happen outside the timed interval.

| Seed | Exhaustive deep CE | Gap above floor | T32, K=32/64 | T64, K=64/128 | Updates |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.161459555 | 0.067% | 100% / 100% | 100% / 100% | 1.98 s |
| 2 | 0.161549746 | 0.123% | 100% / 100% | 100% / 100% | 1.02 s |
| 3 | 0.161535136 | 0.114% | 100% / 100% | 100% / 100% | 1.15 s |
| 4 | 0.161646040 | 0.183% | 100% / 100% | 100% / 100% | 1.05 s |
| 5 | 0.161507535 | 0.097% | 100% / 100% | 100% / 100% | 1.24 s |

The long evaluations each use 256 independently drawn words. Every learned neutral gain is positive and every learned negation gain is negative. For any length, multiplication by these gains preserves a known sign for the neutral token and flips it for the negation token. Multiplication by a gain leaves a zero predecessor at zero, and tanh preserves all three cases. This establishes the parity sign and the undecided exterior by induction. The known signs also stay away from zero. The smallest learned gain magnitude is 7.5576, and `tanh(0.99 × 7.5576) > 0.99`. All positions become exactly stationary by pass `T` because position 1 reads a constant and each subsequent position reads only its predecessor's changing state.

At length 64, the seed-1 99%-accurate front advances one site per pass from position 2 through 64. It is 100% correct inside the cone and assigns probability 0.5 everywhere outside it. Centered RMS state change is approximately 0.125 at each pass through 63 as one more site's state becomes known, then exactly zero from pass 64 through 128. The scalar state is represented on the centered unit axis `(-1,+1)/√2` for the norm measurement. The probe stores the `64 × 128` position-by-pass accuracy matrix in column-major order under `cone_T64`.

For the terminal covector, seed 1 stores the gradient of the class-2 minus class-1 logit at site 16, pass 16. Each backward pass moves one site left. The pre-tanh cotangent norm is 1 at the seeded site, `1.05 × 10⁻⁶` at site 15 and `4.33 × 10⁻¹³` at site 14. The left-moving path reaches site 1 at pass 1, then terminates at the fixed boundary. The strong attenuation is from the derivative of the saturated tanh. Register coordinates explain the exact support on one site per pass. [Seed screen](../artifacts/window-round2-register.json) · [Accuracy cone and covectors](../artifacts/window-round2-probe.json)

### Matched controls and Rule 60

The tied one-pass control keeps the two gains and uses the same uniform lengths. Its five losses fall between 0.497608948 and 0.497659044 nats. Its measured accuracies at lengths 32 and 64 are 52.77% and 51.06%, close to the respective information ceilings of 53.125% and 51.563%. The untied control uses 24 independently initialized blocks and selects the first `K` on each batch, for 48 parameters. Its five losses range from 0.164146406 to 0.174839251 nats, all above the 1% threshold. Its best pass-24 long evaluation reaches 88.92% at length 32 and 69.26% at length 64. The floor is achievable in the same limiting sense as for the tied model by using identical large-magnitude gains in every block.

The ReLU register controls keep the fixed head, scalar feedback and initialization of the recurrent state. Their two-input transition receives the current symbol and the predecessor's state. The 17- and 33-parameter models extrapolate at 100% in five of five seeds and the 13-parameter model in four of five. Every ReLU run missed the loss target, with losses from 0.20533 to 0.69035 nats. The gain transition keeps zero log odds invariant by construction. [Matched controls](../artifacts/window-round2-controls.json) · [ReLU screen](../artifacts/window-round2-register.json)

Rule 60 initializes the same register from every input bit, then runs autonomously. The transition interpolates between the gains using the current recurrent state. The target at each pass is the corresponding Rule 60 iterate, with an exact zero information floor. After separate training with the same schedule, all five seeds exactly match Lucas' theorem on every pass through 96 on 256 length-64 words and on a length-97 single impulse. Held-out mixed deep loss ranges from 0.000232 to 0.000612 nats. The script independently verifies Lucas' theorem against direct iteration on the evaluation words. [Rule 60](../artifacts/window-round2-rule60.json)

### Stock PicoGPT stream variants

The stock ablations retain content-based attention, both block LayerNorms, a ReLU MLP, and a trainable embedding and readout. The three stream maps are `B(X) + X₀`, `X₀ + (B(X) − X)`, and `LN(B(X) + X₀)`. In the third map, the final LayerNorm has unit gains and zero offsets, with no learned parameters. For the second map the implementation adds the corrections directly and cancels the old stream's direct gradient. The update still lets the block's MLP see the original attention residual internally.

Matched runs at 582 parameters use `D=8, H=1, F=12`, 900 updates, batch 12 and peak learning rate 0.02. With the boundary marker and mixed supervision, seed 2 shows the effect of the stream map.

| Stream map | Deep CE | T32, K=32 / 64 | T64, K=64 / 128 |
|---|---:|---:|---:|
| Post-injection | 0.432126669 | 56.68% / 56.68% | 53.71% / 53.72% |
| Corrections plus fresh input | 0.193085925 | 99.68% / 97.14% | 96.49% / 94.07% |
| Post-injection then LayerNorm | 0.161369632 | 100% / 100% | 100% / 100% |

The same three recipes in seed 1 all stalled near 0.512 nats and 52.20% at length 64. Their measured stream sizes distinguish the maps even in this seed. Each cell pairs the RMS column-centered state norm and the RMS column-centered step norm at length 64.

| Stream map | Pass 1 | Pass 8 | Pass 32 | Pass 64 | Pass 128 |
|---|---:|---:|---:|---:|---:|
| Post-injection | 33.05 / 31.37 | 202.02 / 23.96 | 774.32 / 23.87 | 1,538.05 / 23.88 | 3,066.96 / 23.90 |
| Corrections plus fresh input | 18.24 / 16.88 | 13.09 / 0.370 | 13.09 / `1.77e−8` | 13.09 / `2.13e−15` | 13.09 / `2.14e−15` |
| Post-injection then LayerNorm | 2.83 / 2.40 | 2.83 / 0.096 | 2.83 / `3.29e−9` | 2.83 / `3.44e−16` | 2.83 / `3.08e−16` |

Unnormalized post-injection produces roughly linear growth even after column centering. The two bounded updates reach numerically stationary states for this seed, with inaccurate predictions. [Stock ablations](../artifacts/window-round2-diagnosis.json) · [Stream norms by pass](../artifacts/window-round2-stream.json)

The 582-parameter normalized checkpoint in seed 2 also learned a sharp cone with content-based attention. At length 64 and pass 128, average predecessor weight was 0.621 across nonboundary queries, with a range of 0.123–0.9996. About 25.2% put at least 0.95 on the predecessor. Its centered RMS step norm was 0.186 at pass 64 and `1.08e−15` at pass 128. The terminal covector from site 16 spreads across the allowed backward cone. Its norm is 3.92 at the seed, then 0.77 at position 15 and 0.56 at position 16 one pass back. [Stock cone and covectors](../artifacts/window-round2-stock-probe.json)

In seed 2, the matched ablation below varies the position signal and the training distribution while keeping the normalized stock block and optimizer settings. The fixed boundary-marker run reached within 0.008% of its own training floor and lost accuracy on long passes. The mixed recipe at the same boundary marker held 100% through the long rollouts. The combined ablation changes lengths and pass counts together.

| Position signal | Distribution | Deep CE | T32, K=32 / 64 | T64, K=64 / 128 |
|---|---|---:|---:|---:|
| Sinusoidal | Fixed | 0.531706 | 52.43% / 52.43% | 51.09% / 51.09% |
| Sinusoidal | Mixed | 0.482898 | 53.65% / 53.67% | 52.32% / 52.40% |
| Boundary marker | Fixed | 0.227457 | 87.54% / 82.43% | 84.42% / 81.95% |
| Boundary marker | Mixed | 0.161370 | 100% / 100% | 100% / 100% |

With the mixed recipe, the 582-parameter normalized stock block met both targets in one of five seeds on the 900-update schedule. The smallest successful stock checkpoint uses `D=4, F=16` and 270 parameters. It also succeeded in one of five seeds, with loss 0.1613745 nats and 100% on all four rollouts in seed 2. The 1,250-update schedule used batch 8 and peak rate 0.02. Corrections plus fresh input also produced exact long rollouts under a 1,500-update schedule and batch 8. The 444- and 582-parameter shapes each met both targets in one of five seeds. The 786-parameter shape missed both targets in every seed of that refresh screen. [Normalization screen](../artifacts/window-round2-stock-normalized.json) · [Narrow normalization seeds 3–5](../artifacts/window-round2-stock-narrow.json) · [Refresh screen](../artifacts/window-round2-stock-tune.json)

The same two-gain register also met both targets in five of five seeds with fixed `T=K=8`. Its loss range was 0.2275274–0.2276773 nats against the 0.2274389-nat fixed floor. Its recurrence carries the known state one site per pass at every length, including when training sees only one horizon. [Matched position and distribution ablation](../artifacts/window-round2-distribution.json)

### Round 2 reproduction

Run from the repository root with Julia 1.13.3.

~~~sh
julia --project -O1 loop/test/window.jl
for mode in floor diagnosis stream distribution register controls probe stock-probe stock-normalized stock-narrow stock-tune rule60; do
    julia --project -O1 -t 4 loop/scripts/window_round2.jl "$mode"
done
~~~

For the most comparable timing of the two-gain probe, run the `probe` command on one Julia thread. The test suite checks the exact mixed floor, equivalence of traced and ordinary forward passes, finite differences for each new stream rule and register transition, and the causal support of the forward and backward passes. [Implementation](../src/window.jl) · [Round 2 runner](../scripts/window_round2.jl) · [Tests](../test/window.jl)

## Round 3

The 270-parameter block met the complete target in **4 of 5** independent initializations. Those four seeds came within 1% of the mixed loss floor and scored 100% on the length-32 and length-64 evaluations at both `K=T` and `K=2T`. The block keeps learned attention, both block LayerNorms, the ReLU MLP, embedding and final LayerNorm/readout. It starts the recurrent stream with only the first token embedded, anchors the attention residual to the input embedding, and adds a fixed predecessor bias to the attention logits. Timed updates took **6.51–9.87 seconds** per seed in Julia. [Five-seed screen](../artifacts/window-round3-anchor-boundary-start.json)

### Selected configuration

The shape is `V=2, D=4, H=1, F=16, L=1`. The stock parameter layout contains 270 trainable scalars. The boundary start, predecessor bias and stream normalization add zero parameters. All weights start from PicoGPT's ordinary initialization. At length 64 and pass 128 in the selected seed 3, predecessor weights on sites 2–64 had mean 0.989 and spanned 0.967 to almost 1. The centered RMS step was `9.08e−16`. In seed 5, the same bias produced weights from 0.540 to almost 1.

| Seed | Exhaustive mixed CE | Above floor | T32, K=32 / 64 | T64, K=64 / 128 | Updates |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.193545053 | 19.9528% | 86.38% / 99.08% | 77.12% / 96.06% | 9.87 s |
| 2 | 0.161377578 | 0.0165% | 100% / 100% | 100% / 100% | 9.56 s |
| 3 | 0.161356316 | 0.0033% | 100% / 100% | 100% / 100% | 6.51 s |
| 4 | 0.161749377 | 0.2469% | 100% / 100% | 100% / 100% | 9.39 s |
| 5 | 0.161622634 | 0.1684% | 100% / 100% | 100% / 100% | 9.49 s |

The exact mixed floor is `(384975233676617 / 1653814216454400) × ln(2) = 0.161350951729293` nats. Validation enumerates all binary words at each length 4–12 with ordinary hard labels and equal weight by length, by `K=T−1…2T` at each length, and by pass within each rollout. The long tests use 256 fixed pseudorandom words per length. Timing covers batches and optimizer updates, with two Julia threads and one BLAS thread. Julia startup, compilation, exact validation and rollout evaluation occur outside the timed intervals.

Seed 1 showed a delayed front at length 64. The contiguous 99%-accurate prefix reached site 25 at pass 64 and site 44 at pass 128. Seed 3 advanced one site per pass through pass 63 and stayed exact. The selected configuration came from a screen on seeds 1–5, and the table reports those same seeds.

The forward changes from stock PicoGPT are these lines.

~~~julia
X₀, _ = window_embed(x, E, :boundary)   # token embeddings with (+1,−1,…) at site 1
X = zero(X₀)
X[:, 1:T:end] .= X₀[:, 1:T:end]         # start only site 1 with a known state
for k in 1:K                           # reuse p.blocks[1]
    O, _, _ = window_mix(Q, Kkeys, V, H, T; lag_bias = 4.0)
    X₁ = X₀ + first(affine(O, Wo, bo))  # fresh input enters before LN₂ and the MLP
    X = first(layernorm(X₁ + Y₂, ones(D), zeros(D)))
    Z[k] = first(readout(X, p))
end
~~~

Each pass computes queries, keys and values from `LN₁(X)` and computes `Y₂` from `LN₂(X₁)` and the ReLU MLP. `window_mix` replaces the full causal mask with self and predecessor, and subtracts 4 from the self-minus-predecessor logit difference. The stream LayerNorm has unit gain and zero offset. The fresh embedding supplies each position's current bit at every pass. The first embedding can reach data site `k+1` at pass `k`, so the original light cone and its exact floor still apply. [Attention and loop implementation](../src/window.jl)

The **1,250 unrolled updates** use batch size 8, with one uniformly sampled length and horizon per batch. The deep loss uses the known parity in the cone and the exact fair-label expectation outside the cone. AdamW uses a 20-step ramp to 0.02, a cosine finish at 0.002, `β₁=0.9, β₂=0.95, ε=1e−8`, matrix weight decay 0.01 and global norm clipping at 1. The selected recipe uses ordinary random initialization and the same training and inference recurrence. [Training implementation](../scripts/window_round3.jl)

### Seed reliability by change

The floor column requires the nominal mixed loss within 1% and all four long rollouts exact. The rollout column requires 100% on all four long rollouts. The historical stock row uses the saved Round 2 checkpoints. The Round 3 baseline samples the same distribution through Julia's range samplers and uses a binary-gap implementation of the same conditional loss.

| Configuration and change | Parameters | Rollouts | Floor and rollouts |
|---|---:|---:|---:|
| Round 2 gain register with a fixed predecessor head and learned two-symbol transition | 2 | 5/5 | 5/5 |
| Round 2 ReLU registers with fixed predecessor head | 17 / 33 | 5/5 each | 0/5 each |
| Round 2 normalized stock block with the first-token marker | 270 | 1/5 | 1/5 |
| Round 3 baseline with the same block and schedule | 270 | 0/5 | 0/5 |
| Baseline with a BOS embedding in place of the first-token marker | 274 | 0/5 | 0/5 |
| Baseline with query and key matrices scaled by 0.01 at initialization | 270 | 1/5 | 1/5 |
| Near-uniform initialization with the output projections and readout additionally scaled by 0.1 | 270 | 0/5 | 0/5 |
| Near-identity initialization combined with BOS | 274 | 0/5 | 0/5 |
| Baseline with a curriculum that grows length and depth for 300 updates | 270 | 1/5 | 1/5 |
| Curriculum with BOS, with or without near-identity initialization | 274 | 0/5 each | 0/5 each |
| Curriculum with fresh input added to both corrections after the MLP | 270 | 0/5 | 0/5 |
| Baseline with fresh input anchored in the first residual | 270 | 3/5 | 1/5 |
| Anchored residual with a predecessor logit bias of +2 | 270 | 4/5 | 3/5 |
| Same +2 bias followed by 300 low-rate updates, with two thirds of batches at `K=1…3` | 270 | 4/5 | 3/5 |
| Anchored residual with the logit bias raised from +2 to +4 | 270 | 3/5 | 3/5 |
| Same +4 bias with a zero initial recurrent stream after site 1 | 270 | **4/5** | **4/5** |
| Anchored residual with +4 bias and width increased to `D=8, F=32` | 922 | 5/5 | 4/5 |
| Anchored residual with +2 bias and BOS | 274 | 2/5 | 0/5 |
| Anchored residual with +4 bias, `D=6, F=16`, and 900 mixed updates | 444 | 5/5 | 3/5 |
| Same 444-parameter configuration with a 750-update simplex warmup | 444 | 5/5 | 3/5 |
| Warmup with the undecided state between the two known parity codes | 444 | 3/5 | 2/5 |
| Simplex warmup with width increased to `D=8, F=32` | 922 | 5/5 | 4/5 |

With +2 bias, the fourth exact-rollout seed missed the loss target because the arriving parity predictions remained less confident at the front. Extra shallow updates reduced its loss from 0.176318 to 0.175469 nats. Both 922-parameter models had the same issue in their one loss failure. The boundary-start configuration tied them on the complete criterion with fewer parameters and faster updates. [Anchored bias screen](../artifacts/window-round3-anchor-bias.json) · [Shallow finishing updates](../artifacts/window-round3-anchor-polish-resume.json) · [Wide direct screen](../artifacts/window-round3-anchor-wide.json)

The 750-step simplex warmup uses batches of 24 length-6 words. It alternates the real embedding stream for pass 1 with teacher-forced previous states for a pass sampled from 2–6. These previous states receive Gaussian noise with standard deviation 0.05. The warmup loss is mean squared error to three fixed centered state codes plus 0.2 times parity CE for that pass. The optimizer then resets its moments for 900 unrolled updates. [Warmup screen](../artifacts/window-round3-teacher-wide.json)

### Where the Round 2 seeds stalled

I replayed the five saved 270-parameter normalized checkpoints on the same 256 length-64 words. The front is the last position in the contiguous prefix with at least 99% accuracy. The centered step is the RMS across tokens of the change in the column-centered stream.

| Seed | Front at passes 1 / 4 / 16 / 64 / 128 | Pass-128 predecessor weight, mean | Pass-128 centered step | Pass-128 accuracy |
|---:|---|---:|---:|---:|
| 1 | 2 / 2 / 3 / 2 / 2 | 0.540 | 0.370 | 53.85% |
| 2 | 2 / 5 / 17 / 64 / 64 | 0.567 | 3.38e−16 | 100% |
| 3 | 1 / 1 / 2 / 2 / 2 | 0.510 | 0.0304 | 52.29% |
| 4 | 1 / 2 / 2 / 2 / 2 | 0.483 | 1.24e−16 | 52.91% |
| 5 | 2 / 4 / 14 / 5 / 5 | 0.735 | 0.0752 | 77.38% |

Seeds 1 and 3 stalled near the boundary while the stream kept moving. Seed 4 converged to an incorrect fixed point. Seed 5 transported parity for the first few passes but accumulated errors on longer rolls despite assigning the most average attention to the predecessor. All four failed seeds assigned some queries nearly all the weight on self and others nearly all on the predecessor at pass 128. Seed 2 settled with predecessor weights from 0.284 to 0.991. [Checkpoint replay](../artifacts/window-round3-archived-diagnostics.json)

### BOS and the light-cone floor

Replacing the first-site marker with a separate, unscored BOS embedding gives site 1 a real predecessor. At pass `k`, BOS is still outside the cone of data site `k+1`. That site has the same distribution of visible `k+1` bits as every later site. Even if the predictor is given the length `T` and pass `k`, for each such context it sees one position with a determined label and `T−k−1` positions with fair labels. Writing `m=T−k`, this gives a lower bound of `m × h((m+1)/(2m)) / T` nats at each pass up to `T−1`, where `h` is binary entropy. The mixed deep lower bound rises to 0.200602663 nats. Two BOS seeds with the anchored residual reached all four long accuracy targets, and their deep losses were 0.21231 and 0.21255 nats. [BOS with anchored residual](../artifacts/window-round3-anchor-bos.json)

### Independent Rule 60 check

Enumerating all 256 binary words of length 8 and applying the zero-boundary Rule 60 recurrence twice produced disagreement counts `[0, 128, 128, 128, 128, 128, 128, 128]` by site. A shared horizon-blind prediction sees one label of each kind on each of the 896 disagreements and two matching labels otherwise. Thus the CE infimum is `(7/16) ln 2 = 0.3032518915` nats and the accuracy ceiling is `25/32 = 78.125%`, agreeing with Astra. The enumerator lives in the Round 3 runner and its counts are covered by the window tests. [Enumeration code](../scripts/window_round3.jl)

### Specimen and reproduction

The artifact stores seed 3's parameter vector in PicoGPT's order, the exact floor, the per-update minibatch losses and the exhaustive curve. Its length-16 trace covers passes 0–16. Each record has the stream, the two attention weights in self/predecessor order, the two predicted probabilities and the cotangent of the terminal class-2 minus class-1 logit at site 16. During that differentiation, the fresh input embedding is held fixed. Pass 0 shows the initial stream and its reference readout. All tensors store shape and column-major data. [Figure artifact](../artifacts/window-specimen.json)

Run from the repository root with Julia 1.13.3.

~~~sh
julia --project -O1 -t 2 loop/scripts/window_round3.jl anchor-boundary-start
julia --project -O1 loop/scripts/window_round3.jl specimen anchor-boundary-start 3
julia --project -O1 loop/scripts/window_round3.jl rule60
julia --project -O1 test/runtests.jl
julia --project -O1 loop/test/runtests.jl
julia --project -O1 loop/test/window.jl
~~~

The window tests cover finite differences for the BOS input embedding, both anchored residual variants, the boundary start, and the warmup pullback. They also check the trace shape, causal support, and the two Rule 60 bounds. [Window tests](../test/window.jl)

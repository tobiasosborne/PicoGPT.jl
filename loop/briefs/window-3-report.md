# Prefix parity with a two-token attention window, Round 3

The 270-parameter block met the complete target in **4 of 5** independent initializations. Those four seeds came within 1% of the mixed loss floor and scored 100% on the length-32 and length-64 evaluations at both `K=T` and `K=2T`. The block keeps learned attention, both block LayerNorms, the ReLU MLP, embedding and final LayerNorm/readout. It starts the recurrent stream with only the first token embedded, anchors the attention residual to the input embedding, and adds a fixed predecessor bias to the attention logits. Timed updates took **6.51–9.87 seconds** per seed in Julia. [Five-seed screen](../artifacts/window-round3-anchor-boundary-start.json)

## Selected configuration

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

## Seed reliability by change

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

## Where the Round 2 seeds stalled

I replayed the five saved 270-parameter normalized checkpoints on the same 256 length-64 words. The front is the last position in the contiguous prefix with at least 99% accuracy. The centered step is the RMS across tokens of the change in the column-centered stream.

| Seed | Front at passes 1 / 4 / 16 / 64 / 128 | Pass-128 predecessor weight, mean | Pass-128 centered step | Pass-128 accuracy |
|---:|---|---:|---:|---:|
| 1 | 2 / 2 / 3 / 2 / 2 | 0.540 | 0.370 | 53.85% |
| 2 | 2 / 5 / 17 / 64 / 64 | 0.567 | 3.38e−16 | 100% |
| 3 | 1 / 1 / 2 / 2 / 2 | 0.510 | 0.0304 | 52.29% |
| 4 | 1 / 2 / 2 / 2 / 2 | 0.483 | 1.24e−16 | 52.91% |
| 5 | 2 / 4 / 14 / 5 / 5 | 0.735 | 0.0752 | 77.38% |

Seeds 1 and 3 stalled near the boundary while the stream kept moving. Seed 4 converged to an incorrect fixed point. Seed 5 transported parity for the first few passes but accumulated errors on longer rolls despite assigning the most average attention to the predecessor. All four failed seeds assigned some queries nearly all the weight on self and others nearly all on the predecessor at pass 128. Seed 2 settled with predecessor weights from 0.284 to 0.991. [Checkpoint replay](../artifacts/window-round3-archived-diagnostics.json)

## BOS and the light-cone floor

Replacing the first-site marker with a separate, unscored BOS embedding gives site 1 a real predecessor. At pass `k`, BOS is still outside the cone of data site `k+1`. That site has the same distribution of visible `k+1` bits as every later site. Even if the predictor is given the length `T` and pass `k`, for each such context it sees one position with a determined label and `T−k−1` positions with fair labels. Writing `m=T−k`, this gives a lower bound of `m × h((m+1)/(2m)) / T` nats at each pass up to `T−1`, where `h` is binary entropy. The mixed deep lower bound rises to 0.200602663 nats. Two BOS seeds with the anchored residual reached all four long accuracy targets, and their deep losses were 0.21231 and 0.21255 nats. [BOS with anchored residual](../artifacts/window-round3-anchor-bos.json)

## Independent Rule 60 check

Enumerating all 256 binary words of length 8 and applying the zero-boundary Rule 60 recurrence twice produced disagreement counts `[0, 128, 128, 128, 128, 128, 128, 128]` by site. A shared horizon-blind prediction sees one label of each kind on each of the 896 disagreements and two matching labels otherwise. Thus the CE infimum is `(7/16) ln 2 = 0.3032518915` nats and the accuracy ceiling is `25/32 = 78.125%`, agreeing with Astra. The enumerator lives in the Round 3 runner and its counts are covered by the window tests. [Enumeration code](../scripts/window_round3.jl)

## Specimen and reproduction

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

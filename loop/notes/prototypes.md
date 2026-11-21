# Looped transformer prototypes, round 1

The experiments distinguish fitting the target at the trained pass from learning a transition that
can be iterated. Every model uses Float64 and the same final LayerNorm and affine readout at
each inspected pass. The loop uses one block's parameter vector. The untied control uses separate
blocks in the same execution graph. The shallow controls use either the tied model's exact
parameter count or an MLP widened to approximately match the untied budget.

## Reproduction

Run from the repository root.

```sh
julia --project loop/scripts/prefix_parity.jl
julia --project loop/scripts/rule60.jl
julia --project -O0 loop/scripts/prefix_parity_widths.jl
julia --project -O0 loop/scripts/prefix_parity_local.jl
julia --project -O0 loop/scripts/rule60_local.jl
julia --project -O0 loop/scripts/prefix_parity_local.jl --cautious
julia --project -O0 loop/scripts/rule60_local.jl --cautious
julia +1.12.5 --project -O1 loop/scripts/pointer_chasing.jl
julia --project -O1 loop/test/runtests.jl
```

The stock P1 and P2 runs used Julia 1.13.3. P3 used Julia 1.12.5. The JSONs store each seed,
configuration, optimizer settings, parameter vector, pre-update minibatch losses and final loss,
and evaluation metrics. Across the stock P1 and P2 runs, the timed update loops took
1.35–9.34 seconds per model. Julia startup, compilation and evaluation also contribute to
script wall time. Training samples fresh uniform independent binary words with a separate
Xoshiro stream per seed. Stock P1 and P2 use BRIEF §13's initialization, peak and final learning
rates, warmup, matrix-only decay, AdamW constants, and clipping. They use 400 updates and batch
size 24 instead of the language specimen's full-batch 200 updates. The separate initial and final
losses use the same fixed evaluation batch. The exact accuracy at training length enumerates the
whole support, so those words can also occur in training. Longer lengths use independently
sampled 256-word batches.

## P1 · “Not not to be”

The input is 1 for a neutral word and 2 for a negation. The target at each position is the XOR
of all negations through that position. The stock model has D=8, H=1, F=12, and K=5 at training
length T=5. The block adds the original embedded stream after every pass, including pass one.
The untied control uses the same injection. The matched-budget control has F=137. Deep
supervision in the stock experiment scores the full prefix target at every pass. Training length is exhaustively tested on all 32 words. Traces
also cover T=10 and T=20, with up to 10 passes for each one-block checkpoint.

**Stock accuracy at the trained pass, in seed order 1/2/3**

| Model | Parameters | T=5 | T=10 | T=20 |
|---|---:|---:|---:|---:|
| Tied, final loss | 582 | 87.50 / 98.75 / 69.38% | 70.78 / 74.53 / 61.80% | 60.27 / 62.30 / 55.37% |
| Tied, deep loss | 582 | 85.63 / 86.25 / 69.38% | 67.38 / 69.34 / 60.39% | 59.49 / 59.02 / 54.43% |
| One pass | 582 | 80.63 / 78.75 / 79.38% | 64.26 / 63.59 / 63.98% | 57.32 / 57.09 / 56.72% |
| Untied, five blocks | 2,710 | 100 / 100 / 100% | 78.32 / 77.19 / 74.88% | 63.77 / 62.42 / 63.65% |
| Wide one pass | 2,707 | 69.38 / 69.38 / 68.75% | 61.21 / 61.76 / 61.88% | 55.61 / 55.35 / 55.29% |

The strongest stock tied model's fixed-batch loss went from 0.7003 to 0.0232. At T=5 its
contiguous 95% accuracy front went through positions `2,2,3,4,4` by the trained pass and stayed
at position 4 through pass 10. On pass 5 it got the first four positions exactly and missed two of
the 32 fifth positions. This was a short advance followed by a plateau. At T=10, accuracy ran
from 74.53% at pass 5 to 73.24% at pass 10. The stock traces therefore support neither a sustained
linear recurrence nor a doubling front. Since full causal attention can already read token one
at any position, the accuracy front is an empirical diagnostic. The artifact also records how
often toggling the first token changes each pass's prediction at every position.

**Width screen.** The screen reused seed 2, the strongest stock seed, for six smaller
models. At T=5, using (D,F) of (2,4), (4,4), (4,12), (6,8), (8,4), and (8,8) yielded
50.00%, 67.50%, 92.50%, 84.38%, 88.13%, and 78.13%. The best smaller configuration
had 234 parameters and reached 71.29% at T=10. The 582-parameter model remains the
smallest random initialization tested here to clear 95% at T=5, and only cleared it in one seed.

**Structured local transition.** D=6, H=1, F=22 gives 522 parameters. Each token reads itself
and its predecessor with a fixed zero boundary. There is no positional code. Each pass adds the
original embedding after the block and then applies a fixed LayerNorm. The spline initializer
constructs residual updates for every combination of the raw bit, current state bit, and
predecessor or boundary. With perturbation standard deviation 0.002, 120 end-to-end updates,
batch size 16, peak learning rate 0.001, final rate 0.0001, ten warmup updates and zero decay,
the trained-pass accuracy was 70.63%, 67.50%, and 67.50%. It lost the intended recurrence.

The unperturbed 522-parameter reference matched the exact sliding-window parity target on every
evaluated token and pass. Its front grew from position 2 after pass 1 to position 20 after
pass 19 at T=20. This is the linear front predicted by an adjacent window and a state initialized
from the input bit. At T=5 any model with one such pass has an expected accuracy ceiling of 70%
on uniform inputs. It can see all required bits at the first two positions. At each later
position there is an unseen, independent fair bit, making its target conditionally fair.

With perturbation reduced to 5 × 10⁻⁵, 60 updates, batch size 16, peak rate 10⁻⁵, final
rate 10⁻⁶, five warmup steps and zero decay, all seeds preserved exact window parity on the
first four passes at T=5, 10 and 20. At each length the front moved through `2,3,4,5` and then
fell to 2 on pass 5. At T=20, full prefix accuracy at pass 20 was 51.46–56.48%. The reference
transition exposes linear information propagation. The fine-tuning result exposes the separate
requirement of preserving that transition after the supervised depth.

Data [prefix-parity.json](../artifacts/prefix-parity.json),
[prefix-parity-widths.json](../artifacts/prefix-parity-widths.json),
[prefix-parity-local.json](../artifacts/prefix-parity-local.json),
[prefix-parity-local-cautious.json](../artifacts/prefix-parity-local-cautious.json).

## P2 · Rule 60

Rule 60 updates each bit by XOR with its left neighbor, with an implicit zero at the left
boundary. The stock experiment uses D=8, H=1, F=12, T=6 and K=3, without injection. The main
objective supervises the third iterate end to end. The deep variant scores the true iterate at
each pass. Both single-pass controls train on the third iterate. The wide control has F=75.
Evaluation compares predictions with Lucas's formula, which selects lag j exactly when `(j & k) == j`. The script verifies that
formula against iterative rule 60 on every evaluation batch and exports a single-impulse run.

**Stock third-iterate accuracy at trained depth, in seed order 1/2/3**

| Model | Parameters | T=6 | T=12 |
|---|---:|---:|---:|
| Tied, final loss | 582 | 70.31 / 98.70 / 67.71% | 61.82 / 74.80 / 59.93% |
| Tied, deep loss | 582 | 74.22 / 77.08 / 78.91% | 61.91 / 63.05 / 64.62% |
| One pass | 582 | 72.14 / 68.75 / 82.55% | 62.11 / 61.20 / 66.31% |
| Untied, three blocks | 1,646 | 92.97 / 98.70 / 98.44% | 72.75 / 74.97 / 76.30% |
| Wide one pass | 1,653 | 67.71 / 69.79 / 96.88% | 59.41 / 60.77 / 74.87% |

The strongest tied seed's first and second iterates were 66.67% and 61.46% correct at T=6.
The fourth was 61.46% and the twelfth 52.08%. On the exported single impulse it matched the
complete row at pass 3 only. The 98.70% figure at pass 3 is evidence of a good fitted endpoint.
The intermediate predictions and rollout show how badly that figure can overstate what a model
has learned about the dynamics.

**Structured local transition.** D=4, H=1, F=10 gives 216 parameters. The model uses the same
adjacent window, no positional code, and fixed LayerNorm after each pass. The ReLU spline
initializer codes the left/current XOR directly. The perturbation and optimizer schedule are
the same as in local P1. Training still uses just the third-iterate loss pulled through all
three passes. In all three seeds passes 1–3 are exactly correct at T=6, T=12 and T=24.
At T=24, pass 4 falls to 91.91–97.49% and pass 24 to 51.92–52.77%. The 49-position impulse
has exact rows through pass 3 in seed 1, and through pass 4 in the other two.
The fixed-batch loss improves from 0.1215–0.2264 to 0.0326–0.0360. The example makes the
loss and dynamical stability measurements directly comparable.

The unperturbed 216-parameter reference exactly matched Lucas at every tested length for
24 passes and reproduced all 24 rows of the 49-position impulse. Its one-pass ceiling for the
third-iterate target is 66.67% at T=6 by the same unseen-bit argument as local P1. The cautious
schedule preserved exactly five impulse rows and five exact passes on T=24 in each seed.
At T=24 all three were at 99.38% on pass 6 and fell to 58.66–59.52% by pass 24. Cautious P1
and P2 each fit in about one second of timed optimizer updates per seed.

Data [rule60.json](../artifacts/rule60.json), [rule60-local.json](../artifacts/rule60-local.json),
[rule60-local-cautious.json](../artifacts/rule60-local-cautious.json).

## P3 · Two permutation hops

There are four symbols. A word presents four edge tokens in random source order and a terminal
query token. Each ordered pair has its own token ID. The prediction at the final position is
the second successor. The model is D=12, H=2, F=24, with 1,808 parameters in each control. It adds
the original stream after every pass. The objective scores only the query, using 500 updates and
batch size 32. Evaluation exhausts all 2,304 combinations of permutation, row ordering, and
starting symbol. These combinations can also occur during training.

The one-pass control scored 87.80%, 95.14%, and 89.24%. The two-pass tied model scored 81.42%,
95.18%, and 99.05%. The input-only baseline outputs the start and scores 50%. Thus high accuracy
at this scale is attainable with a single pass. The artifact reports hop-matched and fixed-target
accuracy at each pass through pass 4.

Data [pointer-chasing.json](../artifacts/pointer-chasing.json).

## Specimen recommendation

Lead with P1's exact local reference, then train the same-shaped model and measure how long
the front survives. The front gives every pass a meaning, and the one-pass ceiling follows from
the receptive field. The 582-parameter stock result is a compact demonstration of short-word
fitting and the misleading appearance of an iterative algorithm. P2 makes the autonomous
companion, drawing two versions of the same impulse with an error at the first mistimed row.
The random-init results do not support a claim that the book can already train a reliable
length-generalizing specimen from scratch.

## Round 2

### A trainable predecessor register

Replace the stock attention/residual block with a hard predecessor head and a shared two-layer
ReLU MLP that writes a scalar register, clipping the feedback to `[-1, 1]`. The binary logit
difference is the MLP output before clipping. Bits 0 and 1 enter as −1 and +1. The register
starts at the input bit, and the left boundary supplies −1. Prefix parity gives the MLP two
inputs, the immutable input bit and the predecessor's register. Rule 60 gives it the current
register and its predecessor. The fixed head provides the position dependence. A block with
`F` hidden units has `4F+1` learned parameters, all in the MLP. Its weight matrices start
from independent Gaussian draws with standard deviations `1/√2` and `1/√F`, and biases
start at zero.

Each update uses 16 fresh words in two equally sized sub-batches of distinct lengths drawn
uniformly from 2–12. Prefix parity runs for the word length. Rule 60 draws its requested pass
uniformly from 1–12 for each sub-batch. The optimizer is AdamW with 450 steps, a 20-step warmup,
a cosine from 0.015 to 0.0015, matrix-only decay 0.001, gradient norm clip 1, betas 0.9 and 0.95,
and epsilon `1e-8`. Timings measure the update loop after a warm-up objective call. Julia
startup, compilation of new experiment paths, and evaluation take additional time.

All training lengths are evaluated separately on 128 held-out draws per length. Some short
words necessarily occurred during training. The long evaluation uses 128 independently drawn
48-bit words through pass 96. The artifacts contain every update loss and every position's
accuracy by pass. Stream changes are recorded as RMS and maximum absolute changes on four
additional words per traced length. Attention statistics exclude the boundary in their
predecessor and self averages and report the boundary weight separately.

### Four hidden units

The 17-parameter block gave **5/5 exact seeds on both tasks**. Prefix parity used final-pass
loss and took 0.80–0.94 seconds per seed. Rule 60 used deep loss and took 0.78–0.96 seconds per
seed. At each required training length and pass the accuracy was 100%. On the 48-bit tests,
prefix parity was exact from pass 48 through 96 and Rule 60 was exact at every pass through 96.
All Rule 60 seeds reproduced the 96 complete Lucas impulse rows. Their smallest four-input
gate margins were 1.60 for prefix parity and 1.59 for Rule 60, enough to write each correct bit
exactly. The measured front, attention, and stream changes match the eight-hidden-unit results
below because the clipped states implement the same recurrence.
[Prefix parity](../artifacts/round2-prefix-compact-tied.json)
[Rule 60](../artifacts/round2-rule60-compact-tied.json)

| Matched four-hidden-unit controls | Parameters | Prefix parity at lengths 12 and 48 | Rule 60 at lengths 12 and 48 |
|---|---:|---:|---:|
| One pass | 17 | 58.46%, 52.10% | 60.16–62.91%, 52.77–56.43% |
| Untied, 12 layers | 204 | 65.23–87.63%, 54.09–59.44% | 49.61–50.26%, 49.84–50.07% |
| Wide one pass | 205 | 58.46%, 52.10% | 61.36–62.91%, 54.47–56.43% |

Rule 60 reports the average over requested horizons 1–12 for the one-pass controls and pass
12 for the untied controls. Both one-pass widths read two bits and produce one prediction
for all horizons. The untied prefix model has at most a 13-position receptive field at its
last layer. The untied Rule 60 models fit the first iterate exactly at length 48 and lose it
by layer 12, where accuracy is at chance.
[Prefix one pass](../artifacts/round2-prefix-compact-one.json)
[Prefix untied](../artifacts/round2-prefix-compact-untied.json)
[Prefix wide](../artifacts/round2-prefix-compact-wide.json)
[Rule 60 one pass](../artifacts/round2-rule60-compact-one.json)
[Rule 60 untied](../artifacts/round2-rule60-compact-untied.json)
[Rule 60 wide](../artifacts/round2-rule60-compact-wide.json)

### Three and two hidden units

With 13 parameters and the same objectives and optimizer, four of five seeds qualified on
each task. Each passing prefix seed was exact at all trained lengths and from pass 48 through
96 at length 48. Each passing Rule 60 seed was exact at every evaluated pass and on all 96
impulse rows. Timed updates took 0.56–0.64 seconds for prefix parity and 0.71–0.98 seconds for
Rule 60. Prefix seeds 1–4 had four-pair margins of at least 1.83. The Rule 60 margins cleared
1 in seeds 1, 3 and 4. Seed 2's smallest was 0.61, so its evidence comes from the rollouts.
The successful prefix models had the same linear front and exact fixed point as the
17-parameter models. Rule 60 seed 2 retained nonzero feedback differences at pass 96 (RMS
1.342). On both tasks seed 5 stalled at a 99%-accurate front of one position, with zero
state change by pass 48. Its length-48 pass-48 accuracies were 52.15% and 51.99%, respectively.
[13-parameter prefix parity](../artifacts/round2-prefix-three-tied.json)
[13-parameter Rule 60](../artifacts/round2-rule60-three-tied.json)

| Matched three-hidden-unit controls | Parameters | Prefix parity at lengths 12 and 48 | Rule 60 at lengths 12 and 48 |
|---|---:|---:|---:|
| One pass | 13 | 56.97–58.46%, 51.30–52.10% | 61.36–62.91%, 54.47–56.43% |
| Untied, 12 layers | 156 | 57.42–77.93%, 52.05–57.57% | 48.83–54.62%, 49.84–51.29% |
| Wide one pass | 157 | 58.46%, 52.10% | 61.36–62.91%, 54.47–56.43% |

The one-pass ceilings and stack depth apply as above. The two-hidden-unit, nine-parameter
prefix pilot gave one exact seed and four at 50.43–52.08% at pass 48. The four-hidden-unit
variant was the smallest tested one that worked in all five seeds on both tasks.
[Prefix one pass](../artifacts/round2-prefix-three-one.json)
[Prefix untied](../artifacts/round2-prefix-three-untied.json)
[Prefix wide](../artifacts/round2-prefix-three-wide.json)
[Rule 60 one pass](../artifacts/round2-rule60-three-one.json)
[Rule 60 untied](../artifacts/round2-rule60-three-untied.json)
[Rule 60 wide](../artifacts/round2-rule60-three-wide.json)
[Nine-parameter pilot](../artifacts/round2-screen-register-two-final.json)

### Prefix parity

All five seeds of the 33-parameter register scored 100% at every training length with
`K = n`, and 100% for every pass from 48 through 96 at length 48. With window supervision,
each pass is scored on the trailing `k+1` bits, including positions past the full-prefix front.
This run took 1.74–2.89 seconds per seed. Its fixed validation cross-entropy ranged from
0.662–0.764 at initialization and from 0.000006–0.008582 after training.
[Window-supervised runs](../artifacts/round2-prefix-tied.json)

The 99%-accurate contiguous front at length 48 is `2, 3, 4, …, 48`. Each pass adds one
position, beginning at position two because the initial register already encodes each raw bit.
The fixed head puts weight 1 on the predecessor and 0 on self at every nonboundary position.
At passes 1, 12, 46, 47 and 48, the RMS changes in the register are 1.465, 1.267, 0.250,
0.250 and 0. The change stays exactly zero through pass 96. For any fitted MLP, each position
becomes stationary once its predecessor is stationary. The first position has a constant
boundary, so every position has settled by pass `n`.
[Per-pass data](../artifacts/round2-prefix-tied.json)

| Window-supervised control | Parameters | Length 12 at last available pass | Length 48 at last available pass |
|---|---:|---:|---:|
| One pass | 33 | 58.46% | 52.10% |
| Untied, 12 layers | 396 | 100% | 63.62% |
| Wide one pass | 397 | 58.46% | 52.10% |

The one-pass model sees two bits. All targets past position two contain an independent unseen
bit, giving an expected ceiling of `1/2 + 1/n`. The wider model has the same receptive field.
The untied model learns the trained-length computation and its front reaches position 13 at
layer 12. A pass 48 would require additional layers with new parameters.
[One pass](../artifacts/round2-prefix-one.json)
[Untied](../artifacts/round2-prefix-untied.json)
[Wide](../artifacts/round2-prefix-wide.json)

Final-pass cross-entropy on the full prefix alone also gave 5/5 exact seeds and took 1.17–2.16
seconds per seed. Cone supervision, which scores only `t ≤ k` and presumes a linear front,
also gave 5/5 in 1.81–2.88 seconds. All three objectives produced the same observed front,
attention and state changes. The final-only one-pass and wide controls give 58.46% and 52.10%
at lengths 12 and 48. With final-only loss the 396-parameter untied model gives 95.90–100%
at length 12 and 61.51–63.38% at length 48 at its layer limit of 12.
Under cone supervision the untied model gets 100% at length 12 and 62.58% at length 48.
The one-pass controls score the full prefix target at the requested final horizon for
all three supervision settings.
[Final-only tied](../artifacts/round2-prefix-final.json)
[Final-only one pass](../artifacts/round2-prefix-final-one.json)
[Final-only untied](../artifacts/round2-prefix-final-untied.json)
[Final-only wide](../artifacts/round2-prefix-final-wide.json)
[Cone](../artifacts/round2-prefix-cone.json)
[Cone untied](../artifacts/round2-prefix-cone-untied.json)

The 33-parameter final-pass model also got 5/5 exact rollouts with unclipped feedback in
0.74–1.63 seconds per seed. Its front continued linearly, starting at position one in two
seeds and position two in the other three. The fixed head used weight 1 on the predecessor.
Seed 4 still changed at pass 48 with RMS difference 2.88. Every seed became exactly stationary
by pass 49, after the last position's dependency on the initial state expired. The 396-parameter
untied version got 100% at length 12 and 62.58–63.38% at length 48 at its last layer. Clipping
affects feedback only, so the single-pass controls share the clipped model's outputs.
[Unclipped prefix](../artifacts/round2-prefix-unbounded.json)
[Unclipped prefix untied](../artifacts/round2-prefix-unbounded-untied.json)

In the five-seed window-loss pilot, gradually raising the maximum length from 3 to 12 during
the first 60% of training left seed 3 at 52.08% at length 48. Training with `K = n + rand(0:4)`
gave 5/5 exact pilot rollouts, matching the fixed `K = n` pilot. Both measurements use 24 words
at each of lengths 12 and 48.
[Curriculum screen](../artifacts/round2-screen-register-curriculum.json)
[Extra passes screen](../artifacts/round2-screen-register-extra.json)

### Rule 60

The autonomous register with deep supervision learned every iterate in all five seeds.
Accuracy was 100% at each measured training length and pass through 12, and at every pass
through 96 at length 48. On a 193-position impulse, every model reproduced all 96 rows of
Lucas' formula exactly. The impulse's right edge advances one position per pass. Training took
1.42–2.14 seconds per seed. Fixed validation cross-entropy went from 0.681–0.774 to
0.000009–0.02822.
[Tied runs and impulse rows](../artifacts/round2-rule60-tied.json)

Attention is the same exact predecessor shift as in prefix parity. The sampled register keeps
changing. RMS changes at passes 1, 12, 48 and 96 are 1.465, 1.436, 1.346 and
1.407.
[Per-pass traces](../artifacts/round2-rule60-tied.json)

| Rule 60 control | Parameters | Length 12 | Length 48 |
|---|---:|---:|---:|
| One pass, average over requested passes 1–12 | 33 | 60.16–62.91% | 52.77–56.43% |
| Untied, pass 12 | 396 | 50.59–100% | 50.76–100% |
| Wide one pass, average over requested passes 1–12 | 397 | 61.36–62.91% | 54.47–56.43% |

The first and third controls have the same output for every requested horizon of the same
input, although the requested iterate can change. Width also leaves their two-bit receptive
field intact. One untied seed got 100% at every measured training length and depth, including
pass 12 on the long batch. Its trained stack ends at pass 12.
[One pass](../artifacts/round2-rule60-one.json)
[Untied](../artifacts/round2-rule60-untied.json)
[Wide](../artifacts/round2-rule60-wide.json)

Final-pass loss at random horizons also qualified in four of five seeds. Each passing seed was
exact on all measured words and impulse rows, taking 1.32–2.27 seconds per run across the five
seeds. Seed 3 scored 52.08% at pass 48, reached only 6 exact impulse rows, and converged to a
wrong state with zero RMS change by pass 48. Deep supervision recovered that seed. The
final-only untied control reached 49.61–77.02% at length 12 and 49.93–56.67% at length 48,
both at pass 12. The one-pass controls have the same objective under either supervision
mode because both score their one output on the requested iterate.
[Final-pass runs](../artifacts/round2-rule60-final.json)
[Final-pass untied](../artifacts/round2-rule60-final-untied.json)

Without clipping, the deep-supervised seeds all achieved at least 99.72% on the measured
training lengths and passes. Their length-48 results were 64.45–89.70% at pass 48 and
52.25–78.21% at the worst pass from 48 through 96. The 99%-accurate front for seed 5 at passes
48 and 96 reached positions 15 and 14. The fixed head continues to select the predecessor.
The best pass-48 seed's RMS state change grew from 5.55 at pass 1 to `1.08e33` at pass 48
and `4.57e38` at pass 96. This isolates the effect of bounded feedback in the autonomous task.
[Unbounded runs](../artifacts/round2-rule60-unbounded.json)

| Mean minibatch cross-entropy | Steps 1–50 | 101–150 | 251–300 | 401–450 |
|---|---:|---:|---:|---:|
| 17 parameters, prefix parity, final loss, seed 1 | 0.63336 | 0.11254 | 0.03026 | 0.01951 |
| 17 parameters, Rule 60, deep loss, seed 1 | 0.66430 | 0.14044 | 0.03234 | 0.02082 |
| 13 parameters, prefix parity, final loss, seed 1 | 0.59243 | 0.13135 | 0.03303 | 0.01989 |
| 13 parameters, Rule 60, deep loss, seed 1 | 0.65480 | 0.11888 | 0.02927 | 0.01757 |
| Prefix parity, window loss, seed 1 | 0.50167 | 0.00850 | 0.00018 | 0.00002 |
| Rule 60, deep loss, seed 1 | 0.51051 | 0.00848 | 0.00019 | 0.00002 |
| Rule 60, final loss, seed 3 | 0.66383 | 0.46620 | 0.40734 | 0.39639 |
| Rule 60, unbounded, deep loss, seed 5 | 0.66036 | 0.04239 | 0.13123 | 0.00148 |

The fitted clipped cells were checked at the four binary input pairs. A margin of at least
1 at each pair means the correct sign is written back exactly by clipping. Starting
from binary inputs and a binary boundary, induction then gives the discrete XOR update
at any length and pass. This gate check is independent of the number of random words used
to estimate rollout accuracy.

### Stock-block screen

The larger-length diagnosis used the same mixed-length batches with the stock block and
`K = n`. All entries below use seed 1 and 24 evaluation words at each length. The default
block has `D = 8, H = 1, F = 12`, sinusoidal positions and post-addition. These configurations
used 600 AdamW updates, batch 16, peak rate 0.004, final rate 0.0004, 20 warmup steps and matrix
decay 0.001. The relative-attention experiments use `F = 16` and drop the sinusoidal code.
The normalized variants apply a fixed LayerNorm to the entire state after each pass. Refresh
means `X₀ + B(X) − X`. The scaled variant multiplies the block residual by `1/K` for the
chosen rollout budget. Its pass 48 below is the midpoint of a 96-pass rollout at scale `1/96`.

| Change | Parameters | Length 12, pass 12 | Length 48, pass 48 |
|---|---:|---:|---:|
| Stock, post-addition | 582 | 62.15% | 52.43% |
| Add input before block | 582 | 66.67% | 55.21% |
| Post-addition and state LayerNorm | 582 | 63.89% | 51.39% |
| Post-addition with scaled residual | 582 | 64.93% | 50.43% |
| Full attention and refresh | 582 | 60.76% | 48.09% |
| Full deep prefix supervision | 582 | 61.81% | 51.82% |
| Wider stream, `D = 16, F = 24` | 2,058 | 59.38% | 51.04% |
| No positional code | 582 | 63.19% | 52.08% |
| Length and pass curriculum | 582 | 64.24% | 51.39% |
| Peak rate 0.012, final rate 0.0012 | 582 | 63.54% | 53.39% |
| Full relative bias +4, refresh, window loss | 650 | 62.85% | 50.87% |
| Two-key window, bias effectively zero, refresh, window loss | 650 | 60.76% | 51.82% |
| Two-key window, bias +3, refresh, window loss | 650 | 76.04% | 57.73% |
| Two-key window, bias +3, post-addition and LayerNorm, window loss | 650 | 70.14% | 55.21% |
| Hard predecessor, refresh, window loss | 650 | 66.32% | 53.30% |
| Hard predecessor, refresh, cone loss | 650 | 64.24% | 53.30% |
| Hard predecessor, refresh, final loss | 650 | 63.54% | 52.95% |
| Hard predecessor, refresh, window loss, curriculum | 650 | 65.28% | 53.30% |
| Hard predecessor, refresh, window loss, extra training passes | 650 | 63.54% | 53.39% |

On the larger 128-word evaluation, the best stock-style model (two-key window, fixed +3
lag-one bias, refresh) reached 80.21% at length 12 and 58.63% at length 48. Its front at
length 48 went `1,2,2,3` and stayed at 3 through pass 96. At pass 48 the mean predecessor
weight was 0.905 and the predecessor was the argmax in 97.87% of sampled rows. The RMS state
change fell from 1.165 at pass 1 to 0.0082 at pass 48 and 0.0003 at pass 96. The state
approached a stationary solution with the wrong parity. Loss averaged over updates 1–50,
101–150, 251–300 and 551–600 fell through `0.66302, 0.39567, 0.33465, 0.30316`. The timed
updates took 53.23 seconds. In the fixed-hard-head arm 144 of the 650 allocated parameters
belong to the unused Q and K projections.
[Best stock-style diagnostics](../artifacts/round2-adjacent-five.json)

In the full diagnostic run with standard post-addition, stream RMS grew from 3.03 at pass 1
to 103.53 at pass 48 and 207.85 at pass 96. Pass-96 RMS change was 2.18. At pass 48 the mean
predecessor weight was 0.089 and the predecessor was the argmax in 9.04% of sampled rows.
The model reached 58.92% at length 12 and 52.51% at pass 48. Its 99% front reached position one.
[Stock diagnostics](../artifacts/round2-stock-diagnostics.json)

For autonomous Rule 60, the 650-parameter normalized two-key and hard-predecessor models
both matched the first iterate exactly at length 48. They scored 52.18% and 52.29% at pass 48,
respectively, and both reproduced only the first impulse row. At pass 48 mean predecessor
weight was 0.933 and 1, while RMS state change dropped to `9.40e-9` and 0.00318. Both collapsed
to a stationary or nearly stationary state on an evolving target. The hard model's mean update
loss in the first and last 50 steps was 0.66799 and 0.38406.
[Rule 60 two-key](../artifacts/round2-rule60-soft.json)
[Rule 60 hard predecessor](../artifacts/round2-rule60-hard.json)

### Reproduction

Run from the repository root with Julia 1.12.5 for the register runs and Julia 1.13.3 for the
initial stock screen. The update loop timings above exclude evaluation and most compilation.

```sh
julia +1.12.5 --project -O0 loop/scripts/trainable.jl prefix prefix-compact-tied prefix-compact-one prefix-compact-untied prefix-compact-wide
julia +1.12.5 --project -O0 loop/scripts/trainable.jl rule60 rule60-compact-tied rule60-compact-one rule60-compact-untied rule60-compact-wide
julia +1.12.5 --project -O0 loop/scripts/trainable.jl prefix prefix-three-tied prefix-three-one prefix-three-untied prefix-three-wide
julia +1.12.5 --project -O0 loop/scripts/trainable.jl rule60 rule60-three-tied rule60-three-one rule60-three-untied rule60-three-wide
julia +1.12.5 --project -O0 loop/scripts/trainable.jl prefix prefix-final prefix-final-one prefix-final-untied prefix-final-wide
julia +1.12.5 --project -O0 loop/scripts/trainable.jl prefix prefix-tied prefix-one prefix-untied prefix-wide prefix-cone prefix-cone-untied prefix-unbounded prefix-unbounded-untied
julia +1.12.5 --project -O0 loop/scripts/trainable.jl rule60 rule60-tied rule60-one rule60-untied rule60-wide rule60-final rule60-final-untied rule60-unbounded
julia --project -O0 loop/scripts/trainable.jl screen stock-curriculum stock-fast
julia --project -O1 loop/scripts/trainable.jl screen stock-after stock-before stock-normalized stock-scaled stock-refresh stock-deep stock-D16 stock-nope relative-full adjacent-soft adjacent-biased adjacent-normal hard-refresh hard-cone hard-final hard-curriculum hard-extra
ROUND2_SEEDS=1 julia --project -O0 loop/scripts/trainable.jl rule60 rule60-soft rule60-hard
ROUND2_SEEDS=1,2,3,4,5 julia +1.12.5 --project -O0 loop/scripts/trainable.jl screen register-four register-four-final register-two-final register-three-final register-unbounded register-curriculum register-extra
ROUND2_SEEDS=1 julia +1.12.5 --project -O0 loop/scripts/trainable.jl prefix adjacent-five stock-diagnostics
julia --project -O0 loop/test/runtests.jl
julia --project -O0 test/runtests.jl
```

The new Float64 finite-difference checks span full and local relative attention, hard routing,
residual scaling, register refresh, light-cone loss, and clipped and unclipped scalar feedback.
The largest relative loss-gradient error among them was `7.77e-10`.

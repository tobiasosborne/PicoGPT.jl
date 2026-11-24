# Longer context: what is inherited, what must be learned

Verdict: **promising** for staging (on two seeds, one small world), with one clear negative:
a model trained at T=8 does **not** work beyond position 8 without further training.

Everything here comes from `context.jl` (Julia, the repository's `PicoGPT` module, unchanged):

    OPENBLAS_NUM_THREADS=1 julia -t2 --project grow/context/context.jl                         # output.txt, about 7 min
    SEED=1 S32=5000 SKIP16=1 OPENBLAS_NUM_THREADS=1 julia -t2 --project grow/context/context.jl  # output_seed1.txt

## 1. The lift

In `src/embedding.jl` the positional code is fixed, not learned: `positions(D, T)` gives sinusoids
of the absolute position t − 1 in the window. No parameter has a dimension T
(`nparams(Config(T=8)) == nparams(Config(T=32))`, 9,648 here). So the lift T → 2T is

    θ ↦ θ (the identity),   Config.T ↦ 2T,

and with the causal mask the logits at positions 1..T of a length-2T input are those of its
length-T prefix. Tested on 200 windows, T = 8 → 16 and 8 → 32: max |Δ logit| = 0 (bit for bit).
The coarse-graining back is truncation of the history to its first T tokens.

A consequence: "the model run at T=16" and "run at T=32" are one curve; the first is the first
16 positions of the second.

## 2. The world, with exact floors

Token = (a, b), a and b in 1..4, V = 16, two independent processes:

- a_n = a_{n−3} with probability 0.9, else uniform (short dependency, visible at T=8);
- b_n = b_{n−12} with probability 0.9, else uniform (long dependency, invisible at T=8).

Each is a bundle of independent lag-chains with uniform stationary law, so the floor for the
token after position t of a window is exact: 2 ln 4 = 2.7726 nats for t < 3, 1.7351 for
3 ≤ t < 12, 0.6976 for t ≥ 12. The long dependency is worth 1.0375 nats per token.
Mean floor over a window: 1.5405 (T=16), 1.1191 (T=32); a sliding T=8 window can reach at
best 1.7351 everywhere.

## 3. Numbers (seed 0)

Model: D=32, H=2, L=1, F=64, 9,648 parameters. AdamW, η = 2·10⁻³, batch 32, 2,000 steps each,
all from the same initial θ. Test: 807 fresh windows of length 32. Entries are the mean excess
over the exact floor, in nats (sampling noise about ±0.03 per position).

| model | t = 1..8 | 9..11 | 12..16 | 17..32 | mean loss 1..32 |
|---|---|---|---|---|---|
| T=8 model run at T=32, no training | 0.008 | **1.207** | 1.983 | 2.242 | 2.665 |
| T=8 model, sliding window of 8 | 0.008 | −0.023 | 1.027 | 1.037 | 1.798 |
| scratch T=16 (run to 32) | 0.010 | −0.024 | 1.026 | 2.169 | 2.364 |
| scratch T=32 | 0.041 | −0.011 | 1.044 | 1.063 | 1.823 |
| staged: T=8 model + 2,000 steps at T=32 | 0.009 | −0.024 | **−0.018** | **−0.002** | 1.118 |
| floor | 0 | 0 | 0 | 0 | 1.119 |

Read:

- **Inherited for nothing:** positions 1..8, exactly. Nothing else.
- **Not inherited:** the short-range skill at new positions. At t = 9..11 the T=8 model is 1.2
  nats above a floor it reaches at t = 3..8, and worse than uniform guessing further out
  (excess above 1.04 + 1.04). The fixed sinusoidal code does not make the learned "look back 3"
  shift-invariant; the same holds for the T=16 model beyond 16. Untrained, the grown model is
  *worse* than its parent used with a sliding window (2.665 against 1.798 nats).
- **Must be learned:** the lag-12 dependency (1.04 nats), and the lag-3 rule at new positions.
- Scratch at T=16 and T=32 did not find the lag-12 rule in 2,000 steps (excess 1.03–1.06 =
  exactly the b-information left unused). They equal the sliding-window parent.

## 4. Is staging cheaper? (work = tokens processed, batch × T × steps)

Seed 0, mean excess over the 32 positions:

| tokens (thousands) | 409 | 716 | 1024 | 1331 | 1536 | 1843 | 2048 |
|---|---|---|---|---|---|---|---|
| scratch T=32 | 1.231 | 0.999 | 0.86 | 0.791 | – | – | 0.72 |
| staged (512k at T=8, then T=32) | – | 0.80 | – | 0.65 | 0.098 | 0.002 | 0.00 |

Staged reaches excess ≤ 0.02 after 1,740k tokens in total (512k at T=8 + 1,228k at T=32).
Scratch T=32 is still at 0.72 after 2,048k: it has the lag-3 rule and not the lag-12 rule.

Seed 1 (`output_seed1.txt`), scratch T=32 given 5,000 steps: the same picture. T=8 model run
long: excess 0.009 / 0.416 / 1.580 / 2.169 on t = 1..8 / 9..11 / 12..16 / 17..32; sliding
window −0.026 / 1.026 / 1.035 beyond 8.

| tokens (thousands) | 1024 | 1536 | 1843 | 2560 | 3788 | 5017 |
|---|---|---|---|---|---|---|
| scratch T=32 | 0.839 | – | 0.71 | 0.692 | 0.684 | **0.684** |
| staged (512k at T=8, then T=32) | – | 0.021 | 0.001 | – | – | – |

Staged: excess ≤ 0.02 after 1,638k tokens in total. Scratch T=32 sits on a flat plateau at
0.68 (= 21/32 × 1.04, the whole lag-12 information) from 2.3M to 5.0M tokens; it never left
it in the budget, so no finite cost ratio can be given: at least 3× on this seed, and growing.

Control (`output_control.txt`, `CONTROL=1 SEED=1 SKIP16=1`): the staged run restarts AdamW and
its warm-up, so the scratch T=32 model was continued in the same way (fresh AdamW, 2,000 more
steps). It stays on the plateau (0.681; t ≥ 12 excess 1.03). The gain is not an optimiser
restart. In that file the line labelled "staged" is this control.

## Verified, and how

- Lift exactness: tested, max |Δ logit| = 0.
- Floors: derived (independent lag-chains, uniform stationary law); the trained models sit on
  them within sampling noise where the dependency is in reach, which checks the formula.
- No new gradients were written (the module's pullbacks are used unchanged).

## Open

- Two seeds, one world, one model size, one learning rate. Scratch is stuck on a plateau, not
  slow; whether a larger model, a learning-rate sweep or a schedule frees it is not established,
  and that decides whether the gain is general or a property of this small model.
- Why staging helps: not analysed. Note the T=8 parent arrives at T=32 with a *large* loss at
  the new positions (2.2 nats excess), unlike the scratch model on its plateau; the strong
  gradient at new positions, not the inherited skill, may be what finds lag 12. Untested.
- Work is counted in tokens. Attention costs T² per window, which favours the T=8 stage further.
- A relative or rotary position code would probably make the lag-3 rule inherited at new
  positions (the failure at t = 9..11 is about absolute positions). Not tried.
- T=16 as an intermediate stage (8 → 16 → 32) not tried.

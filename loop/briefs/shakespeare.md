# Brief: does the looped specimen generalise to Shakespeare?

## Context

PicoGPT's first book ends (Chapter 9) by training the same code at scale on Tiny Shakespeare:
D=128, H=4, L=4, T=256, F=512, V=65, 810,049 parameters, `scripts/train_shakespeare.jl`, 5,000 steps,
B=32, AdamW 1e-3→1e-4, λ=0.1, clip 1, warmup 200, seed 1234, Float32, 12 threads, 36.2 min, best
held-out loss **1.5313** (main checkout `/home/tobias/Projects/PicoGPT.jl/artifacts/train.log`; read its
header for the exact settings, note STEPS=5000 overrides the script default).

A proposed third book (`loop/CONCEPT.md`; specimen in `loop/notes/window.md`, code in
`loop/src/window.jl`) loops one PicoGPT block on a toy task, with: window-2 attention, a +4
predecessor bias, a first-word marker, fresh input in the first residual
(`X₁ = X₀ + Wo·O + bo`), a normalised stream (`X = LN(X₁ + Y₂)`, unit gain, zero offset), readout
and loss at every pass, and a random number of passes per batch. The user asks whether the looped book
generalises to Shakespeare as Book 1 does. This trial settles the empirical half.

## Setup

- Data: symlink the main checkout's `artifacts/data/tinyshakespeare.txt` into this worktree's
  `artifacts/data/` (git-ignored); never write into the main checkout. Same split, same held-out tail,
  same evaluation windows (`evalset` in the training script) so losses compare to 1.5313.
- Code under `loop/` only (`loop/src/`, `loop/scripts/shakespeare_*.jl`); do not change `src/` in any way
  that alters Book 1. The specimen and the Shakespeare model should be one code path with window width
  w and predecessor bias b as configuration (w=2, b=4 specimen; w=T, b=0 Shakespeare), plus the
  stream map and supervision as options. Keep the specimen's results reproducible (its tests pass).
- **Gradcheck the tied forward and pullback in Float64 on a small configuration before any training**
  (full causal mask, positional code, every-pass loss, both stream maps).
- Run trainings one after another (12 threads). Float32 for training.

## Models (all D=128, H=4, F=512, T=256, V=65, 5,000 steps, Book 1's recipe otherwise)

1. **One block, one pass**, 215,233 parameters (16,961 outside the block + 198,272 for one block;
   verify): the equal-parameter control.
2. **Looped, the specimen's recipe at full width**: one block, full causal mask, Book 1's sinusoidal
   positions, no predecessor bias, no marker, fresh input in the first residual, normalised stream,
   readout and loss at every pass (mean over passes), number of passes sampled per step around 4
   (e.g. uniform on 2…6, or 3…5; choose, state why, keep compute comparable to Book 1's 4 blocks and
   report it). State every choice exactly.
3. **Stock tied**: Book 1's forward with `blocks[1]` applied four times and nothing else changed, loss at
   the last pass only.
4. *Optional, only if time and pace allow:* model 2 with the window of two kept; pass k then sees k+1
   characters; set its per-pass held-out loss against a smoothed (k+1)-gram model of the training
   text. Say plainly if you skip it.

## Measure, for each model

- held-out loss per pass, passes 1–16 (well beyond the trained range), on Book 1's evaluation windows,
  against 1.5313;
- sampled text at τ = 1 decoded after passes 1, 2, 4 and 8, same prompt and RNG seed for every model;
- whether the state settles: ‖X_{k+1} − X_k‖ per pass (and column-centred), and per position the pass
  after which the argmax prediction stops changing, on the held-out passage of `fig-shakespeare-attention`
  (`ATTENTION_PROMPT` in `scripts/export_book_data.jl`, the "Good morrow, neighbour" lines); which
  characters settle late;
- training wall time, loss curve, any instability (spikes, NaN, divergence).

## Deliverable

`loop/notes/shakespeare.md`: setup, the per-pass table (models × passes 1–16), the samples, the settling
measurements, timings, and a plain verdict in three parts: what carries over from the specimen
unchanged, what does not, and what a Shakespeare chapter in this book could honestly claim (and could
not). Artifacts (losses, samples, settling data) as JSON in `loop/artifacts/shakespeare-*.json`;
checkpoints in a git-ignored place or small enough to commit (state which). Copy a short report to
`loop/briefs/shakespeare-report.md`. Do not edit `loop/CONCEPT.md` (the orchestrator will). Do not commit.
Do not start Muse or Gemini agents. A negative result is a useful answer: report failures as plainly as
successes.

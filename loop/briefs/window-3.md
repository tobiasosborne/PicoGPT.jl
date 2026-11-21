# Brief: the light-cone specimen, round 3 — a *transformer* block that trains reliably

## Where we are

Read `loop/briefs/window-2-report.md` and `loop/notes/window.md` (both rounds), `loop/src/window.jl`,
`loop/briefs/prototypes-2-report.md`. The task: prefix parity ("not not to be"), looped, attention
window 2, deep supervision, exact light-cone floor. Results so far form a ladder:

- a **2-parameter gain register** (fixed predecessor head, s_t ← tanh(g(x_t)·s_{t−1})): 5/5 seeds,
  floor to 0.1 %, 100 % at T = 64, K = 2T; rule 60 matches Lucas through pass 96;
- **17/33-parameter ReLU registers** (fixed head, learned MLP, clipped scalar): extrapolate 5/5 but
  miss the loss floor;
- the **stock block** (learned attention, both LNs, MLP, embedding, readout), window 2, first-token
  marker, normalised stream LN(B(X)+X₀) or fresh-input corrections, mixed T ∈ 4…12 and
  K ∈ T−1…2T: reaches the floor and extrapolates exactly, but in **1 of 5 seeds** (270 params at
  D = 4, F = 16; also 444 and 582).

A book about a looped *transformer* needs the last rung to be reliable: it is meant to train live
in a browser. This round's single goal: **≥ 4 of 5 seeds** for a block that is recognisably the
PicoGPT block (learned attention over the window, LayerNorm, ReLU MLP, embedding, readout), with
the floor reached within 1 % and 100 % at T = 32 and 64 with K = T and K = 2T, training ≤ ~30 s.

## Ideas to try (diagnose why seeds fail first: where do the 4 failing seeds get stuck, and what
does their attention / state look like?)

- A beginning-of-sequence token instead of the additive marker (standard ML practice; it also
  gives position 1 a real predecessor and removes the singleton boundary problem).
- Initialisation that starts near the right dynamics without hand-building it: Wq = Wk = 0
  (uniform attention over the window, which already suffices: (v_t + v_{t−1})/2 plus the residual
  recovers the predecessor); Wo, W2 small or zero (the loop starts near the identity); readout
  near zero.
- Optimiser: lower peak rate with more steps, larger batches, longer warmup; a curriculum that
  grows T and K together; gradient clipping radius.
- Width: D ∈ {4, 6, 8}, F ∈ {8, 16, 32}, H = 1.
- Which stream map is most reliable: LN(B(X)+X₀), X₀ + (B(X) − X), or another bounded form; keep
  whichever is simplest to state if they tie.
- Intermediate rungs, if the full block stays unreliable: frozen uniform attention (no Wq, Wk:
  "a transformer with fixed attention"), no LN₁, etc. Report the success rate of each rung so that
  the ladder from 2 parameters to the full block is documented with seed counts.

Each change must be small and stateable in one sentence; report each with its effect on the seed
success rate (a table of rungs × success rate is the core deliverable).

## Also, as an independent check of a claim by another agent

Astra (`loop/notes/derivations.md` §7, last paragraph) claims: for Rule 60 scored at passes 1 and 2
with equal weight on uniform 8-bit words (zero left boundary), a horizon-blind one-pass predictor
has CE infimum (7/16) ln 2 and accuracy ceiling 25/32. Verify by exhaustive enumeration with your
own code; report agreement or the correct numbers.

## Deliverables

For the best configuration: the parameter count, the exact list of changes from the stock PicoGPT
block and loop (in lines of code), the per-seed results, timing, and an exported artifact
`loop/artifacts/window-specimen.json` holding θ, the training curve, the exact floor, and a
per-pass trace on one T = 16 word (the stream, the attention weights, the predicted probabilities,
and the terminal covector per pass) for a future figure. Add a "Round 3" section to
`loop/notes/window.md`; copy your report to `loop/briefs/window-3-report.md`.

## Rules

You own `loop/src/window.jl`, `loop/test/window.jl`, `loop/scripts/window_*.jl`,
`loop/artifacts/window-*.json`, `loop/notes/window.md`. Edit nothing else. New variants get gradcheck
tests; keep all tests passing. Do not commit. Do not start Muse or Gemini agents.

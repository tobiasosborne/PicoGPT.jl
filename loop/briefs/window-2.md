# Brief: the light-cone specimen, round 2 — make it extrapolate and train reliably

## Where we are

Read `loop/briefs/window.md` (round 1 brief), `loop/briefs/window-report.md` and
`loop/notes/window.md` (its results), and `loop/src/window.jl`. Summary: prefix parity, looped
post-injection `X[k+1] = B(X[k]) + X[0]`, attention window 2. The exact deep-supervision floor
21 ln 2/64 = 0.227438919 nats at T = K = 8 was confirmed by enumeration and reached to 8·10⁻⁶ by
one 582-parameter seed (1 of 5). That model is exact inside the cone and undecided outside it at
T = 8, but at T = 32 / 64 it is only 72 % / 62 % right: its 99 % front stops at position 15, and it
is overconfident at unseen positions. A specimen must extrapolate: a local rule learned at T = 8
should work at every length and every pass count.

## Diagnoses to test (orchestrator's hypotheses; confirm or refute each with a number)

1. **Absolute positions.** The sinusoidal code makes the learned rule position-dependent, and
   positions > 8 were never seen. With a window of two, no positional code is needed: even a
   *uniform* two-key attention gives (v_t + v_{t−1})/2, and the residual carries v_t, so the
   predecessor is recoverable. Try **no positional code** (NoPE; the boundary at position 1, which
   reads only itself, is the only place that breaks translation symmetry). Prediction: the
   learned map is then the same at every position, so length extrapolation should come free.
2. **Stream drift over passes.** Post-injection adds X₀ every pass, so |X_k| grows like k and,
   at 64 passes, the re-added input swamps whatever the state carries (LN only rescales). Compare
   (a) post-injection as now, (b) a bounded update in equilibrium-model form,
   `X[k+1] = X[0] + (B(X[k]) − X[k])` (the block's two corrections added to the fresh input,
   not to the old stream), (c) LayerNorm (no learned parameters) of the stream after each pass.
   Measure |X_k| and |X_{k+1} − X_k| (column-centred) per pass up to 128 passes.
3. **Training distribution.** Train with mixed lengths (e.g. T uniform in 4…12, or T = 8 with
   random crops) and a random number of passes K ∈ [T − 1, 2T] per batch, deep supervision over all
   passes (the floor formula generalises: per pass k, ln 2·#{t > k + 1}/T; verify it for mixed T and
   K by enumeration and report the exact floor of whatever distribution you train on). Extra
   passes beyond T − 1 teach the state to be a fixed point.
4. **Seed reliability.** Only 1 of 5 seeds trained. Try the fixes above first; then learning rate,
   steps, batch, F (e.g. 8–32), D (4–8). Report the fraction of seeds that reach the floor.

## Success

At least 4 of 5 seeds from random initialisation: training loss within 1 % of its exact floor;
then, with no retraining, accuracy ≥ 99 % at T = 32 and T = 64 with K = T passes, and still ≥ 99 % at
K = 2T. Training ≤ ~30 s per seed. Find the smallest such model; report its parameter count.

For the winning configuration, also report: the attention pattern (uniform, predecessor, self?);
whether the state converges to a fixed point after the front passes (centred |X_{k+1} − X_k| per
pass); the per-pass accuracy cone at T = 64 (a matrix position × pass; save it to the artifact, a
figure might be drawn from it later); a one-pass control and an untied L = K control at the same
setting with their exact floors; and the terminal-covector light cone as before. If time remains,
the same configuration on rule 60 (autonomous, window 2), compared with Lucas' theorem to 4× the
trained number of passes.

## Rules

You own `loop/src/window.jl`, `loop/test/window.jl`, `loop/scripts/window_*.jl`,
`loop/artifacts/window-*.json`, `loop/notes/window.md` (add a "Round 2" section; keep round 1).
Another Sol agent is concurrently working on `loop/src/Loop.jl`, `loop/scripts/` files without the
`window_` prefix and `loop/notes/prototypes.md`: do not edit those. Every new loop variant gets a
gradcheck test. Do not commit. Do not start Muse or Gemini agents. Report back (copy to
`loop/briefs/window-2-report.md`): which diagnosis was right, the winning configuration and its
numbers, the seed success rate, and what still fails.

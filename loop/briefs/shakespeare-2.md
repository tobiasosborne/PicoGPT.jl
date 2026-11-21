# Brief: Shakespeare trial, two further runs (A: without the confound; B: window of two)

Read `loop/briefs/shakespeare.md` (round 1 brief) and `loop/notes/shakespeare.md` (its results: your
own previous work, code in `loop/scripts/shakespeare_*.jl`). Same data, evaluation windows, recipe
(5,000 steps, B=32, AdamW 1e-3→1e-4, warmup 200, λ=0.1, clip 1, seed 1234, Float32, 12 threads),
D=128, H=4, F=512, T=256, 215,233 parameters. Runs one after the other. Power profile: leave it as it
is (do not change system settings); record wall times as before. One seed each; say so.

## Run A: the looped recipe without the confound

As the round-1 looped model (fresh input in the first residual, normalised stream, loss at every
pass, K uniform on 3…5, full causal attention, sinusoidal positions, no bias), except the stream
starts from the **full input** (token embedding plus position at every column), not from the first
column only. Report the same per-pass held-out table (passes 1–16), the settling figures on the
"Good morrow, neighbour" passage, and the samples at τ = 1 after passes 1, 2, 4, 8 (same prompt and
RNG as round 1). Answer: with the confound removed, what do the stabilisers cost against stock tied
(1.654) and one pass (1.698)?

## Run B: the window of two on Shakespeare

As run A but with the window mask (each query sees only j ∈ {i−1, i}), no predecessor bias,
full-input start, the stabilisers, loss at every pass. Choose the trained pass range so the cone reaches
a useful depth (e.g. K uniform on 1…8, or fixed K = 8 with loss at every pass) and state the choice
and the compute it implies. Then:

- **Held-out loss per pass**, passes 1–16.
- **The n-gram reference ladder.** Be exact about the offset: with a full-input start, the readout
  after pass k at position t can depend on characters t−k … t, i.e. **k + 1 characters of context**,
  so the matching n-gram has order **n = k + 2** (context k + 1 plus the predicted character). Verify the
  offset against the code. For each k = 1…16 (as far as is meaningful), fit an n-gram model of that
  order to the training text with a standard smoothing: interpolated Kneser–Ney (preferred) or
  Witten–Bell; state exactly which estimator and variant. Tune its free constants (discounts, or
  interpolation weights) on a held-out slice of the **training** text only, never on the held-out tail;
  then refit on the training text if that is your protocol, and say so. Evaluate on Book 1's same
  held-out windows (same 256 × 256 predictions; for the first characters of a window, use the shorter
  context that exists, and match whatever the model does there). The note must say the n-gram ladder is
  a reference line, not an exact floor: a smoothed estimator is neither a lower bound nor the Bayes
  risk, and the transformer also sees absolute positions.
- **Check the cone is real** in the trained model: on a few held-out windows and positions, change a
  character more than k + 1 places back (i.e. outside t−k … t) and confirm the pass-k logits are
  unchanged to rounding; and change one inside, to show it does matter. A perturbation test is
  enough; a causal-Jacobian check is better if cheap.
- **Samples** at τ = 1 after passes 1, 2, 4, 8.
- Settling figures as for the other models.

Answer plainly: pass by pass, does the looped window model **follow, lag or beat** the n-gram ladder?

## Deliverable

Extend `loop/notes/shakespeare.md` with sections for runs A and B (keep round 1 as it is), add both to
the per-pass table, and revise the verdict if the results change it. Artifacts in
`loop/artifacts/shakespeare-{fullstart,window}.json` (and the n-gram ladder, e.g.
`shakespeare-ngram.json`). Gradcheck any new code path in Float64 before training. Copy a short report to
`loop/briefs/shakespeare-2-report.md`. Do not edit `loop/CONCEPT.md`; do not commit; do not start Muse
or Gemini agents.

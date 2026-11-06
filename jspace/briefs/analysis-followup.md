# Follow-up brief: the scale of J, and a training-time export

Read `jspace/briefs/analysis.md`, your `jspace/briefs/analysis-report.md`, `jspace/artifacts/README.md`
and `jspace/src/JLens.jl` first. Everything below is a change to that work; keep the structure,
names and conventions you established unless told otherwise. Stay inside `jspace/`; no commits.

## 1. Adopt the scale-free convention for J (and keep the literal one as a lesson)

Your finding is right and will be in the text: with uniform weight on pairs, J_L = 2/(T+1)·I, and
LayerNorm's fixed ε then makes the final J-lens differ from the model (0.19 in probability for
Shakespeare). The book should define the averaged Jacobian so that nothing depends on ε:

    J_ℓ := mean over sources t (and windows) of the SUM over targets t′ ≥ t of K_{t′t}

i.e. your `averaged` times (T+1)/2 (the same matrix up to the scalar, since the pair weighting is
uniform). Then J_L = I exactly, the J-lens at depth L is the model's own readout, and the J-lens
vectors v_a = rows of Wout J have the natural scale (at depth L they ARE the rows of Wout).

Make this the exported convention everywhere: `jacobians.averaged` (document the factor; keep
`self` and `future` as they are: `self` already sums to I at depth L, and `averaged = self +
future` under the new convention — assert that), `blocks` and `per_window_attention` (same
factor so they remain comparable), `lenses.vectors`, all readouts and metrics, the prompt grid, the
J-space decompositions (cone fits are invariant, but the coefficients' scale changes: recompute,
do not rescale by hand), the interventions (steer with the rescaled v_a and the same α grid; the
swap and ablation are scale-invariant but recompute anyway), and the structure statistics (unit
directions: unchanged, but recompute for one consistent file). Update `sanity` so the final-lens
test is the exact one (≤ 1e-9 before rounding), and turn the "broken" test into a passing test
of the new identity. Keep the test that documents the old discrepancy, now phrased as a check
that the **literal paper-weighted** lens differs by the amount it does.

Keep the literal paper-weighted lens results as an explicit, separate, clearly labelled set so the
text can show the lesson: for the specimen `lenses.literal.j_probs[a,n,j]` and
`lenses.literal.final_max_probability_error`; for Shakespeare `lenses.literal.metrics[j]` (same
fields as `metrics`) and `lenses.literal.final_max_probability_error`. Nothing else literal.

Re-export both files, rerun the tests, and update the README and the report (replace the numbers
that changed; say in one paragraph what the convention is and why). Note in the report whether
the Shakespeare J-lens still trails the logit lens at intermediate depths under the new scale, and
by how much; both answers are fine, and the text will report whichever is true.

## 2. A training-time export for the frontispiece (small, exact)

`jspace/artifacts/specimen-jlens-training.json`: train the specimen exactly as
`scripts/export_specimen.jl` does (same θ0 = `theta0`, same recipe, so that step 200 reproduces
`train.theta_final` to roundoff; assert that), and after every update k = 0 … 200 record, at
every depth, the J-lens vectors v_a (V×D) under the new convention and the J-lens probabilities of
the 7 characters at the last position of window 1 — that is `vectors[a,q,j,k]` and
`last_probs[a,j,k]`, plus `loss[k]`, `depth`, `vocab`, conventions and provenance. Five significant
digits; keep it under 1 MB. A plotter-style trace will be drawn from it: every v_a wandering
through training.

## 3. Report

Append a dated section to `jspace/briefs/analysis-report.md` with: the convention change and its
effect on every table (reprint the tables that changed), the new sanity values, test output,
timings, and the training export's description. Your final message summarises it.

# Brief: fresh-seed reliability of the frozen 270-parameter specimen

## Why

`loop/CONCEPT.md` recommends the 270-parameter light-cone specimen (Round 3 in `loop/notes/window.md`,
`loop/briefs/window-3-report.md`, configuration `anchor-boundary-start` in
`loop/scripts/window_round3.jl`). Its reported 4/5 success rate was measured on the same five seeds
and the same long test words the recipe was selected on (final review, `loop/notes/review-concept.md`
§1). The user's decision turns on the true rate. Scope is exactly the two measurements below.

## Steps, in this order

1. **Freeze.** Write the recipe exactly as reported (shape, stream map, window, lag bias +4, start,
   marker, schedule, steps, batch, distribution of T and K, loss, AdamW constants, init) into a
   frozen configuration that a new script `loop/scripts/window_fresh.jl` reads, with no tunable
   argument except the seed list and the lag bias. Fix in advance: training seeds **1001–1030** (never
   used in any screen; check the artifacts and scripts to confirm, and choose another unused block if
   they were), and test words drawn from an RNG seeded with a value never used before (record it),
   256 words at each of T = 32 and T = 64, evaluated at K = T and K = 2T. Then **commit only the
   frozen configuration and the script** on the current branch, with a message saying it is frozen
   before measurement (end the message with
   `Co-Authored-By: GPT 6.2 Sol <noreply@openai.com>`), before running any training.
2. **Measure, lag bias +4.** Train all 30 seeds. Success criterion, unchanged: exhaustive mixed loss
   within 1 % of 0.161350952 nats, and 100 % accuracy on all four long rollouts. Report the success
   count, a 95 % Clopper–Pearson interval, and for failing seeds how they fail (loss gap; front
   position at the last pass; stalled near the boundary, wrong fixed point, drifting stream, or
   correct-but-unconfident at the front, using the Round 3 diagnostics), plus timing per seed.
3. **Lag bias 0**, same seeds, same everything else (window of two kept). Same report, plus the
   learned predecessor attention weights at length 64, last pass (mean, min, max per seed), for
   successful and failing seeds separately. **Lag bias +2** too if time per run allows (it should:
   about 10 s per seed).
4. Write the results to `loop/artifacts/window-fresh-*.json`, append a section "Fresh seeds" to
   `loop/notes/window.md` with the numbers and the exact commands, and copy a short report to
   `loop/briefs/fresh-seeds-report.md`. Do not edit `CONCEPT.md` (the orchestrator will); do not
   commit the results (the orchestrator will).

## Rules

Change nothing in the recipe after the freeze commit; if the script has a bug that forces a change,
say so explicitly and explain. Edit only `loop/scripts/window_fresh.jl`, the frozen configuration
file, `loop/artifacts/window-fresh-*.json`, `loop/notes/window.md` (append only) and the report.
Do not start Muse or Gemini agents.

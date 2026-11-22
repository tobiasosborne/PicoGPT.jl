# Brief: independent check of `loop/notes/written-vs-looped.md`

You are the second, independent reviewer; another Astra agent wrote the note. Check every claim
against your own derivation and, where cheap, your own code (not the author's `loop/scratch/wvl/`):
the three-way comparison (task variant, token format, exact loss floors, the hand-built 270-parameter
solution, block applications and written tokens, the exchange rate), the error-propagation law
(1 − (1 − 2ε)^t)/2 and its loop counterpart, the gate (closed gate = one-pass model, gradient at
initialisation, Jacobian/contraction, relation to LayerNorm's unit eigenvalue), steps versus distance
and the chained-lookup bound incl. pointer doubling, and the Shakespeare sandwich (810,177 parameters,
eight block applications per token, the pullback). Context: `loop/briefs/written-vs-looped.md` (the
brief it answered), `loop/notes/derivations.md`, `loop/notes/inkling.md`, `BRIEF.md` §1–3.

Deliverable `loop/notes/check-wvl.md`: verdict per claim (correct / incorrect with correction /
correct with caveats), corrections first, ranked; copy a short summary to `loop/briefs/check-wvl-report.md`.
Battery rule: only Julia checks that finish in seconds on one or two threads (code in
`loop/scratch/check-wvl/`); no training. Edit nothing else. Do not commit. Do not start Muse or Gemini
agents. Do not change system settings.

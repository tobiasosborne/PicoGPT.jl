# Concept revision — report

Revised `loop/CONCEPT.md` for the five approved changes of 22 Nov, from
`loop/notes/written-vs-looped.md` and `loop/notes/inkling.md` as they stand. The analysis's
numbers are marked "pending check" once, in the headnote at the top of `CONCEPT.md`. No number
was added that is not in those two notes or in the 21 Nov document itself, and no earlier result
was deleted: the diff to `CONCEPT.md` is additions only, with superseded claims and plans marked
"superseded by the revision of 22 Nov: …" on the spot.

What changed, by section:

- **Headnote** — the revision of 22 Nov stated, its five changes listed, the pending-check mark.
- **§1** — added "steps versus distance" (the window of two turning dependent steps into a
  visible distance, with the KL(w − 1) reach bound and the doubling contrast), the negation task
  three ways (one pass / written feedback / tied passes: end-only floors, T − 1 dependent calls
  both exact ways, column and scratch counts, the streaming protocol, the error product law, the
  hand-set 270-parameter witness and its margins), the gate that starts closed (restart plus
  message m_k = X_{k−1} − S under one scalar g from zero; exact single-call identity at g = 0,
  the 21 Nov recurrence at g = 1, the contraction scope, the vanishing population opening signal
  and the one checked example), and a superseded note shrinking the fixed point to the
  Huginn / small-recursive-model (HRM/TRM, DEQ) aside and replacing the Shakespeare plan.
- **§5** — a revised 12-item arc (writing it down; steps versus distance; the gate; finite
  unrolling in the main line with the equilibrium formula in an aside; Shakespeare in Inkling's
  layout as the closing chapter), followed by the 21 Nov arc kept verbatim and marked where
  superseded.
- **§6** — items 1–5 untouched; added item 6 (restart and gated message, forward and pullback,
  one new parameter, its exact scope) and item 7 (the written schedule as a caller protocol, not
  a model change), a pointer to the sandwich's separate departures, and a superseded note on
  "zero new parameters".
- **§7** — five new dictionary rows: writing it down / chain of thought, the message, the gate,
  the blocks that run once, steps versus distance.
- **§12** — all 21 Nov runs kept; added the sandwich (Book 1's four blocks, first and last once,
  middle two three times, per-channel gated restart from B₁; 810,177 parameters, eight block
  applications, exactly Book 1 at g = 0; forward, accounting and pullback; where it differs from
  production Inkling) and a revised "reading for the book", the earlier reading marked superseded
  as a plan.
- **§13 (new), "Runs for mains power"** — the analysis's four runs in order, with its conjectured
  allowances and the 21 Nov timings they rest on: written and latent parity (5–15 min); gate
  versus stream normalisation, 2 × 2, five then 30 seeds per cell (20–60 min); adaptive hops and
  doubling (10–30 min); the Shakespeare sandwich, closed-gate and finite-difference checks first,
  then equal-update and block-matched comparisons (2–3 h per 5,000-update run plus controls,
  against the 72–90 min four-application Power Saver runs; specimen runs ~10 s per seed).

Not done, as instructed: no computation was run (battery), nothing was committed, and only
`loop/CONCEPT.md` and this report were written. Everything the revision asserts beyond the
21 Nov results is either a proved identity of the analysis, a reported property of Inkling from
the source note, or an explicitly conjectured run in §13; the gated specimen and the sandwich are
designs, and the document now says so wherever they appear.

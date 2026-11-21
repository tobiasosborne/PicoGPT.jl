# Brief: final review of the concept document for the looped-transformer book

You are the ultimate reviewer. `loop/CONCEPT.md` proposes a third PicoGPT book and its specimen.
It will go to the project owner (a physicist) as the answer to "what would a book on the tiniest
looped transformer look like, and is it worth building?". Check it against the evidence, not
against its own prose: `loop/notes/window.md` (all rounds), `loop/briefs/window-3-report.md`,
`loop/notes/prototypes.md`, `loop/notes/derivations.md`, `loop/notes/audit.md`, the artifacts in
`loop/artifacts/` (JSON) and the code in `loop/src/window.jl`. Notation: `BRIEF.md` §1–3.

Report, in `loop/notes/review-concept.md`:
1. Every number or exact claim in CONCEPT.md that is wrong, unsupported by an artifact or script, or
   misattributed (e.g. a result from the 2-gain cell quoted as if from the 270-parameter block). Give
   the correction with its source.
2. Every mathematical statement that is wrong or overstated (light-cone floor, one-pass floor and its
   "any width" claim, Lucas, the two-horizon control, the LN eigenvalue-1 claim, the Neumann series,
   the Ising transfer-matrix reading, the KSW citation).
3. Whether §6 (the minimal change to PicoGPT, "about 15 lines forward and 6 back") describes the
   specimen's actual code path correctly; count against `src/` and `loop/src/window.jl`.
4. Whether the verdict ("worth building, with one stated cost") follows from the evidence, and the
   single most important risk the document understates, if any.

Do not edit CONCEPT.md or any other file except `loop/notes/review-concept.md`,
`loop/briefs/review-concept-report.md` and scratch code in `loop/scratch/review/`. Do not commit.
Do not start Muse or Gemini agents. Keep the report short: corrections first, ranked by importance.

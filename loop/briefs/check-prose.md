# Brief: check and correct the mathematics in the revised prose and concept

Three texts were written from `loop/notes/written-vs-looped.md` (the analysis), which an independent
review has since checked: `loop/notes/check-wvl.md` (read its corrections first: the one-pass ln 2 is a
population minimum; the intended Ising update is preferred only below a 50 % flip rate; the LayerNorm
bound of one applies to coefficients in the witness basis; and the rest). The texts:

- `loop/book/00-prologue-sample-2.md`, `loop/book/sample-written-vs-looped.md` (prose samples);
- `loop/CONCEPT.md`, the parts revised on 22 Nov (§1, §5, §6, §7, §12 sandwich, §13 runs), and its
  "pending check" headnote. Also `loop/notes/inkling.md` for statements about Inkling 2 Mini.

Check every mathematical statement and every number in these texts against the analysis, the review and
(for Inkling) the source note. Then **apply minimal corrections in place**: change only what is wrong or
overstated, keep the authors' wording and voice otherwise, do not restyle. Replace the "pending check"
headnote in CONCEPT.md with one line saying the analysis was independently checked (cite
`loop/notes/check-wvl.md`) and that its corrections are applied. Leave `{{TODO: …}}` markers in place.

Deliverable: the corrected files, and `loop/notes/check-prose.md` listing each correction (file, the old
statement, the new one, the reason), plus statements you checked and found correct in one line each, and a
short summary in `loop/briefs/check-prose-report.md`. Battery rule: Julia only for checks that finish in
seconds. Edit nothing else. Do not commit. Do not start Muse or Gemini agents. Do not change system
settings.

# Brief: two sample chapters for the revised looped book (prose)

The project owner wants to feel the tone of a third PicoGPT book, on the tiniest looped transformer,
after a revision. Read for voice: `BRIEF.md` (§0 reader and tone, §7 conventions, §12), Book 1's
`book/chapters/00-prologue.md` and one later chapter (e.g. `book/chapters/04-surprise.md` or whichever
exists for loss/surprise), and the earlier sample `loop/book/00-prologue-sample.md`. Read for content:
`loop/notes/written-vs-looped.md` (the analysis: the negation task three ways, the exchange rate between
passes and written tokens, error propagation, the gate, steps versus distance), `loop/notes/inkling.md`
(the real model this is measured against), and `loop/CONCEPT.md` §1–2 (the specimen's measured numbers).

Write two samples, each 1,000–1,400 words, pandoc markdown in the conventions of BRIEF §7 (a `::: claim`,
a `::: names` table, prose, at most two `::: aside`, at most one `::: {.figure #fig-…}` placeholder with a
caption that says exactly what the figure shows):

1. `loop/book/00-prologue-sample-2.md`, a new prologue that opens on the real question, *think inside, or
   write it down?*, and keeps the two-word language `be`/`not` (the meaning of a sentence is the parity of
   its `not`s). It introduces the loop, the window of two as a device that turns steps into distance, and
   what the reader will see.
2. `loop/book/sample-written-vs-looped.md`, the chapter on writing versus looping: the same negation task
   shown three ways (one pass with nothing written; one pass that writes intermediate answers and reads them
   back; the loop with nothing written), with their exact floors and costs, the exchange rate, when writing is
   the cheaper of the two (say it honestly), and the error propagation (the Ising chain again).

Use **only** numbers and mathematical statements that appear in those notes, stated as they are there; where
you need one that is not there, write `{{TODO: …}}`. Plain, exact, dry; every term defined in the reader's
terms; three names; no hype. Write only those two files and a three-line summary in
`loop/briefs/prose-2-report.md`. The laptop is on battery: no heavy computation. Do not commit. Do not start
any subagents. Do not change system settings.

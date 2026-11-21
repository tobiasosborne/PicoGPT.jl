# Brief: two pages of sample prose, the prologue of a third PicoGPT book

The project owner wants to *feel the tone* of a possible third book before deciding to build it.
Write about two pages (900–1300 words) of the prologue, pandoc markdown in the conventions of
`BRIEF.md` §7 (a `::: claim`, a `::: names` table, prose, one `::: aside`, one figure placeholder
`::: {.figure #fig-cone}` with a caption describing what the figure shows).

Read first, for voice and conventions: `BRIEF.md` (§0 reader and tone, §7, §12), `book/chapters/00-prologue.md`
(the first book's prologue: match its register exactly: plain, exact, dry, no hype, every term defined in
the reader's terms, three names), then `loop/CONCEPT.md` (the concept: what the book is, its specimen, its
numbers). Use **only** numbers that appear in `loop/CONCEPT.md` (§1, §2), stated as they are there; do not
invent any. If you need a number that is not there, write `{{TODO: …}}` instead.

The prologue should: say what a looped transformer is (one block, the same parameters, applied again and
again to its own output; the reader already knows the block from the first book); introduce the specimen
(the two-word language `be`/`not`, the meaning of a sentence as the parity of its `not`s, the window of two,
270 parameters); introduce the light cone and the floor ℓ*, in the manner the first book introduced the
entropy floor; and end on what the reader will see (the cone filling in, pass by pass). The working title
is *Not Not To Be*.

Write it to `loop/book/00-prologue-sample.md`; copy a three-line summary to `loop/briefs/prologue-sample-report.md`.
Edit no other file. Do not commit. Do not start any subagents.

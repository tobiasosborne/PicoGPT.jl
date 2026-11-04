# Brief for chapter authors (prose + figures)

You own a set of chapters end to end: the prose in `book/chapters/NN-slug.md` and every
figure for it in `web/src/figures/NN-slug/`. Read these first, in this order:

1. `BRIEF.md`. It is binding, especially §1 (organising idea), §2 (notation), §7 (source
   conventions), §8 (arc) and §12 (findings the prose must honour).
2. `DESIGN.md`. It is the visual system and figure contract, and it is binding.
3. `web/README.md`. Conventions for adding chapters and figures.
4. `book/chapters/05-attention.md` and `web/src/figures/05-attention/`. This is the finished
   reference chapter. Match its voice, density, page architecture and figure craft, then
   exceed them.
5. `book/figures.md`. The figure catalogue. Treat it as a strong proposal, not a cage.
6. `book/notes/derivations.md`. The verified mathematics. Every claim you make must agree
   with it.
7. `src/*.jl`. The real code, which you quote with `::: {.code file=… fn=…}`. Never paste code
   by hand. Build with `BOOK_STRICT=1` so a missing excerpt fails.

## The reader and the voice

The reader is a BSc physics or maths student who knows linear algebra, basic analysis and
basic probability. The maths is easy for them; the jargon is the obstacle.

The model for the voice is Hohm, arXiv:2605.06394:
- Define every object as a map between named spaces.
- Show index form and matrix form side by side.
- Give each object three names, in a `::: names` box.
- Write dry, exact, occasionally witty prose. No hype, no "magic", no anthropomorphism
  beyond what you define.
- Every ML term is defined, in the reader's own language, before it is used.
- Prefer finite sets and concrete numbers from the specimen. Never type a number that a
  figure could compute. Never invent numbers.

Each chapter opens with `::: claim`, uses sidenotes (`::: aside`) generously for history and
"in the code" pointers, and ends with exercises: 3 to 6, a mix of pencil-and-paper ones and
"try it in the figure" ones.

## Figures

Every figure answers ONE question. Prefer direct manipulation over controls: drag the thing
itself. Each figure is a pure function of the specimen and its own state, has a beautiful
static print state, respects reduced motion, and reads its data only through
`web/src/lib/model/specimen.js`.

The first time a glyph appears, give it a "how to read" key. Verify everything visually:
- run the dev server;
- take Playwright screenshots at desktop and mobile widths, in day and night modes, and in
  `?print` mode;
- iterate until it is genuinely beautiful and clear.

Look at the images. "It builds" is not done.

## Coordination (several authors work in parallel in the same checkout)

- Edit only your own chapters, your own figure folders, and the shared primitives assigned
  to you.
- Any other shared file (the specimen adapter, the Lua filter, CSS, other primitives) may
  only be *extended*, never changed in behaviour. Keep each extension minimal and say what
  you added in your report.
- Before your final report, make sure `npm run build` passes from `web/`, so you don't break
  the others.
- Do not commit; the orchestrator commits.

## Report

Report: the chapters and figures you built, with screenshot paths; extensions to shared
files; open issues; and anything the orchestrator must decide.

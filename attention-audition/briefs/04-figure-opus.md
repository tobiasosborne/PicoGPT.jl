# Brief 04 — one new figure: the mixture against its mean (Opus, design)

PicoGPT is a GPT in Julia plus a book (web + print, one source) for BSc physics/maths students.
The working directory is the repository root, a git worktree. You are the designer of **one new
figure** for a draft revision of Chapter 5 (Attention). Read first:

1. `DESIGN.md` — binding: two inks (value indigo / covector cinnabar; this figure is all forward,
   so value ink plus near-black and greys), every mark a number, direct labels, no chrome, print
   first, the figure contract (§10) and "adding figures" (§11).
2. `BRIEF.md` §2 (notation) and §4 (the specimen).
3. `attention-audition/README.md` and `attention-audition/edition.js` — the audition edition this
   figure lives in; build it with `node print/build.mjs --edition=attention-audition --proof`, and
   on the web with `BOOK_EDITION=attention-audition npm run dev` in `web/`.
4. `attention-audition/notes/spine.md` §5–6 (the mean-field step and the geometric mixture) and
   `attention-audition/notes/numbers.md` (the specimen measurements) — the mathematics the figure
   shows. Data: `attention-audition/artifacts/mixture.json` (figure readers included), delivered to
   figures by the edition's `artifacts()` (check how `getContext('artifacts')` reaches it; the
   jspace edition's figures in `jspace/figures/` do the same thing).
5. The existing chapter-5 figures `web/src/figures/05-attention/` and the primitives in
   `web/src/figures/primitives/` (`EnergyLadder.svelte`, `Arcs.svelte`, `TriangleDots.svelte`,
   `HowToRead.svelte`): reuse the book's glyphs and craft, especially the distribution glyph of
   Chapter 3 (`web/src/figures/03-energies/fig-energy-ladder.svelte`).

## The figure

Id `fig-attention-meanfield`, file `attention-audition/figures/05-attention/fig-attention-meanfield.svelte`.

**The one question it answers:** *for a real reader of the trained specimen, does attention predict
like a mixture of the per-source predictions, or like their geometric mixture?*

The argument in the chapter: an honest probabilistic model would choose one source $j$ with
probability $A_{ij}$ and predict from that source alone ($p_j$, the prediction with the head
output replaced by $v_j$); averaging those gives the arithmetic mixture $\sum_jA_{ij}p_j$.
Attention instead averages the vectors and predicts once. With a readout whose logits are affine
in the attention output, that equals the normalised geometric mixture $\prod_jp_j^{A_{ij}}/Z$
exactly; in the real model it is only close. The specimen says: across 80 readers, mean total
variation from the actual prediction is about 0.12 for the arithmetic mixture and 0.003 for the
geometric one (read the real numbers from the data; never type them).

The natural plate, for the spread reader of `numbers.md` (window 7 `or not t`, position 6, head 1;
use the data's own record): the reader and its sources with their weights; the per-source
predictions $p_j$ as small distributions (small multiples); then, side by side, the arithmetic
mixture, the geometric mixture and the model's actual prediction. The eye should see at once that
the actual prediction sits with the geometric mixture and not with the arithmetic one, and *why*:
a geometric mixture lets one source vote a character out. A contrast reader (the near-copy, where
all three agree) may appear small, if it helps; it is optional. Decide the layout yourself; this
paragraph is a suggestion, not a spec. One excellent static plate is the goal. Interaction is
optional (e.g. choose the reader among the figure readers, or drag along the 80 readers); the
print state must carry the whole argument.

Give a caption draft (2–4 sentences, book voice, in the book's notation) in your report; the
writer will place it. If the glyph is new to the book, give it a "how to read" key.

## Verify

Look at the result: web desktop and mobile width, day and night, `?print`, and the PDF proof page
of the audition edition. Iterate until it is genuinely beautiful and clear. "It builds" is not done.
The current audition chapter is a copy of the old chapter; to see your figure in print, you may
temporarily add `::: {.figure #fig-attention-meanfield}` with a caption at the end of section
"Scores, and why a softmax" in `attention-audition/book/chapters/05-attention.md`, but a writer is
about to replace that file: put your test block in, build, look, and **remove it again** before you
finish (and say so), or simply test through the web route.

## Rules

- Write only in `attention-audition/figures/05-attention/` (and, if the data delivery needs it, the
  `artifacts()` function of `attention-audition/edition.js`; say what you changed). Do not change
  `web/src/`, `print/`, `book/`, or the notes. Do not commit.
- Every mark is a real number from `mixture.json`. If the data lacks something you need, compute it
  in the figure from what is there, or say what is missing; do not invent.
- Do not start Muse or Gemini subagents.

Report: file(s), screenshot/proof paths you looked at, the caption draft, any edition.js change.

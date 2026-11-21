# Brief 07 — review the draft chapter as its intended reader (Muse)

Working directory: the repository root of PicoGPT.jl (a git worktree). PicoGPT is a GPT written
from scratch in Julia and a book about it. Its reader is **a BSc physics or mathematics student**
who knows linear algebra, basic analysis and basic probability, and Gibbs distributions — and no
machine learning. The book's promise: the maths is easy; the jargon is the obstacle, and it
removes it.

Read the draft of **Chapter 5, Attention**, as that reader would: the PDF
`attention-audition/AttentionDraft.pdf` (render pages to images with `pdftoppm -r 110 -png` into a
folder under `attention-audition/.work/` if you cannot read PDFs directly, and look at the
figures), with its source `attention-audition/book/chapters/05-attention.md` at hand. You may skim
`book/chapters/03-energies.md` and `01-task.md` to know what the reader already has, and the
current Chapter 5 `book/chapters/05-attention.md` (the old version, which builds attention
bottom-up: an average, a score, a mask) for comparison.

Context you need: the book's author found attention unclear in the old chapter. What finally
worked for them was a top-down route — the conditional P(next symbol | context) is an
exponentially large table; which structural restrictions replace it? — with attention reached by
a sequence of named approximations (an uncertain index over earlier positions, exact but empty;
one shared rule on vectors; a Gibbs distribution with bilinear energy; low rank, whose two factors
are "query" and "key"; a mean-field step). The draft attempts this route. What failed for the
author before: jargon introduced before the object exists; vague phrases like "attention lets
earlier positions influence the prediction"; toy examples offered as the explanation.

Write your review to `attention-audition/briefs/07-review-muse.report.md`. Edit nothing else; do
not commit; do not start Muse or Gemini subagents.

## Report
1. **Where you got lost or bored**, with the exact sentence, and why (undefined term, a step that
   skipped something, an ambiguous symbol, a sentence that says nothing).
2. **Jargon** used before the thing it names was on the page.
3. **Does the route work?** After reading, could you say in your own words why attention is the
   formula it is, and what each approximation costs? Say it, in five lines. Where is the argument
   weakest?
4. **Each figure:** what question it answers for you, whether its caption lets you read it, and
   whether it sits where the argument needs it.
5. **Length and order:** anything to cut, merge, or move. The draft is ~24 % longer than the old
   chapter; the goal is clearer, not longer.
6. **Voice:** any hype, ML-blog tone, filler, or sentence a physicist would wince at.
7. The ten most valuable concrete edits, each with replacement wording, most important first.

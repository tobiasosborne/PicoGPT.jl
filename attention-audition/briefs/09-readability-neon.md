# Brief 09 — readability pass on the revised chapter (Gemini Neon)

Working directory: the repository root of PicoGPT.jl (a git worktree). The chapter
`attention-audition/book/chapters/05-attention.md` (committed as revision 1) is now
**mathematically right** — two reviewers' corrections are in it — but to reach a word limit it
was **compressed into telegraphic prose**: stacked parentheticals, semicolon chains, symbols used
before they are introduced, whole arguments squeezed into a clause (e.g. "Forced at $i=1$, this is
free for $i>1$ ($P$ never fixes $r$); uniform $r=1/i$ (ignoring $y$) leaves $p_j=P(y\mid c)$
intact."). That is exactly what the book's author could not read: they asked for the explanation
"explicitly, line by line". The word limit was the orchestrator's mistake; it is relaxed.

Your task: **rewrite the chapter into full, plain book prose without changing what it claims.**

Read first: `BRIEF.md` §2 and §7, `book/briefs/writer.md` (the voice), the old chapter
`book/chapters/05-attention.md` (its sentence rhythm is the target: look how it reads), the
current chapter, and the proof pages `attention-audition/proof/pages/p-07.png` … `p-16.png` to see
how it reads now. For the user's own words on what fails and what works, turns 3, 6 and 7 of
`/home/tobias/Projects/PicoGPT.jl/book/sources/attention-as-ansatz.transcript.md` (read only).

## Rules

- **Every claim, scope condition, formula, figure, value placeholder, code block and exercise
  stays.** The maths was reviewed; do not weaken, strengthen or drop a statement. If a rephrasing
  would change a claim, keep the claim. If something is too fine for the main text (a
  side-condition only an expert needs), move it into a short `::: aside` rather than cramming it
  into the sentence; do not delete it.
- One idea per sentence. Introduce each symbol in words before or as it is used. Display the
  equations that carry the argument (the step-1 identity, the mixture $p_\text{mix}$, the Gibbs
  weights, the factorisation, the mean-field replacement, the geometric mixture) on their own
  lines. No parenthetical stacks; no "X; Y; Z" chains.
- Each of the five steps opens by saying in a plain sentence what it does, then what is exact,
  what is assumed, when it holds and when it fails.
- Step 1 must answer the author's turn-7 question plainly: adjoining the index enlarges the sample
  space; the enlarged distribution is *chosen*, not determined by $P$; the uniform choice shows the
  identity alone says nothing; the first real assumption is that some choice makes both factors
  simple, which nothing guarantees.
- Length: **about 2,000–2,200 words** of source (the old chapter is 1,609). Clarity first; if you
  need more, cut repetition, not explanation.
- Keep the section ids (`{#sec-…}`), the figure ids and attributes, `{{value:…}}` names, and the
  `::: {.code …}` blocks exactly. Do not start Muse or Gemini subagents. Do not commit.

## Check

`node print/build.mjs --edition=attention-audition --proof` from the repository root must succeed.
Then read pages p-07 … p-16 as the reader, start to finish, and fix whatever still reads like
notes. Report the word count, and any claim you were unsure how to phrase without changing it.

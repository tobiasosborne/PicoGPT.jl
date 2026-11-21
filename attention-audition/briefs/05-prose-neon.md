# Brief 05 — draft the revised attention chapter (Gemini Neon, prose)

PicoGPT is a GPT written from scratch in Julia plus a book about it for BSc physics and maths
students: the maths is easy for them, the jargon is the obstacle. The working directory is the
repository root, a git worktree. You write **one chapter**: a draft revision of Chapter 5,
*Attention*, into `attention-audition/book/chapters/05-attention.md` (that file is at present a
copy of the old chapter, as a placeholder; replace it entirely). It is an audition: the author
will read it as a PDF and accept, reject or redirect it.

Do not start Muse or Gemini subagents. Do not commit.

## Read, in this order

1. `BRIEF.md` — binding: §2 notation, §3 the model, §7 source conventions, §12 findings.
2. `book/briefs/writer.md` — the voice and the rules for chapter authors (ignore its coordination
   and figure-building parts; you write prose only).
3. `book/chapters/05-attention.md` — the **current** chapter. Correct, and it stays in the book
   until the author decides. It builds attention bottom-up (an average, a score, a mask). The
   author wrote the book and still found attention unclear that way.
4. Its neighbours `book/chapters/03-energies.md`, `04-surprise.md` (what the reader owns: softmax
   is the Gibbs distribution, temperature, cross-entropy) and `06-block.md` (what comes after:
   LayerNorm, the MLP, the residual stream). Also `01-task.md` (the chain rule of probability).
5. `/home/tobias/Projects/PicoGPT.jl/book/sources/attention-as-ansatz.transcript.md` (outside this
   checkout; read only, do not copy). **Read the user's own messages closely, turns 1–7.** They
   show what failed: query/key/value jargon before it was earned; "attention lets earlier
   positions influence the prediction" (vague); worked toy examples offered as the explanation.
   And what worked: turn 6's answer ("ok this is much more like it"), and their sharp question in
   turn 7 about the enlarged sample space.
6. `/home/tobias/Projects/PicoGPT.jl/book/sources/attention-as-ansatz.md` — a distillation of that
   conversation (read only). Its §2 is the route. Its §3 is the dictionary to the book's notation.
7. `attention-audition/notes/spine.md` — **the mathematical authority for this chapter**, written
   by a mathematician from the distillation and checked against the model. Where the spine and
   the distillation differ, the spine wins (it corrects two points of the distillation and the
   scope of the "geometric mixture" remark).
8. `attention-audition/notes/numbers.md` — the specimen's measurements of the mean-field step.

## The route

Top-down. The conditional $P(\text{next character}\mid\text{context})$ is a table with exponentially
many entries; the chain rule (Chapter 1) is exact and gives no recipe for the factors. Every map so
far acts column by column (a bigram model with a clock). The question a physicist asks: *which
structural restrictions replace the table*, as a rank bound gives a matrix product state? Then
attention appears as an ansatz reached by a short sequence of named approximations:

1. an uncertain index over earlier positions — **exact, and empty** until both factors are
   required to be simple (answer the turn-7 question: the extension is chosen, not determined);
2. one shared small rule on vectors attached to positions (a mixture of per-source predictions);
3. a Gibbs distribution over positions whose energy is bilinear;
4. a low rank for that bilinear form — its two factors are all that "query" and "key" mean;
5. a mean-field step that replaces the selected vector by its expectation.

Each approximation is stated **as** an approximation: what is exact, what is assumed, when it
holds, when it fails. **Jargon is named only after the object exists**: "query", "key", "value",
"attention", "head", "mask" appear when the thing they name is already on the page; the `names`
table at the top may list them (it is the chapter's dictionary), but the prose earns them.

## What must survive from the current chapter

The route changes; the facts do not. Keep, reorganised under the new argument: every position is
a reader, and one pass gives the $T$ conditionals of the chain rule; rows are readers and columns
sources ($O=VA^\T$ and why the transpose); the causal mask as lower-triangularity (in the new
route: the index ranges over $j\le i$); the bias terms and why $b_k$ never learns; why $\sqrt d$
(temperature; the variance argument is about initialisation only); heads as a direct sum
$\R^D=\R^d\oplus\R^d$ recombined by $W_o$ (in the new route: one index per head); the
parameter count `{{value:attentionParams}}`; the "In the code" section with its two code blocks
exactly as now (`::: {.code file="src/attention.jl" fn="attention"}` / `fn="mix"`); exercises
(3–6; keep the good old ones where they still fit, e.g. the shift-invariance/$b_k$ one, the
permutation one, the Jacobian one; one new pencil exercise from the distillation §4 is welcome,
e.g. the data-dependent average that computes a product $ab=2a+3b-4o$). Correct the old mask
aside: in the specimen 28 of the 64 scores are discarded, not half (see the spine).

## Figures (place them where the argument wants them)

Existing, unchanged, with their ids and attributes exactly as in the current chapter (you may
rewrite their captions to fit the new route, keeping what they say true; the `[…]{.web}` spans
are web-only instructions):
- `fig-attention-reads` (arcs: what each character reads; head 1 above, head 2 below);
- `fig-attention-pipeline` (one reader, one head: keys, query, scores, weights, weighted values);
- `fig-attention-triangle` (the two attention matrices as lower triangles of dots).

New, being designed in parallel: `::: {.figure #fig-attention-meanfield}` — for one real reader
of the trained specimen (window `or not t`, position 6, head 1), the per-source predictions $p_j$,
their arithmetic mixture, their geometric mixture and the model's actual prediction; the actual
prediction sits with the geometric mixture. Place it in the mean-field section and write a caption
draft (the designer's caption may replace it later).

## Numbers

Never type a number the model computed. Use computed values: `{{value:attentionParams}}`,
`{{value:mixtureReaderCount}}`, `{{value:mixtureLossActual}}`, `{{value:mixtureLossArithmetic}}`,
`{{value:mixtureLossGeometric}}` exist. If you need another (e.g. mean total variation of each
mixture from the actual prediction, from `numbers.md`), invent a descriptive name, use it as
`{{value:name}}`, and **list every new name with its intended meaning and source field in your
report**; an engineer will wire them in. Small exact numbers of the architecture ($D=8$, $H=2$,
$d=4$, $T=8$, 28 of 64) and pencil examples may be written out.

## Form

- BRIEF §7 conventions: `# Attention {#ch-attention}`, `::: claim` (one sentence, new), `::: names`
  (maths | physics | ML), `## … {#sec-…}` sections, `::: aside` margin notes (generous, short),
  figures, code, `## Exercises {.unnumbered}` with `::: exercise` blocks.
- Cross-references: `@ch-task`, `@ch-energies`, `@ch-surprise`, `@ch-block`, `@ch-pulling-back`
  work; `@sec-…` and `@fig-…` only for ids **inside this chapter** (a reference into another
  chapter's section renders as raw text in this edition).
- The matrix product state comparison and the one-particle (direct-sum) picture: the reader may not
  know MPS. Decide: a short margin aside where the question is posed (an MPS squeezes the past
  through a bond of fixed dimension; attention keeps one vector per position and reaches in with a
  shared rule) is the default; the one-particle picture only if it fits in one exact aside (use
  only the statements the spine marks as exact, and honour its correction: attention acts on the
  LayerNorm output while the residual carries the raw stream). Quantum generalisations and the
  modern-decoder rewrite are out of scope.
- **Length close to the present chapter** (about 1,600 words of source, 126 lines). Clearer, not
  longer. If you must exceed it, by at most a quarter.
- Voice: dry, exact, plain; Hohm's lecture notes are the model (objects are maps between named
  spaces; index and matrix form side by side; softmax is the Gibbs distribution). No hype, no
  "magic", no anthropomorphism beyond what is defined, no ML-blog tone.

## Check before you finish

Build the edition and fix anything that breaks: `node print/build.mjs --edition=attention-audition
--tolerant` from the repository root (a missing figure component or an unknown `{{value:…}}` name
will be reported; unknown value names fail the build — if so, keep them, and tell me, or
temporarily test with them removed). Read your own chapter once, start to finish, as the reader.

## Report

The file; the chapter's outline (sections with one line each); where each old fact went and any
you dropped (with the reason); new value names; the MPS / one-particle decision and why; anything
in the spine you found unclear or could not use.

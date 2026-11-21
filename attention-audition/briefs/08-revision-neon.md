# Brief 08 — the single revision of the draft chapter (Gemini Neon)

Working directory: the repository root of PicoGPT.jl (a git worktree). You wrote the draft
`attention-audition/book/chapters/05-attention.md` (brief: `attention-audition/briefs/05-prose-neon.md`,
still binding). Two independent reviews came back. This is the **one** revision pass before the
author reads it; make it count. Do not start Muse or Gemini subagents. Do not commit.

Read: `attention-audition/briefs/06-review-astra.report.md` (mathematics; audited the numbers),
`attention-audition/briefs/07-review-muse.report.md` (the intended reader), the figure's final
state (`attention-audition/proof/pages/p-13.png`, and
`attention-audition/figures/05-attention/fig-attention-meanfield.svelte` if needed), and again
`attention-audition/notes/spine.md` and the user's turns 1–7 in
`/home/tobias/Projects/PicoGPT.jl/book/sources/attention-as-ansatz.transcript.md` (read only).

## Decisions already taken (by the orchestrator)

1. **Mathematics: Astra wins.** Fix every **error** in Astra's report (1–6) and gaps 7–9, in a form
   this reader can follow: true scope, plain words, few symbols. Do not transplant Astra's full
   technical apparatus into the prose (e.g. finding 7's segment-Hessian condition becomes one honest
   sentence: the gap is second order in the spread of the values *where the readout is smooth
   between them*, and a ReLU kink between them can make it first order). Nits (11–12) as cheap.
   - Astra 1 and 2 change exercises 5.4 and 5.3: use Astra's replacement statements.
   - Astra 3: distinguish the exact target $P(y\mid c)$ from the candidate family
     $p_\text{mix}(y\mid c;\theta)$; note the extension is forced at $i=1$.
2. **Where Muse and Astra conflict on mathematics, Astra wins.** In particular Muse's suggested
   paragraph for the Step 1→2 joint says the identity was empty "because $r$ was allowed to depend
   on $y$". That is not the reason: the uniform $r$, which ignores $y$, is equally empty. Keep the
   spirit of Muse's edit (say plainly what the first real assumption excludes: a next character
   fixed by the joint pattern of several positions, none of which carries it alone) but get the
   reason right.
3. **Words.** Muse asks to replace "approximation" by "restriction". The author asked for each step
   *stated as an approximation*. Use both, consistently: steps 2–4 are **restrictions** of the
   family of conditionals one is willing to consider (each is an approximation because the
   restricted family may not contain the true table); step 5 replaces one predictor by another.
   Say this once, early, and keep to it.
4. **Muse's second joint (Step 5, "faithfulness"):** make clear *before* the error estimate that the
   trained weights were fitted for the mean-field predictor, so the comparison measures how far the
   network sits from the selection picture, not an error of the network. The geometric-mixture
   identity and the specimen result stay: they are the chapter's best new content.
5. **The figure (`fig-attention-meanfield`)** is final; adopt the designer's caption below (you may
   tighten its wording; keep it true). The figure is one head choosing, head 2 at its usual output,
   and its strip covers the 70 readers with two or more sources. The prose numbers are the joint
   two-head cross-entropies over 80 readers: say so in prose so the two never look contradictory.
   Mean total variations of the joint mixtures are available as computed values if you want them:
   `{{value:mixtureTvArithmetic}}`, `{{value:mixtureTvGeometric}}` (wired by the orchestrator).
   Designer's caption:
   > One reader of the trained specimen, the 't' at position 6 of `or not t`, read through head 1
   > with head 2 held at its usual output. Under each source $j$ is $p_j$, the prediction when the
   > head hands over $v_j$ alone; below them, three ways to combine these predictions, on one
   > scale. The arithmetic mixture $\sum_jA_{ij}p_j$ adds the sources' shares of probability, and a
   > source can only add; the geometric mixture adds their weighted surprises $A_{ij}(-\ln p_j)$
   > and renormalises, so the source that finds 'o' nearly impossible prices it out, and the
   > model's own $p$, which decodes $o_i$ once, sits with the geometric mixture. The strip repeats
   > both distances at every reader with more than one source: the geometric mixture is the
   > closer, by orders of magnitude.
6. **Length: bring it down to at most ~1,800 words of source** (now ~2,000; the old chapter is
   ~1,600). Muse's §5 gives the cuts: shorten Step 1's rigour to what answers the author's
   question; cut the Taylor paragraph to two or three sentences; drop anything inherited that no
   longer serves the route. Clearer, not longer.
7. **The MPS aside stays, short and corrected** per Astra 9 (rank across a cut, not a correlation
   length; attention stores up to $T$ columns and shares the addressing rule). The author is a
   physicist; it is a margin note, and they will decide. Muse's objection goes into the
   orchestrator's note to the author.
8. **Vocabulary from later chapters** (Muse: LayerNorm, MLP, residual leaking backwards): name such
   things only as "the column-wise maps of @ch-block", unless the chapter needs the specific object.
9. **The causal constraint $j\le i$**: say where it comes from (Chapter 1: the prediction of
   $x_{i+1}$ may use only $x_{\le i}$) when the index is introduced, so the mask section reads as
   the batched form of a choice already made, not a new idea.
10. Take Muse's other top-ten edits on their merits; skip any that lengthen the chapter without
    making it clearer.

## Check before you finish

`node print/build.mjs --edition=attention-audition --tolerant` from the repository root must
succeed; then read your chapter once, start to finish, from the PDF page images
(`--proof` writes `attention-audition/proof/pages/`). Report the word count.

## Report

What you changed for each Astra finding (one line each) and for each Muse edit you took or
declined (one line each, with the reason); the new word count; anything you disagree with.

# Brief 06 — mathematical review of the draft chapter (Astra)

Working directory: the repository root of PicoGPT.jl (a git worktree). PicoGPT is a GPT in Julia
plus a book for BSc physics/maths students. A draft revision of Chapter 5 (Attention) presents
attention top-down, as an ansatz for P(next character | context) reached by five named
approximations. Review it **as a mathematician**. Write your review to
`attention-audition/briefs/06-review-astra.report.md`. Edit nothing else; do not commit; do not
start Muse or Gemini subagents.

## The artefact
- `attention-audition/book/chapters/05-attention.md` (the draft; `{{value:name}}` placeholders are
  filled from `attention-audition/edition.js` → `proseValues`; `::: {.code}` blocks are pulled from
  `src/attention.jl`).
- The built PDF `attention-audition/AttentionDraft.pdf` (you may rebuild with
  `node print/build.mjs --edition=attention-audition --tolerant`).

## The references to check it against
- `BRIEF.md` (§2 notation, §3 exact model, §12 findings), `book/notes/derivations.md`, `src/*.jl`.
- `attention-audition/notes/spine.md` (the derivation spine the writer was given) and
  `attention-audition/notes/numbers.md`, `attention-audition/scripts/mixture.jl`,
  `attention-audition/artifacts/mixture.json` (the specimen measurements). **These numbers have not
  been independently audited yet: audit them** — read the script, check its definitions against
  the model semantics (BRIEF §3) and against what the prose claims, and re-run it if useful
  (`julia --project=. attention-audition/scripts/mixture.jl`; it rewrites the artifact
  deterministically).
- The current chapter `book/chapters/05-attention.md` (bottom-up; it stays in the book): every fact
  it establishes should survive in the draft unless the draft says why not.

## What to check
1. Every mathematical statement: true, with its true scope. Each of the five approximations stated
   as an approximation — what is exact, what is assumed, when it holds, when it fails. Step 1
   exact and empty until both factors are required simple; the extension chosen, not determined.
2. The geometric-mixture statement: exact only for a readout affine in the attention output; is the
   draft's claim about the real model (W_o, residual, LayerNorm, MLP) correctly scoped?
3. The specimen numbers: correctly defined, correctly computed, correctly described in prose and in
   the caption of `fig-attention-meanfield`.
4. Notation per BRIEF §2; consistency of indices (reader i, sources j ≤ i), shapes, transposes.
5. Facts of the old chapter that are missing or changed; anything wrong in the old chapter that the
   draft inherited.
6. Exercises: well posed and solvable with what the chapter gives.

## Report format
A list of findings, most severe first, each with: location (section / line), the problem, why it
is wrong (proof or counterexample where short), and a concrete fix (replacement wording or
formula). Mark each **error** (false or wrongly scoped), **gap** (true but unjustified/unclear),
or **nit**. End with a one-paragraph verdict, and the audit result for the numbers.

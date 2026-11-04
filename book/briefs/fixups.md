# Integration fix-ups (collected by the orchestrator; to be done in the integration pass)

- [ ] `/figures/<id>/` route: `@ch-…` cross-references in captions resolve to `../slug/`, which
      breaks prerendering. Rewrite links in the standalone figure route.
- [ ] Ch. 4 `fig-gradient-p-minus-e`: the first covector key now lives in the prologue
      (`fig-two-inks`), so turn this one into a reminder.
- [ ] Print: add the contents map (`Contents.svelte`) as a plate in the print edition.
- [ ] Shared `Scrubber`: phase labels crowd on narrow screens.
- [ ] Dictionary: some ML-column entries read oddly ("backpropagation through f"). Tidy the
      names tables across chapters.
- [ ] Traces: regenerate all of them under the canonical recipe (BRIEF §13) once the JS engine
      is unified.

## From the print edition (print agent report)
- [ ] Figures too tall for the page. Each needs a print layout ≤ ~200 mm including its caption:
  - `fig-temperature` (3.2): printed at 49 %, illegible. Most urgent.
  - `fig-shakespeare-attention` (9.4)
  - `fig-specimen-anatomy` (0.1)
  - `fig-train-live` (8.1)
- [ ] Controls drawn in the static state:
  - `fig-rotation` (2.4): the shift-k slider
  - `fig-gradcheck` (7.4): the step selector
- [ ] 27 captions tell the reader to drag, click or scrub. Wrap those sentences in a `.web` span and
      hide it in print.
- [ ] `fig-forward-backward`: the `stroke: url(#hatch…)` pattern and the multiply blends don't
      survive print.
- [ ] Duplicate section ids:
  - `sec-floor`: chapters 1 and 4
  - `sec-code`: chapters 1, 2 and 5
- [ ] Asides too long for their page:
  - ch. 6 LayerNorm aside
  - ch. 6 MLP/Rosenblatt aside
  - ch. 8 slingshot aside
- [ ] Layout: gaps on p. 43 and p. 89; widow on p. 110.
- [ ] Glyph coverage:
  - Archivo has no Greek.
  - Literata lacks ℝ ∈ ∝ ⊙ ␣.
  - XCharter Math lacks ‾.
  Print falls back correctly; the web's `tokens.css` still falls back to system fonts. Fix the web
  fallback stack.
- [ ] DESIGN.md §8 is out of date: running heads, notes, bottom margin, SVG pattern rasterising.
- [ ] `task.svg` trace: the faint parallel hairlines between positions 2–4 print as flat grey
      blocks. Thin them out.

## From chapter authors 6–8
- [ ] `Scrubber`: the last phase label is never right-aligned (`:last-of-type` never matches).
- [ ] A paragraph after an aside the filter has moved loses its first-line indent.
- [ ] Inline maths followed by punctuation can wrap before the punctuation.
- [ ] The `/block/` frontispiece caption overflows horizontally.
- [ ] `prerender_missing_id` warnings: `fig-entropy-floor`, `fig-logsumexp`.
- [ ] Verify the citations written from memory: chapters 6–8 and the epilogue.
- [ ] Undocumented 0-based token ids in the gradient trace.
- [ ] `crossentropy` may be quoted in both ch. 4 and ch. 7.

## From the chapter 3/4/9 author
- [ ] Ch. 9 prose quotes the step-5000 laptop export: the real-word shares, "5000 steps", no copying
      head, and 0.89 vs 0.90 nats on the repeated line. Recheck every one of these when the
      monolith run (D=256, L=6, 12k steps) replaces the data.
- [ ] `speaking.svg` frontispiece is missing (orchestrator). Then let ch. 9 refer to it.
- [ ] The first covector key is claimed by both the prologue (`fig-two-inks`) and ch. 4
      (`fig-gradient-p-minus-e`). Keep the prologue's; make ch. 4's a reminder.

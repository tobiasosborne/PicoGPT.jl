# Integration fix-ups (collected by the orchestrator; to be done in the integration pass)

- [x] `/figures/<id>/` route: `@ch-…` cross-references in captions resolve to `../slug/`, which
      breaks prerendering. Rewrite links in the standalone figure route.
      Note: Caption chapter/figure links now resolve from the standalone route, including same-chapter anchors.
- [x] Ch. 4 `fig-gradient-p-minus-e`: the first covector key now lives in the prologue
      (`fig-two-inks`), so turn this one into a reminder.
      Note: Renamed the existing key “Remember the two inks”; the prologue retains the first introduction.
- [x] Print: add the contents map (`Contents.svelte`) as a plate in the print edition.
      Note: Reused Contents.svelte as a single print plate at the existing print grid and type sizes.
- [x] Shared `Scrubber`: phase labels crowd on narrow screens.
      Note: Narrow screens show the current phase below the rail; all ticks and keyboard stops remain.
- [x] Dictionary: some ML-column entries read oddly ("backpropagation through f"). Tidy the
      names tables across chapters.
      Note: Tidied ML terms and kept measured dictionary columns in intact page groups to avoid Paged.js fragmentation.
- [x] Traces: regenerate all of them under the canonical recipe (BRIEF §13) once the JS engine
      is unified. *(done by the orchestrator: displacement view for cover and vectors)*
      Note: Not done: regeneration is owned by the orchestrator; neither excluded trace directory was modified.

## From the print edition (print agent report)
- [x] Figures too tall for the page. Each needs a print layout ≤ ~200 mm including its caption:
  - `fig-temperature` (3.2): printed at 49 %, illegible. Most urgent.
  - `fig-shakespeare-attention` (9.4)
  - `fig-specimen-anatomy` (0.1)
  - `fig-train-live` (8.1)
      Note: Added compact print geometry with unchanged label scale; all four figures fit within 200 mm including captions.
- [x] Controls drawn in the static state:
  - `fig-rotation` (2.4): the shift-k slider
  - `fig-gradcheck` (7.4): the step selector
      Note: The rotation slider becomes a fixed shift label; the gradcheck selector is absent in static mode.
- [x] 27 captions tell the reader to drag, click or scrub. Wrap those sentences in a `.web` span and
      hide it in print.
      Note: Marked 29 caption instructions and prose/exercise instructions with .web spans, hidden in both print paths.
- [x] `fig-forward-backward`: the `stroke: url(#hatch…)` pattern and the multiply blends don't
      survive print.
      Note: Static return strokes use transverse dashes and arcs omit multiply; the PDF’s existing hatch conversion remains vector-only.
- [x] Duplicate section ids:
  - `sec-floor`: chapters 1 and 4
  - `sec-code`: chapters 1, 2 and 5
      Note: Made floor/code IDs chapter-specific and updated every cross-reference to the intended chapter.
- [x] Asides too long for their page:
  - ch. 6 LayerNorm aside
  - ch. 6 MLP/Rosenblatt aside
  - ch. 8 slingshot aside
      Note: Shortened the LayerNorm, rectifier/Rosenblatt and slingshot historical notes without losing their sources.
- [x] Layout: gaps on p. 43 and p. 89; widow on p. 110.
      Note: Recompute heading keeps after floats move; shortened the clipping lead-in; strict PDF pagination reports no gaps or widows.
- [x] Glyph coverage:
  - Archivo has no Greek.
  - Literata lacks ℝ ∈ ∝ ⊙ ␣.
  - XCharter Math lacks ‾.
  Print falls back correctly; the web's `tokens.css` still falls back to system fonts. Fix the web
  fallback stack.
      Note: Web fallback stacks now use the shipped Literata, XCharter Math and JuliaMono faces, matching print.
- [x] DESIGN.md §8 is out of date: running heads, notes, bottom margin, SVG pattern rasterising.
      Note: Updated running-head and note placement, the 21.7 mm foot margin, actual type sizes and vector hatch conversion.
- [x] `task.svg` trace: the faint parallel hairlines between positions 2–4 print as flat grey
      blocks. Thin them out.
      Note: Not done: this SVG belongs to the orchestrator and is explicitly excluded from this assignment.

## From chapter authors 6–8
- [x] `Scrubber`: the last phase label is never right-aligned (`:last-of-type` never matches).
      Note: Explicit first/last phase classes replace selectors confused by the rail and playhead spans.
- [x] A paragraph after an aside the filter has moved loses its first-line indent.
      Note: Paragraphs wrapped in .noted retain their indent after another paragraph, on web and print.
- [x] Inline maths followed by punctuation can wrap before the punctuation.
      Note: Inline MathML and immediately following punctuation share a nonbreaking span; display equations remain separate.
- [x] The `/block/` frontispiece caption overflows horizontally.
      Note: The frontispiece caption wraps normally and can break long identifiers within the existing width.
- [x] `prerender_missing_id` warnings: `fig-entropy-floor`, `fig-logsumexp`.
      Note: Corrected standalone caption URLs; strict prerender resolves fig-entropy-floor and fig-logsumexp without warnings.
- [x] Verify the citations written from memory: chapters 6–8 and the epilogue.
      Note: Audited every chapter citation against sources; linked them and recorded metadata/corrections in book/briefs/citations.md.
- [x] Undocumented 0-based token ids in the gradient trace.
      Note: Verified already resolved: the current shared trace helper encodes tokens as 1…V; engine conformance passes.
- [x] `crossentropy` may be quoted in both ch. 4 and ch. 7.
      Note: Kept the full listing in ch. 4 and replaced ch. 7’s duplicate with a reference to that section.

## From the chapter 3/4/9 author
- [ ] Ch. 9 prose quotes the step-5000 laptop export: the real-word shares, "5000 steps", no copying
      head, and 0.89 vs 0.90 nats on the repeated line. Recheck every one of these when the
      monolith run (D=256, L=6, 12k steps) replaces the data.
      Note: Not done: checked-in artifacts are still D=128, L=4, 5000 steps; the monolith export has not arrived.
- [ ] `speaking.svg` frontispiece is missing (orchestrator). Then let ch. 9 refer to it.
      Note: Not done: the orchestrator’s trace is absent; a prose reference must wait for the actual computation.
- [x] The first covector key is claimed by both the prologue (`fig-two-inks`) and ch. 4
      (`fig-gradient-p-minus-e`). Keep the prologue's; make ch. 4's a reminder.
      Note: Resolved by the same ch. 4 reminder key; fig-two-inks remains the first covector key.

## Integration verification — 4 November 2026

- `node --test web/src/specimen/`: 12 passed, including Julia/JS conformance.
- `cd web && BOOK_STRICT=1 npm run build`: passed, no book/compiler/prerender warnings (only the inherited NO_COLOR/FORCE_COLOR Node notice).
- Julia source was untouched, so the conditional `Pkg.test()` run was not needed.
- `cd print && node build.mjs --strict`: passed; `npm run proof -- --strict` passed on the final sources: 136 pages, zero warnings or errors, only the four shipped font families, no rasterisation warning.
- Playwright: all 40 figures captured and reviewed at desktop, mobile and `?print` widths; zero figure runtime errors. Final temperature and gradcheck changes were recaptured. Screenshot files: `/tmp/pico-fixups-shots/`.
- Opened the changed plate/layout/asides/dictionary page PNGs and the greyscale contact sheet from `print/proof/`; checked all 105 dictionary terms are visible once, with none missing.
- `git diff --check`: passed. No commits; neither trace directory nor `src/` was modified.

# Muse review — application record

Applied on branch `redesign`, 4 November 2026. This records the numbered findings
in `review-muse.md`, including the exclusions in the implementation brief. No
commit was made. Trace generators, trace assets and Shakespeare measurement
rechecks were left to the orchestrator.

## Numbered findings

1. **Fixed.** The three attention figures and their keys default to training
   window 10. The adapter's display windows and the book's context selectors now
   stop at the last window with a complete target. The eleventh suffix remains
   available as an arbitrary generation prompt, not as a training window.
2. **Fixed by making the promise honest.** The Prologue and DESIGN describe
   representative print states and the additional choices available on the web.
   Fig. 8.1 prints the loss and samples, without four illegible parameter and
   attention miniatures; its caption points back to the anatomy plate. Browser
   instructions in the body are edition-specific, with `.print` counterparts
   where needed. The print curve's provenance note describes the curve actually
   drawn. The training-window plate follows its complete explanation, allowing
   intervening text to fill a pagination gap.
3. **Fixed.** Fig. 0.2's caption introduces ink meanings rather than deriving
   the readout pullback. Its sidenote names Chapters 3, 4 and 7 and says what to
   follow now. Chapters 3 and 5 flag LayerNorm at first use. Names tables carry
   “defined in this chapter”. The figure's mathematical detail remains for the
   reader to revisit; no chapter was reordered.
4. **Fixed for in-scope provenance.** Prologue, Chapter 8 and Epilogue use the
   same absolute-tolerance statement: arrays `1e-12`, recorded loss curve `1e-9`,
   with failed replays discarded. Removed “exact” and twelve-decimal-place
   promises. Model figures distinguish model computations, exported data and
   deliberately changed settings. Load-bearing prose readings use
   `{{value:name}}`, expanded before Pandoc from the artifact/layout by
   `web/src/lib/server/prose-values.js`: parameter and subarray counts, finite
   difference costs, prediction totals and Float64 exponential overflow. Layout
   and stored parameter lengths must agree; missing/unknown values fail the
   build. The Chapter 8 cover aside no longer asserts a different initial draw;
   it distinguishes stored seed 0 from engine-specific seeds and refers to the
   plate's provenance. **Deferred to orchestrator:** auditing or changing the
   trace's own metadata/run, excluded by the task.
5. **Fixed.** Shared `fmtProb` preserves endpoint distinctions: compact `≈1` /
   `≈0`, numerical gaps or tails in readouts, and exact endpoints only when the
   supplied arithmetic actually returns them (including masks/frozen limits).
   Applied to context bars, chain rule, energy ladders, attention, surprise,
   residual/readout figures and sampling. Chapter 1 no longer asserts certainty.
   A screenshot iteration replaced long small-tail labels with `≈0` to avoid
   crowding the mobile attention pipeline; readouts retain their values. Its second
   head label is kept within the mobile figure bounds.
6. **Fixed with the less invasive notation note.** The Prologue explicitly
   disambiguates P, V, E, F, H (including Hessians), B, mean/per-prediction loss,
   index uses, log/ln, and attention output versus vocabulary readout. Chapter 2's
   table now uses `(t−1)`. Chapter 8 explains that both epsilon constants use `ε`
   in Julia but belong to different functions. The energy ladder uses the
   calligraphic free-energy symbol in the mathematical face.
7. **Deferred to orchestrator, as requested.** No frontispiece or trace-path
   work. During this implementation the orchestrator landed the three plates in
   commit `a34657a`; their content and metadata were not audited here. The final
   PDF includes that HEAD.
8. **Fixed locally; one author follow-up remains.** Interactive SVG containers
   expose their control subtrees as groups (including the sharpness plate’s
   nested chart sliders, found by the rendered-DOM check); image roles remain on static
   renderings. Pointer readouts have focus equivalents and keyboard activation,
   including surprise, Adam, variance, embeddings and affine cells. Fig. 0.2's
   columns are keyboard buttons; Fig. 9.4 has a single reader slider, independent
   of its head buttons. Anatomy and cover parameters can be traversed by keys;
   minibatch clouds have keyboard point selection. Focus is visible. Narrow
   chapter titles wrap within the viewport. The PDF build now requires a
   nonempty text layer and rejects the observed “Te ” corruption; disabling the
   broken common ligature in print preserves “The”. **For the author:** semantic
   reading of display mathematics remains beyond a text-extraction smoke check;
   choose spoken equation descriptions and a tagged-math delivery strategy.
9. **Fixed.** Shared attention arcs have source-end chevrons (for visible
   non-self weights); loops denote self-reads. The key explains direction.
   Attention readouts and the enlarged key name positions as well as characters,
   including pipeline and matrix selections.
10. **Fixed.** The Prologue's scale rule and DESIGN explicitly name the two
    separate-scale plates. Figs. 0.2 and 7.2 print their actual full-cell/full-scale
    magnitudes for every array, including the share and the summed gradient.
11. **Fixed.** Fig. 6.1 describes RMS changes under updates of a stated RMS,
    without adding scalar norms. Chapter 6 permits preservation rather than
    asserting that a branch can only refine. The names table and stem key qualify
    Euler form without a small-step assumption. Chapter 8 distinguishes position,
    depth and training time, and states that Adam evolves `(θ,m,v)` rather than
    discretising gradient flow in θ alone.
12. **Fixed.** Chapter 5 labels the temperature analogy and explains independent
    per-reader normalisations. The sampler source comment identifies temperature
    as that of the Gibbs system with the same energies. No equations changed.
13. **Partly fixed; measurement updates deferred as requested.** Chapter 9
    defines the corpus-membership word test, including lower-casing and exclusion
    of single-letter words; names the contiguous-tail validation split; says the
    initial loss is near uniform guessing; and replaces unmeasured assertions
    about meaning and syntax with claims about the displayed evidence. No word
    shares, copy thresholds, repeated-line losses or other Shakespeare results
    were rechecked. **For the author/orchestrator after retraining:** regenerate
    these readings and revise all measurement-dependent verdicts together.
14. **Fixed.** “Beginnings and endings” names the nonempty prompt, externally
    chosen draw count, absent BOS/EOS mechanism, first space-at-position-1
    conditional, and re-embedding when the window slides. The chain-rule names
    table also includes the initial-symbol factor.
15. **Fixed.** Fig. 6.2's caption distinguishes coincidence in the two displayed
    coordinates from equality of vectors. The ideal sphere is identified once
    as `S^(D−2)`; the existing epsilon shortfall remains explicit.
16. **Fixed for book figures.** Shared loss chart and key say `log(1 + step)`;
    the key explains the `+1`. The separate sharpness and Shakespeare loss axes
    use the same precise wording, as does the Chapter 8 explanation and Chapter
    9 caption. **Deferred to orchestrator:** the Chapter 4 frontispiece-axis
    description, whose exact coordinate convention belongs with its excluded
    trace generator/metadata audit.
17. **Fixed by narrowing the claim.** The dictionary promises every term given
    three names, matching its generated contents. No forced physics synonyms
    were added merely to fill rows. Its introduction and first entries are kept
    together; the print column packing reserves space for that introduction.
18. **Fixed.** The scale caption describes equal panels and true relative corner
    squares, names the largest GPT-2 specifically, and leaves model counts to the
    figure/cited sources. Every specimen parameter is drawn, using a declared
    minimum visible size; previous models retain their true relative square size.

## Tier 3 (unnumbered in the review, listed in review order)

- **T3.1 fixed:** energy shelves reserve at least the character-label gap as well
  as the bar gap; all seven outcomes remain labelled.
- **T3.2 fixed:** the learning-rate exercise says each drag restarts from θ₀.
- **T3.3 fixed:** shortened the two sampling-weight error strings in the real
  Julia source; excerpts still come from that source.
- **T3.4 fixed:** the cover parameter-field explanation says to read across
  before down and distinguishes this from computation order.
- **T3.5 rejected for this task:** changing night trace treatment would restyle
  the book and contradict DESIGN's explicit inversion rule. Leave the palette
  decision to the design author; trace assets were excluded.
- **T3.6 fixed:** narrowed the Prologue's universal claim about language models.

## For the author — optional connections

These are opportunities in the review, not correctness repairs.

- **Opportunity 1 deferred:** commission the half-page symmetry/quotient-space
  box after §7.5, including the caveats about epsilon and coordinate-dependent
  optimisers. It needs an authored argument, not a caption patch.
- **Opportunity 2 deferred:** decide whether the five ambiguous predictions
  should become a recurring device throughout the book. Existing local floor
  references remain; a new sigil or coordinated narrative deserves an author.
- **Opportunity 3 deferred:** add the Shakespeare embedding/readout comparison
  after retraining, as a new figure with its own question and key.
- **Opportunity 4 fixed:** Chapter 8 now distinguishes convexity in logits from
  non-convexity in parameters and limits the local descent argument accordingly.
- **Opportunity 5 deferred:** recommend a computed bigram comparison beside the
  floor in §1.6, after deciding how much of the existing bigram exercise to give
  away. This adds an exhibit rather than repairing an existing reading.

## Verification

- `node --test web/src/specimen/`: 12 tests passed.
- `BOOK_STRICT=1 npm run build`: passed; no Svelte or book warnings. The final
  invocation clears inherited `NO_COLOR` to avoid Node's conflicting-colour-env
  diagnostic.
- `cd print && node build.mjs --strict`: passed; **148 pages** at HEAD `a34657a`, with no warnings.
  Includes font embedding, vector output, pagination and PDF text-extraction
  checks. The first proof's two layout warnings were fixed, not suppressed.
- Playwright from `print/node_modules`: all 40 figures captured and visually
  inspected at 1440 px desktop, 390 px mobile and `?print` (120 captures), with
  refreshed captures after iteration. Artifacts and contact sheets are in
  `/tmp/review-muse-shots/`; runtime-error/overflow/hidden-control results in its
  `report.json`. No image-role container contains controls in these renders.
- Focus/keyboard checks cover ink columns, Shakespeare readers, window bounds,
  surprise/Adam/variance readouts, minibatch points and anatomy parameters;
  the 390 px Energies opening was checked separately.
- Direct endpoint-format checks cover finite gaps, compact approximations and
  exact limits. `git diff --check` passes. Trace paths remain untouched.

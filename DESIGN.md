# PicoGPT — the visual system

Binding for the web book, the print edition and the films. It implements BRIEF §1 and §9:
**a printed science book that came alive.** Paper, one near-black ink, two spot inks with one
meaning each, every mark a number. Change a value here first, then in
`web/src/lib/styles/tokens.css` (the single source of the tokens).

Contents: 1 Principles · 2 Palette · 3 Type · 4 The web page · 5 Glyphs · 6 Keys ·
7 Motion · 8 Print · 9 Traces · 10 The figure contract · 11 Adding chapters and figures

---

## 1 Principles

1. **Two inks, one meaning each.** *Value ink* (indigo) marks quantities computed going
   forward: parameters and activations. *Covector ink* (cinnabar) marks adjoints x̄ pulled
   back from the loss. Nothing else is ever coloured. Selection, emphasis and categories use
   weight, size, position and the near-black ink, never a third hue.
2. **Every mark is a number.** A dot, a spoke, an arc, a bar or a weight of type stands for one
   real number of the model. Structure (rules, ticks, labels) is hairline and grey. If a mark
   is not a number, it should not be drawn.
3. **Sign is never hue.** Positive and negative are told apart by fill (filled/open), by
   direction (up/down, out/in), never by colour, because colour is spent on value/covector.
4. **Forward flows down, backward flows up.** In every figure the computation runs top to
   bottom in value ink; pullbacks run bottom to top in covector ink. Sequence position runs left
   to right, as in text. **The column is the token**, as in `X ∈ ℝ^{D×N}`.
5. **Survive black and white.** Value marks are solid; covector marks are barred or hatched (a
   covector is drawn as Misner–Thorne–Wheeler draw a 1-form: a stack of planes). In greyscale,
   the inks print at about 33 %, 50 % and 87 % grey (ink, covector, value), and the shapes
   still differ.
6. **Print first.** Every figure has a static state that is complete on paper. Interaction is
   for asking the next question, never for revealing what the page withheld.
7. **Direct labels, generous white space.** No legend boxes and no chart chrome. A "how to read"
   key with hairline leaders appears the first time each glyph appears.
8. **No cards, no panels, no dashboards.** Figures sit on the page like plates in a book: no
   frames, no shadows, no rounded boxes, no tabs, no dropdowns.

## 2 Palette

OKLCH is authoritative; hex is the sRGB rendering, for tools that do not speak OKLCH (plotter
SVGs, PDF checks). Contrast is WCAG against `paper`.

| token | OKLCH | sRGB | role | contrast |
|---|---|---|---|---|
| `--paper` | `0.987 0.005 85` | `#fcfbf7` | uncoated natural white; *not* cream | — |
| `--paper-2` | `0.968 0.007 85` | `#f6f3ee` | a shade of the same stock (rare) | — |
| `--ink` | `0.22 0.012 268` | `#181a20` | text, the selected thing | 16.7 |
| `--ink-2` | `0.43 0.014 268` | `#4c5058` | captions, notes, labels | 7.8 |
| `--ink-3` | `0.60 0.012 268` | `#7d8088` | axes, ticks, line numbers, the faded | 3.8 |
| `--rule` | `0.84 0.008 268` | `#c8cad0` | hairline rules | — |
| `--rule-2` | `0.925 0.006 268` | `#e4e6ea` | faintest structure | — |
| **`--value`** | **`0.43 0.125 268`** | **`#334994`** | **value ink**: activations, parameters | 8.0 |
| `--value-deep` | `0.37 0.11 268` | `#273a79` | value ink for small type | 10.3 |
| `--value-tint` | `0.915 0.03 268` | `#dae3f8` | value at ~15 %: selection wash | — |
| **`--covector`** | **`0.61 0.17 33`** | **`#d55339`** | **covector ink**: gradients, adjoints | 3.95 |
| `--covector-deep` | `0.52 0.16 33` | `#b23a23` | covector ink for small type | 5.7 |
| `--covector-tint` | `0.935 0.03 42` | `#fce4da` | covector at ~15 % | — |

The inks are natural pigments, indigo and cinnabar (vermilion), the pair of the rubricated
book and the Japanese woodblock. The red pen marks the corrections, and gradients are
corrections. Overlapping marks **overprint**: `mix-blend-mode: var(--overprint)` (`multiply` on
paper, `screen` at night) on mark groups, so where two arcs cross the ink darkens as it would
on a press.

**Night** (`prefers-color-scheme: dark`, or the paper/night toggle; `data-ink` on `<html>`)
keeps every hue angle and turns lightness over: paper `0.195 0.012 268` (#13151a), ink
`0.93 0.008 85` (#eae7e2), value `0.76 0.10 268` (#96aff1), covector `0.73 0.14 38` (#f18664).
Traces and other monochrome images are inverted (`invert(1) hue-rotate(180deg)`) rather than
redrawn. Print is always the paper palette.

Rules: text in a spot ink uses the `-deep` variant. Tints are washes for selection only, never
for data. No gradients, no transparency ramps that encode data (opacity only fades the
unselected).

## 3 Type

| voice | face | why | where |
|---|---|---|---|
| prose | **Literata** (variable, `opsz` 7–72, `wght` 200–900, roman + italic; TypeTogether, OFL) | a book face with real optical sizes: sturdy at caption size, fine in display; Greek, small caps, old-style and tabular figures | text, headings, captions, notes, tokens in figures |
| maths | **XCharter Math** (OFL) via native MathML | Charter's sturdy forms sit with Literata's; x-height matched (`font-size-adjust: ex-height 0.5`) so inline symbols read as words | all maths |
| labels | **Archivo** (variable, `wdth` 62–125, `wght`; Omnibus-Type, OFL) | a grotesque with tabular lining figures, a slashed zero and a width axis for dense labels | figure labels, numbers, UI, code headers |
| code | **JuliaMono** (OFL) | the language's own face: complete Unicode for `θ̄`, `ᵀ`, `√`, `⊙`, subscripts | code, shapes `D×N` |

Fonts are self-hosted from `web/src/lib/fonts/` (subset WOFF2, licences alongside), rebuilt by
`web/scripts/build-fonts.sh`.

**Maths.** LaTeX in the source (`$…$`, `$$…$$`) is rendered at build time to MathML by
**Temml** (a KaTeX fork that writes MathML Core: same syntax and macros), and set by the browser
in XCharter Math. KaTeX's HTML output is bound to its own Computer Modern fonts, so maths
matched to the text face requires MathML. Chromium renders it in both the web book and the PDF.
Book macros (`web/src/lib/server/macros.js`): `\R \T \softmax \LN \relu \lse \diag \one \E \KL
\adj{x} \inner{a}{b}`.

**Scale (web).** Body 17–19 px fluid (`--fs-body`), leading 1.56; notes/captions 0.8×; labels
12.5 px Archivo; micro labels 11 px; h2 1.5×; claim 21–27 px italic; chapter title 54–120 px
weight 300, `opsz` 72. **Print** (§8): 9.6/13.6 pt, notes 7.8 pt, labels 7 pt, code 7.1 pt.

**Habits.**
- Text figures are old-style (`onum`); tables, labels and code use lining tabular figures.
- Small caps (`all-small-caps`, +0.06–0.1 em) for structural words: "Chapter five", "How to
  read", the names-table heads, exercise labels. No tracked capitals in colour.
- The first line of each chapter is set in small capitals.
- **The three names.** In the `names` table the maths name is roman, the physics name italic,
  and the ML name is set in the label face between quotation marks: jargon is shown as jargon.
- Numbers: minus sign U+2212, never a hyphen; two decimals in figures (`fmt`); dense tables
  drop the leading zero (`.37`, `−.12`, `fmtDense`).
- The space character is drawn as `␣` in figures.

## 4 The web page

**Grid.** A main column (`--main` 36 rem, ~62 characters) and a margin column (`--margin`
15.5 rem) with a gap of 2.75 rem; together they are `--wide`, centred, so the text column sits
left of centre (Tufte). Flow content sits in the main column. Margin material floats right
into the margin column and never pushes text:

- `::: aside` → `<aside class="note">`, set beside the block that **precedes** it in the source
  (the filter moves it so its top aligns with that block); several asides stack.
- `^[…]` → numbered sidenote beside its reference.
- A "how to read" key and any aside written directly after a figure sit beside that figure's
  caption.

**Figure sizes** (`size=` attribute or the component's `meta.size`): `main` (text column),
`wide` (text + margin, the default for figures), `full` (bleed to the window). Code excerpts
are `wide` so their comment columns fit.

**Breakpoints.** ≥ 74 rem: full grid. 48–74 rem: notes fold into the text under a hairline at
the left, after their block. < 48 rem: one column, figures at the full width of the screen.
Figures lay out from their measured width; they do not scale text.

**Running head.** Book title (italic, left), chapter and current section (Archivo, centre),
and contents, ←/→ and the ink toggle (right). It gets a hairline when the page scrolls.
It does the same job as the running head of the printed page.

**Chapter opening.** Frontispiece slot (§9); then "Chapter five" in small caps; the title
(weight 300, `opsz` 72) with the chapter numeral hanging in the left margin in `--ink-3`; the
claim in italic; the three-names table. **End of chapter:** previous / contents / next.

**Cover** (`/`). A title spread. On the verso is the frontispiece, `traces/cover.svg`, every
θᵢ(t) of the specimen's training. The recto is the title page. Its texture is **θ itself**:
all 1007 trained parameters set as type in storage order, names inline (E, γ₁, β₁, W_q, …),
the weight of the type ∝ |θᵢ|; pointing at a number names it (`W_q[3, 5] = −0.1371 · θ₁₂₃`). A
facts block gives the corpus, shape, loss (start → end, against the entropy floor) and greedy
continuation, all read from the artifact. Below it are the contents. Nothing on the cover is
decoration.

**Contents** (`/contents/`, and below the cover). Numeral, title, gist. Written chapters link;
planned chapters are grey and say "in preparation". Nothing links to text that does not exist.

## 5 Glyphs

Every glyph is a component in `web/src/figures/primitives/`. Scales come from `scales.js` and
are computed **from every number the glyph will be compared with** (small multiples share a
scale; never scale per glyph).

| glyph | component | encoding | sign | ink |
|---|---|---|---|---|
| **vector** | `SpokeGlyph` | n spokes from the centre at 2πk/n, k = 1 at 12 o'clock, clockwise; length ∝ \|x_k\| (`spokeScale(rmax, max)`); faint dots at `rmax` mark full scale and spoke positions | tip filled = +, open = − | value: dot tip; covector: bar tip (the bar of x̄) |
| **matrix** | `MatrixGlyph` | Hinton diagram: square area ∝ \|w_ij\|/vmax, rows top→bottom, columns left→right, as written | filled = +, open = − | value: solid; covector: hatched |
| **attention, arcs** | `Arcs` | reader i, one arc to each source j ≤ i, stroke width ∝ A_ij (weight 1 ↦ `wmax`); arcs of constant aspect, so they nest and never cross; a self-read is a loop; labels at the apex for A ≥ 0.1 | — | value |
| **attention, matrix** | `TriangleDots` | rows = readers, columns = sources; dot area ∝ A_ij; **nothing at all above the diagonal**. Signed mode for scores S: area ∝ \|S\|/max, the upper triangle drawn while it still exists | filled/open | value |
| **scores** | (in figure) | lollipops about a zero line; up = larger score = lower energy | filled/open head | value |
| **weights, probabilities** | (in figure) | bars from a common baseline, height ∝ p; labelled ≥ 0.1 in bold | — | value / covector hatched |
| **θ as type** | `ThetaField` | each parameter a number set in Archivo; type weight ∝ \|θᵢ\| (one reference for all θ); storage order; block names inline | minus sign | value |
| **distribution as energy ladder** | *spec* | vertical energy axis E = −z; one rung per outcome at its energy, labelled directly; occupation bar along the rung ∝ p; temperature stretches the axis; dragging a rung changes its logit; the ladder as a whole slides (shift invariance) | — | value |
| **residual stem** | *spec* | the stream as a vertical stem (forward = down); each block branches off to the side and merges back at a + node; branch width ∝ ‖update‖ per token column | — | value |
| **loss curve** | *spec* | step (log x) vs nats; a hairline at ln V labelled *uniform guessing* and one at the entropy floor; samples set in the margin at their step, joined by leaders | — | value |
| **pullback flow** | *spec* | the forward diagram mirrored: covector marks travel up the same paths, hatched/barred, labelled x̄ | — | covector |

Shared figure classes live in `web/src/lib/styles/figures.css`: `.tok` for corpus characters
(Literata), `.lbl` for labels (Archivo, tabular), `.lbl--micro`, `.lbl--head`, `.axis`, `.rule`,
`.leader`, `.hit` (invisible pointer targets), `.handle` (draggable), `.readout` (state in words),
and the hatch paint servers `url(#hatch-covector)` and `url(#hatch-ink)` defined once per page.
Stroke widths: `--hair` 0.6 px (ticks, rules, leaders), `--line` 1.1 px (data lines), `--heavy`
2 px (the selected). Tokens in figures are Literata; numbers are Archivo; symbols in labels
are Literata italic with real subscripts (never Unicode combining marks: `x̄` is set as a
drawn bar).

## 6 "How to read" keys

The first time a glyph appears in the book, its figure carries a key: **one enlarged instance
of the glyph drawn from the figure's own data**, with straight hairline leaders (`--hair`,
`--ink-3`) from a 1.5 px dot on the mark to a short plain-language note (Archivo 11 px,
`--ink-2`). The heading "How to read" is set in small caps. The key sits in the margin column
beside the caption (in print, the outer margin). Component: `HowToRead` (notes
`{ax, ay, side, len, text}`). A figure module exports it as `Key` (§10). Keys are about
reading, not about the result: "area ∝ weight", "nothing above the diagonal", "filled tip:
positive".

## 7 Motion

- Motion only where something happens in time: a forward pass, a backward pass, a training
  step, sampling, and the masked softmax as a sequence of operations.
- **Every animation is a pure function of one clock t ∈ [0, 1]**, held by a `Scrubber`: a
  hairline track, phase ticks with their names, a playhead the reader can drag. Play runs t at
  a constant rate (2.4–3.6 s for a full pass); there is no autoplay.
- **Forward runs in value ink, downward; backward runs in covector ink, upward.** A scrubber in
  a backward figure uses `ink="covector"`.
- Between phases, marks move along their own data paths (a score's dot shrinks to zero and
  grows into a probability). Nothing flies in, bounces or fades for effect. Easing:
  `cubic-bezier(.2,.7,.2,1)` for discrete jumps; linear in t while scrubbing.
- **Reduced motion** (`prefers-reduced-motion: reduce`): CSS transitions are 0 ms; Play steps
  from phase to phase with no interpolation; dragging still works (the reader's own motion).
- **Films** are the same components captured headless at fixed t: `/figures/<id>/?film&t=…`
  (§10).
- **In print an animation becomes small multiples of its named phases** (e.g. scores → mask →
  softmax).

## 8 Print

The print edition is the web book's HTML under `web/src/lib/styles/print.css`, paginated by
Paged.js in headless Chromium. There is **one renderer**. The seam for the PDF tool:

- **`/print/`** renders the whole book in order: half-title, frontispiece (verso) facing the
  title page, colophon, contents, then each chapter, with its trace on a verso and its opening
  on the facing recto. Every figure is in `mode="print"`.
- When fonts are loaded and figures have laid out at print width, `<html data-print-ready="1">`
  is set and the `pico:print-ready` event fires. Then paginate and print. Verified recipe
  (Paged.js 0.4.3, headless Chromium, viewport 1400 px): collect the page's CSS (the `<style>`
  elements), take `.print-edition-root`'s `outerHTML`, empty `<body>`, and call
  `new Paged.Previewer().preview(html, [{ [location.href]: css }], container)`. It produced the
  20-page proof in 1.7 s. Then `page.pdf({ preferCSSPageSize: true, printBackground: true })`.
  (Injecting `paged.polyfill.js` into the already-hydrated page produced no pages.)
- Paged.js 0.4 cannot rewrite a selector with `:first-of-type` after a `+` combinator ("item
  doesn't belong to list"). Avoid that pattern anywhere in the CSS.
- The print layout is scoped to `.print-edition-root`, not `@media print`, because Paged.js
  renders in screen media. It is also a faithful proof on screen.

**Trim and grid.** 189 × 246 mm (crown quarto). Margins: top 18, bottom 21, inner 21, outer
15 mm. Text block 153 mm = main column 104 mm + gap 5 + notes 44. The notes column sits in the
**outer** margin: on versos the text column moves right and notes float left
(`.pagedjs_left_page` rules). Without Paged.js the notes stay on the right.

**Type.** Literata 9.6/13.6 pt; notes and captions 7.8 pt; labels 7 pt; code JuliaMono 7.1 pt;
h2 15 pt; opening title 38 pt with the numeral in the margin; claim 13.5 pt italic. Hyphenated,
orphans and widows 2.

**Running heads** (`string-set`): verso shows the folio at the outer corner, then the chapter in
small caps; recto shows the current section in italic, then the folio. Old-style folios in
Literata 8 pt. Named pages without heads: `front` (half-title, title, colophon, contents) and
`frontis` (trace plates). The opening page clears the section head with an empty `string-set`
marker (`.rh-clear`) instead of using a named page, because switching named pages
mid-chapter makes Paged.js insert a page. Chapters open on a recto. The page break sits on the
`<article>`: `break-before: left` when the chapter has a trace (so the trace falls on the
verso facing the opening), `right` otherwise (the verso stays blank). Pages inserted by
breaks are `:blank`, with no head and no folio.

**Figures in print.** Static state; no scrubbers, readouts, hit areas or handles. Animations
become small multiples of their phases. Figures lay out at `width / 0.78` and scale to the
column, so a 12 px label prints at ≈ 7 pt (`PRINT_SCALE` in `scales.js`). `break-inside:
avoid`; the key goes in the outer margin beside the caption. Colour is exact
(`print-color-adjust: exact`). For one-colour printing the shape rules of §1.5 keep value and
covector apart.

**Code** keeps its line numbers (they are the real lines of the file) in the gutter; chunks
avoid breaking. Cross-references print without underlines; the contents show page numbers
through `target-counter`.

## 9 Traces (frontispieces)

Each chapter opening, and the cover, has a slot for a **trace**: a plotter-style drawing of a
real computation, made by the orchestrator (BRIEF §11). Drop `web/public/traces/<slug>.svg`
(slugs: `cover prologue task vectors energies surprise attention block pulling-back learning
speaking epilogue`); the page picks it up. Until a file exists the slot is omitted on the web
(dev shows a dashed outline naming the file) and the print verso is blank.

Drawing rules, so the trace belongs to the book:

- **No background.** Transparent; the book's paper shows through, and night mode inverts the
  strokes. (A full-bleed background `<rect>` is stripped by `/frontis/<slug>.svg` as a stopgap.)
- **Ink** `#181a20` (`--ink`); **at most one accent**, chosen by meaning: `#334994` (value) if the
  trace records forward quantities or parameters, `#d55339` (covector) if it records gradients
  or adjoints. (`#b23a23`, covector-deep, for very fine accent strokes.)
- **Strokes.** Draw in millimetres at print size. The plate is printed at up to 159 × 214 mm,
  so use a viewBox in mm (e.g. `0 0 150 200`). Hairline 0.10 mm (≈ 0.28 pt), the base for
  dense families; emphasis 0.18 mm; nothing heavier than 0.35 mm. Round caps and joins.
  Density encodes time or multiplicity through `stroke-opacity` ≥ 0.04 (screen and PDF) or
  repetition (plotter).
- **No text, no frame.** Put the caption in `<title>` (what was computed) and `<desc>` (data
  and parameters). The page shows them as the frontispiece caption.
- Keep files under ~2 MB for the web (simplify paths below 0.05 mm deviation). Plotter files
  may be larger and live elsewhere.

## 10 The figure contract

A figure is `src/figures/<NN-chapter>/fig-<id>.svelte`. The prose places it with
`::: {.figure #fig-<id> key=value …}` and the div's content is the caption. It is

1. **A pure function of state.** Inputs: the specimen (via `getContext('specimen')()`, the
   adapter in `src/lib/model/specimen.js`), its `attrs` (the div's attributes, strings, 1-based
   positions as in the prose), its internal state (reader, window, t, τ …) and its measured
   width. No hidden randomness; no fetches; numbers only from the specimen or from
   `artifacts/` (through the adapter or a server load). **No invented numbers.**
2. **Two modes.** `mode="interactive"` (web) and `mode="print"` (static, complete, no
   affordances; animations as small multiples). Both render on the server (SSR), so the HTML
   already holds the static figure; hydration adds interaction.
3. **Sized by its frame.** Lay out from the measured width (`use:measure`); use
   `layoutWidth(width, mode)` so print labels come out at 7 pt.
4. **Labelled directly.** A `.readout` paragraph states the current state in words and numbers
   (the live caption); the static caption lives in the prose.
5. **Accessible.** `role="img"` with an `aria-label` that states the numbers; keyboard
   equivalents for every pointer interaction (arrow keys, Enter); visible focus.
6. **Exports** from `<script module>`: `meta = { size, title }` and, if the figure introduces a
   glyph, `Key` (a component, usually `fig-<id>.key.svelte`).
7. **Film-ready.** If it animates, its clock is a prop or state t ∈ [0, 1] (the `Scrubber`
   binds it), so a capture tool can set t frame by frame.

Each figure can be viewed alone at `/figures/<id>/` (interactive) and `/figures/<id>/?print`
(static).

## 11 Adding chapters and figures

**A chapter** is `book/chapters/NN-slug.md`, with a slug from the outline (§4) and a chapter id
`ch-<slug>`. It contains `# Title {#ch-slug}`, then `::: claim`, then `::: names`, then sections
(`## Title {#sec-…}`; `{.unnumbered}` to opt out of numbering). Cross-reference with `@fig-…`,
`@sec-…`, `@ch-…` and `@ex-…`; planned chapters resolve to "Chapter n" without a link. Code
comes in only as `::: {.code file="src/x.jl" fn="name"}` (also `struct=`, `region=`, `lines=`,
`method=`, `doc=true`). Maths is LaTeX. Never paste a number that a figure can compute.

**A figure.** Add the `::: {.figure #fig-id}` div, add the entry in `book/figures.md`, and write
`src/figures/NN-slug/fig-id.svelte` against the primitives. Until the component exists the
book shows "Figure in preparation". Check it at `/figures/fig-id/` in both modes, on a phone
width, at night, and with reduced motion.

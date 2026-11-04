# print/ — the print edition

The PDF is the web book's own HTML (the `/print/` route) under `web/src/lib/styles/print.css`,
paginated by **Paged.js 0.4.3** in headless **Chromium** (Playwright). One renderer for web and
print; no LaTeX, no Typst. Design: `../DESIGN.md` §8 and the header of `print.css`.

```sh
cd print
npm install            # once: pagedjs, playwright (uses the Chromium in ~/.cache/ms-playwright), pdf-lib
npm run pdf            # → print/PicoGPT.pdf
npm run proof          # … plus print/proof/: pages/p-NNN.png, contact-sheet.png, contact-sheet-grey.png
npm run pdf:reuse      # skip the site build, reuse print/.work/site/build (fast CSS iteration)

node build.mjs --tolerant --proof   # while chapters are being written: a figure component that does
                                    # not compile is left out (shown "in preparation"), with a warning
node build.mjs --strict             # warnings are errors (use for the final book)
```

Needs `web/node_modules` (`npm install` in `web/`) and pandoc ≥ 3 (as the web build does).
`--proof` also needs `pdftoppm` (poppler) and ImageMagick `montage`. `SOURCE_DATE_EPOCH` fixes the
proof date and the PDF dates for a reproducible build.

## What `build.mjs` does

1. **Stages** `web/` into `print/.work/site` (node_modules symlinked) and prerenders it with Vite,
   `BOOK_ROOT` pointing at the repository, so a running `npm run dev` or another build in `web/` is
   never disturbed, and the PDF is built from one consistent snapshot of the sources.
2. **Serves** the static build on a throwaway localhost port.
3. **Paginates.** Chromium (1400 px viewport, reduced motion, hinting off) opens `/print/`, waits for
   `<html data-print-ready="1">`, injects Paged.js and `paginate.js`, and runs the Paged.js
   Previewer on the page's own HTML and CSS. Cross-reference page numbers change the layout they
   describe, so it repeats (fresh page load each pass) until they stop changing, as LaTeX reruns.
4. **Checks**, and fails loudly on: a build error, `/print/` not becoming ready, a page error, a
   Paged.js layout loop, text or a figure lost off a page, a figure that cannot be made to fit, a
   deferred figure or margin note that found no page, a font that fails to load or is not
   embedded, a page-count or trim mismatch. Everything else (holes, figures printed reduced,
   unresolved references, figures in preparation, fallback fonts) is a warning; `--strict` fails
   on warnings. The full list goes to `.work/report.json`.
5. **Prints** with `page.pdf` (CSS page size, tagged PDF), then **pdf-lib** adds the outline
   (chapters and sections as bookmarks), document metadata, `PageLayout /TwoPageRight` and
   `PageMode /UseOutlines`.
6. `--proof`: page PNGs at 110 dpi and colour and greyscale contact sheets (spreads, 8 pages a row).

## What `paginate.js` adds to Paged.js

Paged.js 0.4 does CSS Paged Media (`@page`, named pages, `target-counter`, `break-*`) but has no
page floats and no margin notes. `paginate.js` registers handlers that supply them:

- **Figure floats.** A figure is never split. If it does not fit in the rest of its page and that
  would leave a hole of more than 3½ lines, it is taken out of the flow and put back into the
  source exactly where the next page starts (splitting the paragraph there, if the page broke
  inside it); the text that follows fills the hole, and the figure heads the next page, as TeX
  defers a float. A figure with no more than about three lines above it moves to the head of its
  own page. A figure drifts past its chapter's exercises only rather than leave over half a page
  empty. A figure taller than the text block is scaled to fit and reported.
- **Margin notes** (`::: aside`, `^[…]`) are page furniture. Each is replaced by an empty mark
  before pagination; after a page is built, the notes whose marks are on it are set in the notes
  column at their anchor's height, stacked, kept off wide figures, code and tables, and pushed up
  from the foot. A note with no room left is carried to the head of the next page's margin.
  Paged.js therefore never splits a note or pushes text away for one.
- **Running heads and folios**, drawn on each page: verso = folio at the outer corner over the
  notes column, chapter in small caps over the main column; recto = the first section that begins
  on the page (or the one in force) in italic, folio at the outer corner. Chapter openings carry
  a drop folio at the foot; blank pages, plates and front matter carry nothing.
- **Typesetting fixes**: the first block on a page loses its top margin; a word is never divided
  across a page turn; a paragraph keeps the indent it has in the unbroken text; the chapter's
  first paragraph opens with a line of small capitals, and no other; code breaks across pages
  only between slices of at least three lines, each with its own line numbers; display maths
  wider than the main column spans the text block; cross-references to figures, sections and
  exercises on another spread get "(p. N)".

- **Paint that prints as vector.** Chromium's PDF backend cannot write the rotated SVG hatch
  pattern (`#hatch-covector`) and paints every hatched mark with a page-sized 72 dpi raster; it
  also rasterises blend modes over translucent marks. So in print the hatch fills become real
  lines at the pattern's spacing, angle, ink and weight, cut exactly to rectangles and circles
  (clipped for other shapes). Hatched strokes become the same ink dashed into bars across the
  band. Drawings that contain translucent marks lose their multiply blend. The paint servers from
  `+layout.svelte` are carried into the book, and the `url(#…)` references that Paged.js rewrites
  against the stylesheet are restored. The build warns if any raster image appears in the PDF.

## Debugging

- `node build.mjs --skip-build --keep-open` leaves the browser running after printing.
- The `/print/` page is a faithful proof on screen too (`npm run dev` in `web/`, open
  `/print/` at ≥ 1280 px): `print.css` is scoped to `.print-edition-root`, not `@media print`,
  because Paged.js renders in screen media (and drops every `@media` block except `print`).
- `.work/report.json` lists every warning, every float decision and the fonts in the PDF.

# Brief 03 — an audition edition that prints one chapter (Sol)

PicoGPT is a book (pandoc markdown → SvelteKit web book and a Paged.js print PDF) about a GPT
written in Julia. The working directory is the repository root, a git worktree. Read `README.md`,
`print/README.md`, `jspace/README.md` (section "How an edition works"), `jspace/edition.js`,
`web/src/lib/editions/book.js`, `web/vite.config.js`, `web/src/figures/registry.js`,
`web/src/lib/server/book.js`, `web/src/lib/server/prose-values.js`.

## Goal

The book's author will read a **draft revision of Chapter 5 (Attention)** as a PDF set in the
book's own design and built by the book's own pipeline. Make a small edition, folder
`attention-audition/`, whose outline is that one chapter, and prove that
`node print/build.mjs --edition=attention-audition` builds a correct PDF. Until the draft exists,
use a **copy of the current `book/chapters/05-attention.md`** as the chapter
(`attention-audition/book/chapters/05-attention.md`); a writer will replace it later.

## Requirements

- `attention-audition/edition.js`: OUTLINE, BOOK, PATHS, FRONT, PRINT, `proseValues`,
  `artifacts` as needed. Title page: e.g. title "Attention, from the top down", subtitle "A draft
  revision of Chapter 5 of PicoGPT, for its author to read" (author Tobias J. Osborne, as in the
  other editions). FRONT minimal (no θ, facts or map unless trivially available); a provenance
  sentence. PRINT.out = `attention-audition/AttentionDraft.pdf`, work/proof under
  `attention-audition/.work` and `.../proof` (git-ignore those two, not the PDF).
- **The chapter keeps its number 5** ("Chapter five" in the opening) and its id `ch-attention`.
- **Cross-references to the main book's other chapters must resolve** to their real numbers
  (e.g. `@ch-energies` → Chapter 3, `@ch-block` → Chapter 6, `@ch-pulling-back` → Chapter 7,
  `@ch-task` → Chapter 1), without links to pages that do not exist in this edition and without a
  contents page full of "in preparation" entries. Find the cleanest way the existing machinery
  allows (e.g. the OUTLINE listing the main book's chapters but only 05 present, and print
  showing only present chapters — check what the print route actually does). Section references
  into other chapters (`@sec-…`) should be avoided by the writer; tell me what happens if one
  appears.
- **Figures: reuse the main book's** `fig-attention-reads`, `fig-attention-pipeline`,
  `fig-attention-triangle` (and their how-to-read keys) **without adding or changing files under
  `web/src/figures/`**. E.g. PATHS.figures = `attention-audition/figures`, holding thin wrappers
  that re-export the main book's components (default, `meta`, `Key`), or whatever works. A new
  figure for this edition will later be added in `attention-audition/figures/05-attention/` by a
  designer; make sure a new `fig-*.svelte` there would be picked up.
- **Computed prose values**: `{{value:attentionParams}}` and every main-book value the chapter uses
  must work (the edition's `proseValues` replaces the default, so re-provide them, ideally by
  calling the main book's function rather than copying it). Also read
  `attention-audition/artifacts/mixture.json` **if it exists** and expose a few values from it
  (names to be agreed later; a colleague is producing that file right now, do not write it). Make
  the edition work whether or not the file is present.
- Frontispiece/trace for the chapter: reuse the main book's chapter-5 trace if one exists.
- **Do not change the shared pipeline** (`web/src/lib`, `web/src/routes`, `web/vite.config.js`,
  `print/`). If something cannot be done without a change there, make the smallest additive change,
  keep the main book's build byte-for-byte unaffected in behaviour, and report it prominently.
- The worktree has no `node_modules`: symlink `web/node_modules` and `print/node_modules` from
  `/home/tobias/Projects/PicoGPT.jl/web/node_modules` and `.../print/node_modules` (do not commit
  the symlinks; check `.gitignore`). Do not write anything in `/home/tobias/Projects/PicoGPT.jl`.
- Build with `--strict` if achievable, otherwise say which warnings remain and why. Run with
  `--proof` once and **look at the page images** (opening page, figure pages, the code excerpts,
  the exercises): the chapter must look like a chapter of the book.
- A tiny `attention-audition/README.md`: what this folder is, how to build the PDF, how to look at
  it on the web (`BOOK_EDITION=attention-audition npm run dev` in `web/`, if that works).

Write only in `attention-audition/` (except the git-ignored symlinks). Other agents write
`attention-audition/notes/`, `scripts/`, `artifacts/`, `briefs/` at the same time; do not touch
those. Do not commit.

Final report: files made, the exact build command, warnings left, the PDF path and page count,
which proof pages you looked at, and any pipeline change (ideally none).

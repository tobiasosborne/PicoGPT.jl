# web/ — the interactive book and the source of the print edition

SvelteKit (Svelte 5), prerendered to static HTML (`adapter-static`). Design rules: `../DESIGN.md`.
Figure catalogue: `../book/figures.md`. Engine: `src/specimen/` (owned by the engine; see its README).

```sh
npm install
npm run dev          # http://localhost:5173  (reloads on book/, src/*.jl, artifacts/, public/traces/)
npm run build        # → build/  (static site; BASE_PATH=/sub/path for a non-root deploy)
node scripts/shots.mjs /attention/ name [--full] [--mobile] [--dark] [--reduced] [--print]
```

Needs `pandoc` ≥ 3 (PATH, `~/.local/bin`, or `PANDOC=`). `BOOK_STRICT=1` turns missing
excerpts, maths errors and unresolved references into build errors.

## Routes

| route | what |
|---|---|
| `/` | cover (title spread) + contents |
| `/<slug>/` | a chapter, from `book/chapters/NN-<slug>.md` |
| `/contents/` | contents |
| `/print/` | the whole book for the PDF: front matter, chapters, every figure static (DESIGN.md §8) |
| `/figures/`, `/figures/<id>/` | every figure alone; `?print` static; `?film&t=0.5` + `window.__pico.setT(t)` for frames |
| `/frontis/<slug>.svg` | a trace from `public/traces/`, background removed |

## Pipeline

`src/lib/server/book.js` runs pandoc with `book.lua` on a chapter, renders maths with Temml
(MathML, set in XCharter Math), pulls `::: {.code}` excerpts from the Julia source
(`julia.js`: token-level, finds every method of `fn=` by name), resolves `@fig-/@sec-/@ch-/@ex-`
references through a book-wide index, and splits the HTML at figure slots. The chapter page
mounts each slot's component (`src/figures/registry.js`: file `fig-<id>.svelte`, found by name).
Everything renders on the server, so the HTML already contains every figure in its static
state; hydration adds the interaction.

Specimen numbers come from `artifacts/specimen.json` when present (`server/specimen-data.js`),
else from the JS engine with a provisional recipe, labelled as such on the page.
Figures read them only through `src/lib/model/specimen.js`.

## Layout of the source

```
src/lib/styles/   tokens.css (palette, type, grid) · base · book · figures · shell · print · fonts
src/lib/shell/    running head, chapter opening, frontispiece, figure frame, contents, θ field
src/lib/server/   book.js · book.lua · julia.js · macros.js · specimen-data.js
src/lib/model/    specimen.js — the only door from figures to the engine
src/figures/      primitives/ (glyphs, keys, scrubber, scales) · NN-slug/fig-<id>.svelte
public/traces/    frontispiece traces (DESIGN.md §9)
scripts/          build-fonts.sh · shots.mjs
```

# jspace/ — *Poised to speak*, a supplement on the Jacobian lens and the J-space

A short book that explains the mathematics of Anthropic's paper *Verbalizable Representations Form a
Global Workspace in Language Models* (Gurnee, Sofroniew, Lindsey et al., Transformer Circuits
Thread, 2026; transformer-circuits.pub/2026/workspace) by computing every object in it on the two
models of PicoGPT: the specimen (1007 parameters) and the Tiny Shakespeare model. It is not part of
the main book or its PDF; it is a second **edition** built by the same pipeline, with the same
design system (`../DESIGN.md`) and the same conventions (`../BRIEF.md` §7).

```
edition.js            outline, title, front matter, data hooks (what makes this an edition)
book/chapters/        the prose, pandoc markdown, one file per chapter
figures/              the figures, Svelte, against web/src/figures/primitives
traces/               frontispieces (plotter-style SVG), optional
src/JLens.jl          the analysis: averaged Jacobians, lenses, J-space, interventions, broadcast
scripts/              export_specimen_jlens.jl, export_shakespeare_jlens.jl → artifacts/*.json
test/runtests.jl      tests of the analysis
artifacts/            the exported data every figure reads (README inside)
briefs/               the briefs given to the agents that built this, and their reports
PoisedToSpeak.pdf     the print edition
```

```sh
julia --project -t auto jspace/scripts/export_specimen_jlens.jl       # data (seconds)
julia --project -t auto jspace/scripts/export_shakespeare_jlens.jl    # data (minutes; needs the checkpoint)
cd web && BOOK_EDITION=jspace npm run dev                             # the web edition at :5173
node print/build.mjs --edition=jspace --strict --proof                # → jspace/PoisedToSpeak.pdf, jspace/proof/
```

## How an edition works

`web/vite.config.js` reads `BOOK_EDITION` (default `book`) and aliases `#edition` to this folder's
`edition.js` and `#edition-figures` to `figures/`. `web/src/lib/outline.js` re-exports the edition's
`OUTLINE`, `BOOK`, `PATHS`, `FRONT` and `PRINT`; `book.js` reads chapters and traces from `PATHS`;
the figure registry globs `#edition-figures`; `artifacts.js` and `prose-values.js` call the
edition's `artifacts(ids, read)` and `proseValues(read)` if it defines them. The print build takes
`--edition=jspace` and writes to `PRINT.out`. Nothing in the main book changes when the variable
is unset; the main edition's own module is `web/src/lib/editions/book.js`.

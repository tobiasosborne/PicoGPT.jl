# Attention audition

This is a one-chapter reading proof of PicoGPT. Edit
`book/chapters/05-attention.md` in this folder to revise Chapter 5. The
edition uses PicoGPT’s book design, attention figures, and chapter trace.

From the repository root, build the PDF and page proofs with:

```sh
node print/build.mjs --edition=attention-audition --strict --proof
```

Open `AttentionDraft.pdf`. The page images are in `proof/pages/`.

To read the web edition, run `BOOK_EDITION=attention-audition npm run dev`
from `web/` and open `/attention/` on the dev server.

References to other chapters print as plain text with their main-book
numbers. Use chapter references for material outside the audition. Section
references to absent chapters render as literal `@sec-…` citations.

When `artifacts/mixture.json` is available, the chapter can use
`{{value:mixtureReaderCount}}`, `{{value:mixtureLossActual}}`,
`{{value:mixtureLossArithmetic}}`, and `{{value:mixtureLossGeometric}}`.
The three losses are mean cross-entropies in nats, rounded to four decimal
places. Figures can read the export through `getContext('artifacts')().mixture`.

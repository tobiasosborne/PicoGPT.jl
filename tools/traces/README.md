# Traces

Each chapter of the book opens with a trace (BRIEF §11, DESIGN.md §9): a plotter-style drawing
in which every stroke is a path a real computation took. Nothing is drawn by hand.

    node tools/traces/build.mjs            # all of them → web/public/traces/<slug>.svg
    node tools/traces/build.mjs cover      # one

- `specimen.mjs` trains the book's specimen with the browser engine (`web/src/specimen`) and
  calls an observer before every update.
- `svg.mjs` writes millimetre-scale, stroke-only SVG with `<title>` and `<desc>` captions. The
  same files can be sent to a pen plotter.
- One file per trace, named after its chapter slug. The header comment of each file says what
  it records and how to read it.

# Follow-ups after the Muse review (6 November 2026)

`review-muse.md` has 30 findings. The orchestrator applied the ones about mathematics, honesty
and wording in the prose and the names tables (findings 1–9, 11–14, 16 in part, 18–22, 27–28,
30, the subtitle, the TV rounding). What remains is design work and small analyses, left for a
later pass so as not to redraw figures in a hurry:

- **Figures 2.1–2.2 (finding 10):** `MatrixGlyph` encodes area ∝ cell²·|v|/vmax, so the triangle
  and the averages are not on one scale; pass an absolute area scale and redraw; audit every
  caption saying "one scale".
- **Two inks (15):** the cone (Fig. 4.1) is hatched in covector ink though it is a set of values;
  chapter 6's statistics use the inks as a legend; Fig. 5.4 draws v_a hatched where the written
  vector is a value. Restore DESIGN §1.1.
- **Cone figure (23):** caption should say "computed exactly, drawn in axonometric projection";
  the key should be an enlargement of the printed pair; the 21-pair triangle is unreadable at
  print size (use the numbers).
- **Keys (25):** Figs 0.1, 1.3, 3.1 ("Reminder"), 3.3 explain the wrong marks; the caret of
  Fig. 0.1 sits between two characters.
- **Print legibility (26):** Fig. 1.2 (agreement by grey only), 4.4 (occupancy mark at k = 1 on
  the axis), 6.4 (head labels collide), 2.1 (ASCII underscores in the title, crowded labels).
- **Notation table (17):** a one-page table for the supplement; reserve K for Jacobian blocks,
  rename CKA Grams, N_a → 𝒩_a, keep s, u for swaps.
- **Layout (29):** sentences split by full-page figures (pp. 56–57, 68–69, 33–34, 58–59);
  running heads that anticipate; two code chunks that break; blank versos where traces would be
  (`jspace/traces/` is empty; `specimen-jlens-training.json` is exported for a cover trace).
- **Small analyses:** leave-one-window-out J for the specimen's 55-vs-39 headline (finding 6);
  the present-only lens for the GREMIO colon count (5); per-position occupancy as the paper
  counts it (8); the autocorrelation band statistic (2).
- **Code tidy:** `figures/05-writing/jdata.js` duplicates `figures/primitives/jdata.js` with a
  different `fibre` signature; merge.
- **Dictionary:** the stray "· —" after "top-1 agreement" comes from an empty physics cell
  ("—"); the `::: dictionary` filter in `web/src/lib/server/book.lua` should skip it.

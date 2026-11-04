# Integration fix-ups (collected by the orchestrator; to be done in the integration pass)

- [ ] `/figures/<id>/` route: `@ch-…` cross-references in captions resolve to `../slug/`, which
      breaks prerendering. Rewrite links in the standalone figure route.
- [ ] Ch. 4 `fig-gradient-p-minus-e`: the first covector key now lives in the prologue
      (`fig-two-inks`), so turn this one into a reminder.
- [ ] Print: add the contents map (`Contents.svelte`) as a plate in the print edition.
- [ ] Shared `Scrubber`: phase labels crowd on narrow screens.
- [ ] Dictionary: some ML-column entries read oddly ("backpropagation through f"). Tidy the
      names tables across chapters.
- [ ] Traces: regenerate all of them under the canonical recipe (BRIEF §13) once the JS engine
      is unified.

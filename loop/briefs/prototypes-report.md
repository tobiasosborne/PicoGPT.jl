# Prototype handoff

Use prefix parity as the specimen, with the exact local transition as the reference. Its front
advances one position per pass and reaches all 20 positions at pass 19. A one-pass local model has
a 70% ceiling at the training length. Rule 60 makes the autonomous companion, with all 24 rows
of the reference impulse checked against Lucas's formula. Stock AdamW training failed to
produce a durable recurrence for either task.

- **Implementation.** [Loop.jl](../src/Loop.jl) reuses the PicoGPT block with one shared parameter
  vector, optional input addition before or after the block, a shared last or per-pass readout,
  and traces of the residual stream, attention and predictions. The controls use one block at
  the same parameter count, distinct blocks at the trained depth, or a one-block MLP widened to
  the untied budget. The adjacent-token variant runs each left/current pair through the same
  PicoGPT block, with a fixed zero boundary.
- **Pullback.** All 58 [loop tests](../test/runtests.jl) passed. Float64 central differences checked
  arbitrary logit covectors and final or deep losses for all three injection modes, plus the
  adjacent and normalized variant. The largest checked relative error was 1.47 × 10⁻⁹.
  Copying one block's weights into three distinct blocks gives the same logits, and their three
  block gradients sum to the tied block gradient.
- **P1 stock.** At T=5 the 582-parameter tied model scored 87.50%, 98.75%, and 69.38%. The
  same-size one-pass model scored 78.75–80.63%, the 2,710-parameter untied model scored 100% in all
  seeds, and the 2,707-parameter shallow model scored 68.75–69.38%. The best tied seed's front
  moved through positions `2,2,3,4,4` and stopped through pass 10. Its T=10 accuracy dropped
  from 74.53% at pass 5 to 73.24% at pass 10. The smaller random width screen peaked at 92.50%
  with 234 parameters.
- **P2 stock.** The same-size tied model reached 98.70% at K=3 in its strongest seed and
  67.71–70.31% in the other two. The one-pass model reached 68.75–82.55%, the untied
  model 92.97–98.70%, and the wide one-pass control 67.71–96.88%. In the best tied seed
  pass 4 reached 61.46%, and pass 12 reached 52.08%. Only pass 3 matched the full impulse row.
- **Local structured variants.** The 522-parameter P1 and 216-parameter P2 both have exact
  evaluated references. The initialization encodes the transition in a ReLU spline and uses an
  adjacent window, no positional code, and fixed normalization at every pass. With the cautious
  schedule P1 scored the first four passes' window targets exactly in all seeds,
  then its front fell from 5 to 2 immediately beyond that depth. P2, trained only on the
  third iterate, preserved five exact passes in all seeds on 24-position inputs and its impulse.
  The larger perturbation and learning rate reduced this to three passes across seeds. This measures how well end-to-end
  optimization preserves an initialized transition.
- **P3.** At 1,808 parameters, two passes reached 81.42–99.05% and one reached 87.80–95.14%
  on the exhaustive four-symbol setting. Even the strongest tied seed scored only 26.22%
  on the one-hop target at pass one. The experiments give weak evidence for a learned hop
  per pass.

The [lab notebook](../notes/prototypes.md) contains exact commands, seed-by-seed controls,
and links to the reproducible numerical artifacts. Training time per model excludes Julia
startup and most compilation. No commit was made.

# Brief 02 — specimen numbers for the mean-field step (Sol)

PicoGPT is a GPT written from scratch in Julia (stdlib only, no autodiff) plus a book about it.
The working directory is the repository root. Read `BRIEF.md` (§2 notation, §3 exact model
semantics, §4 the specimen), `src/attention.jl`, `src/model.jl`, `src/block.jl`, and skim
`book/chapters/05-attention.md`.

A revised attention chapter will present attention as an ansatz for P(next symbol | context).
One step of it is a **mean-field step**: an honest probabilistic model would pick one source
position $j$ at random with probability $A_{ij}$ and predict from that source's value vector
(a *mixture* of per-source predictions); attention instead averages the value vectors first and
predicts once. We want to know, on the real trained specimen, how far apart those are.

## Compute (Julia, from `artifacts/specimen.json`, trained parameters `train.theta_final`)

Use the package's own forward functions wherever possible; do not reimplement the model if a
function exists. The specimen has $H=2$ heads, $T=8$, 10 training windows: 80 readers $(b,i)$.

For every reader $(b,i)$:

1. $p$ = the model's actual next-character distribution at column $i$ (the mean field).
2. For every pair $(j_1,j_2)$ with $j_1,j_2\le i$: replace column $i$ of head $h$'s output
   $O^{(h)}$ by the value column $v^{(h)}_{j_h}$ (head 1 by $v^{(1)}_{j_1}$, head 2 by
   $v^{(2)}_{j_2}$), keep everything else of the forward pass, and run the remaining column-wise
   part of the model (W_o, b_o, residual add, LN2, MLP, residual add, final LN, W_out, softmax) on
   column $i$: $p^{(j_1j_2)}$. Check that only column $i$ can be affected (everything after
   attention is column-wise).
3. Mixture $p^\text{mix}=\sum_{j_1,j_2}A^{(1)}_{ij_1}A^{(2)}_{ij_2}\,p^{(j_1j_2)}$.
4. Geometric mixture $p^\text{geo}\propto\prod_{j_1,j_2}\bigl(p^{(j_1j_2)}\bigr)^{A^{(1)}_{ij_1}A^{(2)}_{ij_2}}$,
   normalised.
5. Single-head variants: mix head 1 only, head 2 held at its actual output; and vice versa.
6. Diagnostics: entropy of each attention row $A^{(h)}_{i,:}$; the weighted spread
   $\sum_jA^{(h)}_{ij}\lVert v^{(h)}_j-o^{(h)}_i\rVert^2$; max weight per row.

Report, over the 80 readers (and listing the worst few by window text and position): KL(p‖mix),
KL(mix‖p), KL(p‖geo), total variation for each pair among p, mix, geo; whether the argmax agrees;
the mean cross-entropy loss of p, mix and geo on the training targets (p's should reproduce the
trained loss ≈ 0.0588 nats, BRIEF §12 — check this). Note that readers at position 1 have one
source and the gap is identically zero there.

**Sanity check of an algebraic identity.** Build an idealised readout whose logits are affine in
the head outputs (e.g. logits $= M o + c$ with any fixed $M$, $c$), and verify numerically, on real
attention rows and values, that the mean-field prediction equals the normalised geometric mixture
of the per-pair predictions to ≤1e-12. Then report how close $p$ is to $p^\text{geo}$ in the real
model (it need not be equal).

**One figure-ready reader.** Pick one or two readers where the attention row is genuinely spread
over two or more sources with different characters (not a near-copy), and export for them: the
window text, $i$, both attention rows, the per-source (single-head-mixed) predictions, $p$,
$p^\text{mix}$, $p^\text{geo}$. Also pick one near-copy reader for contrast. Justify your choice in
the notes.

## Deliverables

- `attention-audition/scripts/mixture.jl` — runnable with
  `julia --project=. attention-audition/scripts/mixture.jl` from the repo root, deterministic,
  writes the two files below.
- `attention-audition/artifacts/mixture.json` — all per-reader arrays (column-major `{shape,data}`
  as in BRIEF §5 where arrays), summary statistics, the chosen figure readers.
- `attention-audition/notes/numbers.md` — a short human-readable account: definitions, the
  summary table, the worst readers, the chosen figure readers, the sanity check result.

Write only inside `attention-audition/scripts`, `attention-audition/artifacts`,
`attention-audition/notes/numbers.md`. Do not change `src/` or anything else; do not commit.
Another agent is building the audition edition in `attention-audition/edition.js` and
`attention-audition/figures/` at the same time — do not touch those.

Final report: the files, the headline numbers, how you verified them (the loss check, the identity
check, the column-locality check), anything surprising.

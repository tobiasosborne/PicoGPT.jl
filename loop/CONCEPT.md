# Not not to be — concept for a third book, on the tiniest looped transformer

Quest *loop*, 21 November 2026. Everything below rests on runs in `loop/` (scripts, artifacts,
notes) and on derivations audited by a second agent (`loop/notes/audit.md`). Lab notebooks:
`loop/notes/prototypes.md` (full-attention prototypes), `loop/notes/window.md` (the light-cone
specimen), `loop/notes/derivations.md` (mathematics), `loop/notes/literature.md` (survey).

## 1. The answer in one page

**What it would look like.** A book about one block applied again and again to its own output,
on a specimen that reads a two-word language, `be` and `not`, and says after every word whether
the sentence so far affirms or denies: `not not be` affirms, `not not not be` denies. It is prefix
parity, the textbook task that a single pass of attention provably cannot do at growing length,
dressed as Hamlet's negation. One extra rule makes the book: **each token may attend only to
itself and its predecessor** (a window of two). Then information travels at most one word per
pass. That is a light cone, and it gives the book an exact anchor of the same kind as the first
book's entropy floor:

    ℓ* = (ln 2 / (T K)) Σ_{k=1}^{K} max(T − k − 1, 0),     21 ln 2 / 64 = 0.227438919 nats at T = K = 8,

the loss of a model that is exactly right inside the cone and exactly undecided (probability ½)
outside it. The specimen is trained to it, and the reader watches it get there.

**How small.** Two numbers, if the routing is fixed: s_t ← tanh(g(x_t)·s_{t−1}), with one gain
per word (for the decoded sign, keep or flip, this is the transfer matrix of a zero-field Ising
chain; the real-valued map tanh(g s) is an analogy, not the Ising recursion m ↦ tanh(βJ) m). It
trains in 1–2 s in 5 of 5 seeds, reaches the floor to within 0.07–0.18 %, is exact at length 64
with 128 passes, and provably settles to a fixed point by pass T. As a *transformer* (learned
attention over the window, both LayerNorms, a ReLU MLP, embedding and readout): **270 parameters**
(V = 2, D = 4, H = 1, F = 16), the smallest qualifying transformer we tested, trained from
PicoGPT's ordinary initialisation in 6.5–9.9 s of Julia updates, **4 of 5 seeds** reach the mixed-length floor 0.161350952 nats to within 1 % (best seed
0.0033 %), and every one of those four is 100 % right at lengths 32 and 64 with K = T and K = 2T
passes, although it never saw a word beyond length 12 or more than 24 passes. That 4/5 was the selection screen's own rate. **Measured on 30 fresh seeds with the recipe frozen
first** (commit `9d526ef`; unseen initialisation seeds, unseen test words): **25/30 = 83 %**
(95 % Clopper–Pearson 65–94 %), about 10 s per seed. With the fixed predecessor bias removed
(window kept), 20/30 = 67 % (47–83 %), and 27/30 still extrapolate exactly; most failures there are
exact rollouts whose predictions at the arriving front are not confident enough for the 1 % loss
criterion. The attention learns the routing itself (predecessor weight 0.64–0.94 per seed in the
qualifiers); the bias buys confidence, not the computation. (`loop/notes/window.md`, Fresh seeds.)

**Is it worth building.** Yes, with one stated cost. The evidence supports every criterion of the
quest: the loop is load-bearing (with the same two-token window, no one-pass model of any width can beat
the information floor 0.4975 nats, against the looped 0.1614; with global attention the
obstruction is only asymptotic, by Kozachinskiy–Steifer–Wałęga), visible (the cone, a fixed point
of the stream by pass 128, the covector cone going back), specimen-sized (270), anchored (two exact
floors, Lucas' theorem for the companion), and it shows what looping buys (5⅓× the trained length
and passes, with the same 270 parameters). The cost: the plain PicoGPT block, looped, does **not** learn this reliably; besides the loop itself, four
small changes, each a line or two of the model's definition and each answering a measured
failure, are needed (§6); the screen supports the combined recipe, not the necessity of each. Those
failures are not an embarrassment to hide; they are half the book's content.

## 2. The recommended specimen and its evidence

| | value | source |
|---|---|---|
| task | prefix parity of `not`s over {`be`, `not`}, target at every word | `loop/scripts/window_round3.jl` |
| block | PicoGPT block, window-2 attention, V=2, D=4, H=1, F=16, L=1 | `loop/src/window.jl` |
| parameters | 270 = 2VD + V + 2D + 4D² + 2DF + 9D + F | stock `nparams` |
| training | lengths 4–12, passes K ∈ T−1…2T per batch, deep supervision, AdamW (BRIEF §13 constants), 1,250 updates, batch 8 | |
| exact floor (training mix) | (384975233676617/1653814216454400) ln 2 = 0.161350952 nats | enumeration; audited |
| reached | 0.161356316 (seed 3, +0.0033 %); seeds 2, 4, 5: +0.017, +0.25, +0.17 %; seed 1 fails (+20 %); all five replayed by the final reviewer | `window-round3-anchor-boundary-start.json` |
| extrapolation | 100 % at T = 32 and 64, with K = T and K = 2T (4 seeds) | same |
| fixed point | centred RMS step 0.0053 into pass 64, 9·10⁻¹⁶ into pass 128, at length 64 (stationary by 128; the pass-T proof holds only for the 2-gain cell) | same |
| attention | learned predecessor weight 0.967–1.000 (mean 0.989) at pass 128 | same |
| one-pass control | floor (89539/124740) ln 2 = 0.497544536 nats on the same mix, any width *with the same two-token access*; accuracy ceiling ½ + 1/T (53.1 % / 51.6 % at T = 32 / 64) | audited |
| untied control (L = K separate blocks) | same floor as the loop, but K× the block parameters, and nothing beyond its depth: best of five 48-parameter gain stacks 69.26 % at T = 64, pass 24 | `window-round2-controls.json` |
| time | 6.5–9.9 s of optimiser updates in Julia | |
| figure data | θ, curve, floor, a T = 16 trace per pass (stream, attention, probabilities, terminal covector) | `loop/artifacts/window-specimen.json` |

**The companion, rule 60.** The same register run autonomously, s_t ← s_t ⊕ s_{t−1}, draws Pascal's
triangle mod 2. Lucas' theorem predicts every row: s_t^{(k)} = ⊕_j [C(k,j) odd] s_{t−j}. The trained
2-gain cells match it on every pass through 96, four times the trained depth, in all five seeds;
the audit proved the learned gains correct for every word and horizon. Its own exact control
(Astra, confirmed independently by Sol): scored at passes 1 and 2, a one-pass model that cannot
know which horizon is asked has CE infimum (7/16) ln 2 and accuracy ceiling 25/32 at T = 8.

## 3. Runners-up, and what killed the rest

| candidate | fate | why |
|---|---|---|
| prefix parity, full causal attention | **killed as the specimen**, kept as chapter material | at small T one pass reads every bit; training finds shortcuts (582 params fit T = 5 in 1/3 seeds, front stalls at position 4, 58–62 % at T = 20). Nothing forces the loop, so there is no floor |
| rule 60 | **companion** | exact (Lucas), visible (Sierpinski), but autonomous dynamics alone is less compelling than "a sentence understood word by word" |
| pointer chasing (follow a permutation) | dead | one-pass control 95 %; no hop-per-pass seen; hand-built witness needs 16,398 params |
| odometer (successor in binary) | dead | carry = prefix-AND, one pass does it; the loop is a clock, not a computation |
| denoising the line "to be or not to be" | dead | an attractor in all directions is impossible for the residual loop (LN's unit eigenvalue, §5); the loop is not forced |
| relaxation / attention as diffusion | dead as specimen | one pass of global attention can average directly; resurrect as an aside on the spectrum of a frozen attention matrix |
| compass turns (Z₄, S₃) | chapter extension | prefix products in a group; S₃ is solvable, so not the "hard" case (audit correction) |
| Shidoku (4×4 Sudoku) | epilogue | what HRM/TRM made famous; needs bidirectional attention, a dataset, ≫ 1000 params |
| the liar ("this sentence is false") | prologue toy | a 2-cycle; charm only |
| odd–even sort | dead | one pass can rank; and causal output cannot be sorted in place at all |

## 4. What was tried and did not work (the book should show some of it)

- **Stock loop, stock recipe.** Fits the trained length, never extrapolates (round 1).
- **Additive input injection** X_{k+1} = B(X_k) + X₀. The stream grows like k: centred RMS 33 at pass
  1, 3,067 at pass 128 on length 64 (582-parameter ablation, seed 1). Normalising the stream each
  pass holds it at ≈ 2.83.
- **Sinusoidal positions.** The learned rule depends on absolute position; positions > 8 unseen.
  Without any position signal there is an exact ceiling of 83.5 % at T = 64 (the first word,
  attending only to itself, looks like a left-padded copy); with a marker on the first word, solved.
- **Seed reliability.** 1 of 5 for the plain window block; a table of 20 one-line changes and their
  seed counts is in `loop/notes/window.md` (Round 3). BOS token, curricula, near-identity
  initialisation: no help. What helped: the fresh input in the first residual, a fixed predecessor
  bias of +4 on the attention logits, and starting the stream from the first word only.
- **Hand-built weights** (Astra) exist for prefix parity in a stock block (4 features, finite
  table; or a 446-parameter previous-token head for 64 bits needing scores ×1024), for pointer
  doubling (27,980 params) and chasing (16,398). They prove existence, not trainability; the
  hand-built reference does not survive fine-tuning.
- **The gradient light cone in Float64**, in the 2-gain cell, is exact on a 16-word trace but the
  pre-tanh covector underflows on long runs (gone by pass 13 on 64 neutral words): a
  vanishing-gradient exhibit, not a bug. It is not a bound on the transformer's training gradient.

## 5. Chapter arc (one-sentence claims, in the manner of BRIEF §8)

0. **Prologue.** One block, applied again and again: a sentence of `be`s and `not`s is understood
   one word further at every pass, and the loss falls to a floor set by how far the news has
   travelled.
1. **Negation.** Prefix parity is the task in which every word matters to every later answer;
   a single pass of attention cannot do it at growing length (average sensitivity, Kozachinskiy–
   Steifer–Wałęga 2026, asymptotic only), and with a window the one-pass floor 0.4975 nats is exact.
2. **Two numbers that loop.** s_t = tanh(g(x_t) s_{t−1}) carries the sign of an Ising chain's
   transfer matrix (keep or flip); two gains learn "keep" and "flip" in a second, and the state reaches a fixed point at pass T.
3. **The light cone.** With a window of two, pass k sees k + 1 words; outside the cone the
   posterior is exactly ½, which gives the exact floor ℓ* and the shape the reader watches fill in.
4. **The block in the loop.** The PicoGPT block replaces the two gains; four changes (window,
   fresh input, normalised stream, a first-word start) each answer a failure we measure.
5. **Pulling back through a loop.** θ̄ is a sum over passes of the same transposed map; the covector
   travels back one word per pass; in the plain residual loop LayerNorm's symmetry pins an eigenvalue
   at 1, so the fixed-point formula (1 − Jᵀ)⁻¹ needs the normalised loop or a quotient.
6. **Learning to loop.** Deep supervision, random extra passes and mixed lengths make the right
   state a fixed point; seeds stall, converge to wrong fixed points or drift, and we show which.
7. **Autonomy: rule 60.** Without the input, the same machinery is a cellular automaton; Lucas'
   theorem predicts every row of the Sierpinski triangle, and a one-pass model asked two horizons
   has floor (7/16) ln 2.
8. **Epilogue: thinking longer.** What the large looped models do (Universal Transformer, Huginn,
   Ouro, HRM/TRM, the GPT-6 "Astra" reports), halting, log-depth pointer doubling (exists, 27,980
   parameters), and what we left out.

## 6. The minimal change to PicoGPT itself

The working specimen differs from a stock one-block PicoGPT in five places (from
`loop/src/window.jl`; Sol's own count of the forward change is the 9-line listing in
`loop/notes/window.md`, Round 3):

1. **Loop**: `for k in 1:K` over `p.blocks[1]` instead of `for ℓ in 1:L` (1 line), readout and loss
   at every pass (2 lines).
2. **Window mask**: `M[i,j] = −∞` unless `j ∈ {i−1, i}`, and a fixed +4 on the predecessor score
   (2 lines in `attention`).
3. **Fresh input in the first residual**: `X₁ = X₀ + Wo·O + bo` instead of `X + …` (1 line in `block`).
4. **Normalised stream**: `X = LN(X₁ + Y₂)` with unit gain, zero offset, no parameters (1 line).
5. **Start and boundary**: the stream starts from the first word's embedding only, with a fixed
   (+1, −1, …) marker on it; no positional code (2–3 lines in `embed`).

Stated as a change to the model's *definition*, this is the nine-line forward sketch in
`loop/notes/window.md` (Round 3), plus the window in the mask. As *implementation* it is larger,
because every forward line needs its pullback (final review, `loop/notes/review-concept.md` §5):
the fused two-key attention and mixing takes 29 lines forward and 26 back (stock attention + mix:
21 + 15), and the loop's reversal 36 lines with its options (stock: 10). It adds each pass's
readout adjoint, pulls back through the fixed LayerNorm, sums the shared block, readout and
fresh-input cotangents into one θ̄, and adds the initial-state cotangent at the first word only.
So: **about ten lines of mathematics, about a hundred lines of Julia with pullbacks**, zero new
parameters. A JavaScript twin repeats that diff in `web/src/specimen` (not yet written or timed).

## 7. Three names for the new objects

| maths | physics | ML |
|---|---|---|
| a map iterated on its own output, X_{k+1} = G(X_k) | a discrete-time dynamical system | "looped / weight-tied / recurrent-depth transformer" |
| the pass index k | time | "loop", "recurrence", "iteration" |
| re-adding X₀ every pass | a source term, a forcing | "input injection" |
| the support {t−k, …, t} of the dependence after k passes | a light cone; a Lieb–Robinson bound | "receptive field" |
| a mask with band width 2 | nearest-neighbour coupling | "sliding-window attention" |
| s_t = a_t s_{t−1}, a_t = ±1 (the decoded sign) | the transfer matrix of a zero-field Ising chain, on signs | "state tracking" |
| X* = G(X*) | a steady state, an equilibrium | "fixed point", "DEQ" |
| a loss at every pass | — | "deep supervision" |
| Σ_k of the same transposed map, in reverse time | the adjoint equation in time | "backpropagation through time" |
| (1 − Jᵀ)⁻¹ = Σ_h (Jᵀ)^h when ρ(J) < 1 | a resolvent, a Green's function | "implicit differentiation" |
| running more passes than trained | extrapolating in time | "test-time compute", "thinking longer" |
| deciding K from the state | a stopping time | "halting", "adaptive computation" |

## 8. The figures that would carry the book

To `DESIGN.md` (two inks, every mark a number). Figures 3 and 6 can be drawn from saved data; 1, 2, 4
and 5 need new exports (five 64 × 128 probability grids, exhaustive curves for all seeds, gain
trajectories, decoded impulse cells), which are cheap reruns of existing scripts.

1. **The cone** (the book's central plate). Position (1…64, left to right) × pass (1…128, top to
   bottom), one cell per (t, k): a value-ink square whose area is |p − ½|·2 for the predicted
   probability of the true label, filled when correct, open when wrong. The filled triangle below the
   diagonal t = k + 1 and the empty paper above it are the light cone; the diagonal is labelled
   directly. Small multiples: seeds 2–5 succeed, seed 1's front lags (site 44 at pass 128).
2. **Falling to the floor.** The exhaustive loss of all five seeds against update number, with two
   labelled hairlines: the one-pass floor 0.4975 and the looped floor 0.16135; the samples in the
   margin are sentences read at four checkpoints.
3. **The covector cone.** The same grid in covector ink for the terminal logit difference at word
   16: the cotangent per pass, nonzero only inside the backward cone, its left edge moving one word per
   pass. In the two-gain cell its norm falls 1, 10⁻⁶, 4·10⁻¹³ over the first three words; the
   underflow is drawn honestly as the end of ink.
4. **Two gains.** The plane (g_be, g_not) with the five seeds' training trajectories, from random
   starts into the quadrant g_be > 0 > g_not; the loss floor as a contour. Every mark a parameter.
5. **Sierpinski.** Rule 60 from a single impulse, 97 positions × 96 passes, cells from the trained
   cell, Lucas' prediction as hairline outlines: a trace in the sense of BRIEF §11.
6. **Drift.** Centred stream norm per pass, additive injection (33 → 3,067) against the normalised
   loop (≤ 2.8): two strokes on a log axis, a trace.
7. **The ladder.** From 2 parameters to 270, the seed counts of §4 as rows of five dots, filled for a
   seed that reaches floor and extrapolation: what each line of the diff buys.

## 9. Title candidates

*Not Not To Be* · *Pass by Pass* · *One Block, Many Passes* · *The Light Cone of a Transformer* ·
*Again* · *Thinking Longer*

My choice: **Not Not To Be**, subtitle *the tiniest looped transformer, pass by pass*.

## 10. Risks and open questions

- **Four changes to the block.** The specimen is a looped *transformer*, but not the stock block
  looped. A purist will object; the book's answer is chapter 4 (every change is forced by a measured
  failure). Whether a cleaner set of changes exists is open.
- **Reliability (measured).** On 30 fresh seeds the frozen recipe succeeds in 25 (83 %, interval
  65–94 %); about one live run in six fails. Of the five failures at +4: two recover errors by pass
  128 but miss at pass 64, one settles on a wrong fixed point near the boundary, one keeps drifting
  and loses its front, one is exact but under-confident at the front. In the browser a failed seed
  is either an exhibit (chapter 6) or a reason to ship a chosen seed for the live run. We did not
  find a 5/5 transformer form.
- **The fixed predecessor bias (+4)** nudges the head towards the predecessor and must be explained as
  a prior. Without it (fresh seeds) the rate falls from 83 % to 67 %, but the routing is still
  learned and 27/30 extrapolate exactly; the bias could be dropped at the price of reliability
  (+2 gives 70 %).
- **The window removes the question of what attention attends to.** The more interesting story of
  full causal attention (does a looped model learn the linear front or log-depth doubling?) remains
  untrained at this size; doubling exists (27,980 hand-built parameters).
- **Training speed in JavaScript.** 1,250 updates of a 270-parameter model unrolled to ≤ 24 passes on
  batches of 8 words: probably seconds in the browser, not measured.
- **Gradient underflow** over long rollouts (audit): fine for training (≤ 24 passes), must be shown
  honestly in figure 3.
- **The tie to the first book.** The text is a two-word language, not Shakespeare; the Hamlet line is
  a framing. Whether that is charm enough is the user's judgement.

## 11. What building the real book would take

The Julia side is largely done: `loop/src/window.jl` (501 lines including variants; the specimen
path is perhaps 150), tests with gradchecks, the exact floors audited. Remaining work, by role:

- **Code (Sol):** fold the specimen path into a clean `loop/src/` of book quality (≈ 2 days of agent
  time), a JS twin conformant to 10⁻¹² in `web/src/specimen` style (≈ 1–2 days), an export script.
- **Mathematics (Astra):** a `derivations.md` for the book's claims in chapter order; audit (≈ 1 day).
- **Prose (Gemini Neon / Muse):** 9 chapters; the first book's chapters total about 29k words
  including markup, so a budget of 20–30k words, with reviews (≈ 3–4 days of agent time).
- **Figures (Opus):** 7 figures and ~8 traces to `DESIGN.md`, an `edition.js` as for `jspace/`
  (≈ 4–5 days).
- **Print:** the edition pipeline already exists (`BOOK_EDITION`, `--edition`).

As a target, not a measurement: about two weeks of orchestrated agent work, the size of the J-space supplement or a little larger,
quota permitting; the main risk is the transformer form's measured 83 % fresh-seed reliability (67 % without the
predecessor bias), which a further round could try to raise.

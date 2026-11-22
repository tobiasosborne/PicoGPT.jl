# Not not to be — concept for a third book, on the tiniest looped transformer

Quest *loop*, 21 November 2026. Everything below rests on runs in `loop/` (scripts, artifacts,
notes) and on derivations audited by a second agent (`loop/notes/audit.md`). Lab notebooks:
`loop/notes/prototypes.md` (full-attention prototypes), `loop/notes/window.md` (the light-cone
specimen), `loop/notes/derivations.md` (mathematics), `loop/notes/literature.md` (survey).

**Revision of 22 November 2026.** After comparison with a real looped model, Inkling 2 Mini
(`loop/notes/inkling.md`), five changes are approved and carried through §§1, 5, 6, 7 and 12,
with the runs they need in §13.
The analysis in `loop/notes/written-vs-looped.md` was independently checked in [loop/notes/check-wvl.md](notes/check-wvl.md), and its corrections are applied here.
The five changes: (1) add the written-versus-looped
comparison, the negation task done three ways; (2) present the window of two as a device that turns
dependent steps into distance; (3) a gate that starts closed as the stabiliser (a proposed design
with proved identities, not yet a trained result); (4) shrink and relabel the fixed-point material
as the Huginn / small-recursive-model family, an aside rather than the spine; (5) rebuild the
Shakespeare chapter on Inkling's layout — Book 1's four blocks, the first and last run once, the
middle two three times, with a gated message. Earlier results are kept throughout; where this
revision supersedes a claim or a plan it says so on the spot ("superseded by the revision of
22 Nov: …") rather than deleting it. No number below is new to this document: every added number
comes from the analysis or the Inkling source note.

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

**Steps versus distance (added 22 Nov).** The window of two is now presented, first of all, as a
device that turns dependent steps into distance. The line of words is a chain of maps on a single
bit, f_t(y) = y ⊕ x_t; the window puts each next map one edge away, so the number of dependent
transfers still missing is a visible distance on the page, and one pass can close it by one word. In
width-w causal attention one layer reads at most w − 1 columns to the left, and a stack of L such
layers iterated K times reads back at most KL(w − 1) columns. That is a bound on reach, not a law of
one hop per pass: with global routing, pairwise composition of stored summaries can double its span
each round (at most 2^r edges after r rounds, pointer doubling meeting the bound), and the notes
keep the serial-oracle, full-attention and doubling cases apart. The window is what forbids those
short cuts here, on purpose, so that "how many dependent steps remain" and "how far away the news
is" are the same number.

**The negation task three ways (added 22 Nov).** Score only the final parity y_T of a sentence of
T ≥ 2 independent fair words, the answer read from the logits at the last word. Then the same
two-word block can be used three ways, and the comparison — not the loop alone — is the book's
subject:

| schedule | what the final decision can see | population end-only CE infimum | dependent block calls | token columns evaluated | intermediate tokens emitted |
|---|---|---|---|---|---|
| one pass, blank scratchpad | at most the last two words | ln 2 if T > 2, else 0 | 1 | T | 0 |
| one pass per decision, written feedback | prior parity (x₁ at T = 2) and next word | 0 (infimum) | T − 1 | 2(T − 1) | T − 2 |
| K tied passes, blank scratchpad | at most the last K + 1 words | ln 2 if T > K + 1, else 0 | K | KT | 0 |

To reach a zero infimum the block can either write T − 2 intermediate one-bit answers or make
T − 2 additional passes over the sentence; both schedules have T − 1 dependent calls, so in this
protocol one intermediate answer trades for one extra whole-sentence pass. Writing is cheaper in
evaluated token columns for T > 2; looping adds only the final answer to the transcript. At T = 8
both exact schedules have seven dependent calls: writing evaluates 14 token columns and emits six
scratch tokens, the dense loop evaluates 56 and emits none. (For T = 1 one call suffices for all
three; for T = 2 the minimum-depth schedules coincide. Counts exclude embedding, decoding, padding
and I/O, describe straightforward execution, and do not by themselves predict latency. When every
prefix answer is wanted in the output anyway, writing it incurs no extra intermediate-output
charge.) The written protocol is streaming: the caller supplies the words in order and feeds each
written parity back with the next word as a fresh two-word context — `not not → be`, `be be → be`,
`be not → not` for `not not be not`. An appended scratchpad after a prompt already supplied, under
the strict two-token mask, is a different protocol with different information. Scoring every prefix
position instead recovers the original floors 0.497545 and 0.161351; those and the end-only rows
measure different objectives, and the book will say which it means each time.

A wrong intermediate answer has the same algebra in the written chain and, under an independent-flip
noise model, along a predecessor-only loop's first arriving diagonal from an exact seed: with per-step error rates ε_i relative to the
correct update of the state actually received, Pr(final error) = ½(1 − Π_i (1 − 2ε_i)), for equal
rates (1 − (1 − 2ε)^t)/2 — two flips cancel, and in the zero-field Ising reading the factor
1 − 2ε_i is the mean error sign. For fixed words, the physical edge correlation is this factor
times the intended copy-or-flip sign, which is preferred only below a 50 % flip rate. At 50 % the
signs tie, and above it the opposite sign is preferred. A teacher-forced error rate
alone supplies neither the independence nor the error law on mispredicted prefixes, and a learned
soft-attention block can share errors across passes; applying the formula to a trained model needs
measurement (§13, run 1).

The three rows are achievable with the specimen's own 270-parameter block: `loop/notes/written-vs-looped.md`
gives explicit hand-set weights (a witness family in a scale M) approaching the cone's infimum and
decoding every written answer, checked on every sentence of lengths 1–8 — at M = 100 the smallest
signed margin inside the cone was 7.03532 and the largest absolute gap outside it 0.099995; at
M = 10,000, 70.70679 and 0.00999995. These are hand-set arrays, as in §4; the trained specimen's
fresh-seed rate below is a separate, measured result.

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

**The gate that starts closed (added 22 Nov; a design, not yet a trained result).** Inkling's later
passes do not carry a stream forward untouched: each pass rebuilds its block sums with the
embeddings and the first block again among its sources, and the previous pass enters only as a
*message* — for each input (Q, K, V, MoE) a learned mixture of the previous pass's two middle-block
sums, mixed in through a gate x′ = κ⊙x + γ⊙m with κ = 1 − tanh(Fz), γ = tanh(Gz) and z a
64-coordinate bottleneck of the RMS-normalised input and message (Goodfire's appendix; the
announcement and cards give no formula). The announcement says the gate at first blocks the message
entirely and opens during training; no source states the initialisation, and zero output matrices
F, G are a sufficient way to realise the identity mix, not a recovered checkpoint. The specimen's
version keeps only the restart and the adjustable message path: with U the marked embeddings,
S the first-word-only initial stream and A_p the existing anchored window block (with or without the
final stream LayerNorm — the choice to be fixed across comparisons), add one learned scalar g,
initialised to zero and shared across positions and passes, and set X₀ = S,
m_k = X_{k−1} − S, Y_k = S + g·m_k, X_k = A_p(Y_k, U). The subtraction makes g = 1 algebraically the
original recurrence; at g = 0 every call receives S and returns A_p(S, U), so the final logits,
loss and backbone gradient equal those of a single call (and likewise for a mean of identical pass
losses with identical targets); K = 1 is independent of g. Its pullback is in §6, item 6. At
closure the state Jacobian g·A_Y vanishes, and if A is L-Lipschitz on the relevant inputs then
|g|L < 1 gives a contraction there (a fixed-point conclusion on a restricted domain needs an
invariant complete domain as well; smallness at one point is local control only, and a small
observed gain in Inkling does not by itself imply a contractive Jacobian). Only exact closure and
this local gain bound currently justify calling the gate a stabiliser; whether the learned scalar
can replace the final stream normalisation without losing training reliability is conjectured and
is §13, run 2. Two cautions from the analysis belong in the prose, not a footnote: the opening
signal ∂_gℒ at closure can vanish — with a calibrated uniform terminal readout and T > 3 the
population gate gradient of end-only CE is zero on fair words, so short lengths or prefix
supervision may be what opens the gate — and one checked three-word example (prefix CE at the final
pass) shows the signal present where expected: gain derivative 1.81722078 at closure and 0.00234017
at g = 0.2, with centred-difference checks agreeing to 1.29·10⁻⁷ on the gain and 3.81·10⁻¹¹ along
one backbone direction.

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

Superseded by the revision of 22 Nov: the fixed point as one of the book's central, visible anchors,
and with it the weight the paragraph above puts on settling. What was proved or measured stands —
the 2-gain cell settles by pass T, and the trained transformer's centred step falls to 9·10⁻¹⁶ into
pass 128 at length 64 (§2) — but neither a stable decoded sign nor a small observed state step
establishes invertibility of I − J_* or ρ(J_*) < 1, and Inkling, the real model now in view, is a
fixed-depth three-pass model decoded once, whose stated stability device is the initially closed
cross-loop channel, not an equilibrium. In the revised book the main account teaches finite
unrolling and shared-parameter pullbacks; the equilibrium material shrinks to a short aside on the
family that does use it or approach it — Huginn (varied unroll, truncated backpropagation), HRM/TRM
(outer deep supervision; TRM removing HRM's one-step equilibrium approximation) and, as the genuine
equilibrium case, DEQs, where the implicit adjoint solves (I − J_*ᵀ)λ = ∇_Xℓ. The fixed-point
formula and its hypotheses stay, in that aside, with the derivation's counterexamples. Also
superseded by the revision of 22 Nov: a Shakespeare chapter that merely tests whether the toy's
recipe transfers (the runs in §12, kept as evidence). The closing chapter is rebuilt on Inkling's
layout (§12, sandwich): Book 1's four blocks with the first and last run once, the middle two three
times under shared weights, and a gated message restarted from the first block — 810,177 stored
parameters (Book 1's 810,049 plus a shared 128-channel gate) for eight block applications per token,
exactly Book 1's predictor when the gate is closed. That design is proved accounting and a proved
pullback; its loss is not yet measured (§13, run 4).

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

Revised 22 Nov. The arc below is the current plan; the 21 Nov arc follows it, kept and marked
superseded where the two differ.

0. **Prologue.** One block, applied again and again — or called on each fresh pair with its own last
   answer written back in: a sentence of `be`s and `not`s can be understood one dependent step further
   at every call, and whether the intermediate answer is written down or carried in the stream is
   the book's first fork.
1. **Negation.** Prefix parity is the task in which every word matters to every later answer;
   a fixed-size one-layer transformer cannot do it at growing length (average sensitivity, Kozachinskiy–
   Steifer–Wałęga 2026, asymptotic only), and with a window the one-pass floor rounds to 0.4975 nats on the training mixture.
2. **Writing it down.** Scored only at the last word, the same block can approach zero cross-entropy either
   by writing T − 2 intermediate one-bit answers (chain of thought, at specimen size) or by making
   T − 2 extra passes — T − 1 dependent calls both ways for T ≥ 2. Writing evaluates fewer token
   columns for T > 2. Looping writes only the final answer. With independent flips on the written
   path or a predecessor-only diagonal from an exact seed, the error probability is ½(1 − Π(1 − 2ε_i)).
3. **Two numbers that loop.** s_t = tanh(g(x_t) s_{t−1}) carries the sign of an Ising chain's
   transfer matrix (keep or flip); two gains learn "keep" and "flip" in 1–2 seconds, and the state
   reaches a fixed point at pass T.
4. **Steps versus distance: the light cone.** The window of two puts each next dependent map one
   edge away, turning the count of missing dependent steps into a visible distance: pass k can see
   at most k + 1 words. Outside the cone the posterior is exactly ½, which gives the exact floor ℓ* — and
   with global routing the same task admits doubling (span up to 2^r after r rounds), which is
   exactly what the window is there to forbid.
5. **The block in the loop, and the gate that starts closed.** The PicoGPT block replaces the two
   gains; the four measured changes of 21 Nov stand (window, fresh input, normalised stream,
   first-word start), and the revision adds a fifth, proved but untrained: restart each pass from
   the first-word stream and admit the previous pass only as a message scaled by a learned scalar
   g initialised to zero, so that at closure the loop is exactly one call, at g = 1 algebraically the
   21 Nov recurrence, and whether g replaces the stream normalisation is the chapter's open
   measurement (§13, run 2).
6. **Pulling back through a loop.** θ̄ sums the parameter cotangents from each call of the shared block, and the
   covector can travel back at most one word per pass; the gate contributes its accumulations (ḡ from the message,
   g·V_k to the previous state, (1 − g)V_k to the restart); finite unrolling with shared
   parameters is the method of the book's own models, and the equilibrium formula (1 − Jᵀ)⁻¹,
   with the LayerNorm symmetry that pins an eigenvalue at 1 in the plain residual loop, is moved
   to the aside in chapter 9.
7. **Learning to loop.** Deep supervision, random extra passes and mixed lengths reward the right
   readout at later passes. Seeds stall, settle on wrong answers or drift, and we show which —
   and, for the gate, whether short lengths or prefix supervision are what gives it an opening
   signal when the end-only population gradient at closure is zero.
8. **Autonomy: rule 60.** Without the input, the same machinery is a cellular automaton; Lucas'
   theorem predicts every row of the Sierpinski triangle, and a whole-word observer with one horizon-blind readout
   has floor (7/16) ln 2 on fair length-eight words at equally weighted passes 1 and 2.
9. **Aside: the fixed-point family.** Huginn's varied unroll with truncated backpropagation,
   HRM/TRM's outer deep supervision (TRM without HRM's one-step equilibrium approximation) and the
   DEQ's genuine equilibrium solve with implicit adjoint are one family of answers to "how deep,
   and how differentiated" — distinct methods, grouped briefly, none of them the specimen's or
   the sandwich's.
10. **Shakespeare in Inkling's layout.** Book 1's four blocks with the first and last run once and
    the middle two three times, restarted each pass from the first block, with a per-channel
    gate on the previous core output that starts closed — 810,177 stored parameters, eight block applications per token,
    exactly Book 1 when the gate is shut — set against the 21 Nov Shakespeare runs (§12) and
    the variable-loop variant's chained lookups improving to ten passes, written reasoning
    shortening on the GPQA subset correct at all five pass counts, and measured closed-book
    recall nearly flat.
11. **Epilogue: thinking longer.** What the large looped models do (Universal Transformer, Huginn,
    Ouro, HRM/TRM, Inkling 2 Mini and its variable-loop variant, the GPT-6 "Astra" reports),
    halting, log-depth pointer doubling (27,980 parameters for the four-bit construction, including a preparation pass), unwritten reasoning and what
    it costs monitoring, and what we left out.

The 21 Nov arc, kept as written. Superseded by the revision of 22 Nov: its chapters 2–3 (the
window introduced as a light cone only, without steps-versus-distance or the written alternative),
its chapters 5–6 (the fixed point and (1 − Jᵀ)⁻¹ as main-line material; now revised chapter 6 and
the aside, revised chapter 9), its chapter 4 (four changes, no gate; now revised chapter 5), and
its epilogue's treatment of Shakespeare as a survey item (now revised chapter 10). Claims repeated
unchanged above are not superseded.

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

Added by the revision of 22 Nov (proved design; untrained, so not yet part of the measured
specimen above):

6. **Restart and gated message**: each pass starts again from the first-word stream S and admits
   the previous pass only as the message m_k = X_{k−1} − S, scaled by one learned scalar g
   initialised to zero and shared across positions and passes: Y_k = S + g·m_k,
   X_k = A_p(Y_k, U), where U is the marked embedding and A_p the block of items 1–5 with the
   final-stream-normalisation choice fixed across comparisons. At g = 0 the loop is exactly a
   single call — same logits, loss and backbone gradient, for every input, horizon and parameter
   vector — and at g = 1 algebraically the corresponding ungated recurrence; merely gating a correction on an autonomously carried
   residual would leave the old residual path open and fail that identity. The pullback, seeding
   each supervised pass's readout cotangent at its own X_k and writing V_k for the block VJP's
   Ȳ_k, is ḡ += ⟨V_k, X_{k−1} − S⟩_F, X̄_{k−1} += g·V_k, S̄ += (1 − g)V_k (with X̄₀ added to S̄
   and S̄ sent through the initial-stream map to U, adding to the direct anchor cotangents from
   every call before the embedding scatter; a per-channel
   gain g_i replaces the Frobenius product by the sum over token columns for its row; a gain
   written as tanh b multiplies its cotangent by 1 − tanh² b, while the direct scalar reaches
   g = 1 exactly). One new parameter. What it buys is, so far, exactly this: closure recovers the one-pass predictor,
   the state Jacobian g·A_Y vanishes there, and |g|L < 1 is a contraction where A is L-Lipschitz;
   whether it can stand in for item 4 in training is §13, run 2, crossing gate enabled/bypassed
   (bypassed = fixed g = 1) with the final stream LayerNorm present/absent. In Inkling the same
   idea is per-input and input-dependent (κ = 1 − tanh(Fz), γ = tanh(Gz) on a 64-coordinate
   bottleneck z, a separate message for each Q, K, V and MoE input mixed from both previous
   middle-block sums); the specimen keeps one shared scalar and one message. For the Goodfire form
   at F = G = 0 the cotangents F̄ and Ḡ are generally nonzero through z while z̄ vanishes on this
   route; setting the bottleneck projection to zero as well would remove the opening signal.
7. **The written schedule is not a sixth model change**: the same block, called by a streaming
   caller on fresh two-word contexts (previous written parity, next word), with the ordinary
   first-column initialisation each time. The departure from stock PicoGPT is in the caller and in
   what is scored (the final word only, against the all-prefix training above), not in the block;
   teacher forcing feeds the gold parity where free decoding feeds the model's own argmax, and the
   hand-set witness of §1 decodes correctly under the latter on every sentence of lengths 1–8.

For Shakespeare the departures are different and are stated with the design in §12 (sandwich):
no window, no first-word start, no per-pass readout — Book 1's blocks and semantics kept, the
middle two tied three times, the restart U = B₁(E + P) and a shared per-channel gate in place of
items 2–6. The two lists share only the loop, the restart and the gate.

Stated as a change to the model's *definition*, this is the nine-line forward sketch in
`loop/notes/window.md` (Round 3), plus the window in the mask. As *implementation* it is larger,
because every forward line needs its pullback (final review, `loop/notes/review-concept.md` §5):
the fused two-key attention and mixing takes 29 lines forward and 26 back (stock attention + mix:
21 + 15), and the loop's reversal 36 lines with its options (stock: 10). It adds each pass's
readout adjoint, pulls back through the fixed LayerNorm, sums the shared block, readout and
fresh-input cotangents into one θ̄, and adds the initial-state cotangent at the first word only.
So: **about ten lines of mathematics, about a hundred lines of Julia with pullbacks**, zero new
parameters. A JavaScript twin repeats that diff in `web/src/specimen` (not yet written or timed).
Superseded by the revision of 22 Nov: "zero new parameters" — the gated specimen of item 6 has one
(the scalar g), and the Shakespeare sandwich of §12 has 128 (one
per channel, shared by both restarts). The line counts were the final review's count of items 1–5;
the gate's forward is the restart line and the message line above and its pullback the three
accumulations, but no revised implementation count exists yet and none is claimed here.

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
| feeding a predicted token back as a later input, so the output becomes state | writing on a tape; an external memory | "writing it down", "scratchpad", "chain of thought" (written feedback) |
| what pass k − 1 hands to pass k, other than the standing input: m_k = X_{k−1} − S (specimen); a learned mix of the previous pass's block sums (Inkling) | a signal carried between steps, distinct from the standing source | "the message" (cross-loop message) |
| a learned multiplier on the message, zero at initialisation: Y_k = S + g·m_k; in Inkling κ⊙x + γ⊙m with κ = 1 − tanh(Fz), γ = tanh(Gz) | a valve; a coupling switched on from zero | "the gate", a gate that starts closed |
| stored blocks applied once around a tied middle: B₁ and B₄ once, B₃∘B₂ three times (Book 1's four; Inkling's first and last six-layer blocks once, middle twelve layers three times) | boundary layers; a fixed enclosure around a repeated interior | "the blocks that run once" — prelude and coda (Inkling's prelude; the sandwich's first and last blocks) |
| the count of dependent transfers still missing, read off as positions not yet reached: at most one word per pass in the window, span up to 2^r in r doubling rounds | a causal horizon measured in lattice spacings | "steps versus distance" — serial depth made visible as reach |

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

## 12. Does it generalise to Shakespeare? (trial, 21 Nov 2026)

One run per model, Book 1's data, evaluation windows and 5,000-step recipe, D=128, H=4, F=512, T=256,
full causal attention, sinusoidal positions, no predecessor bias; **215,233 parameters** each
(Book 1's untied four blocks: 810,049, held-out 1.5313). Details: `loop/notes/shakespeare.md`.

| model | held-out loss at its best pass | at pass 8 | at pass 16 |
|---|---|---|---|
| one block, one pass | 1.698 (pass 1) | 5.973 | 6.899 |
| stock tied, `blocks[1]` ×4, loss at pass 4 | **1.654** (pass 4) | 3.600 | 5.312 |
| specimen recipe (fresh input, normalised stream, loss every pass, K ∈ {3,4,5}, first-word start) | 1.784 (pass 4) | 1.848 | 1.891 |

- **What carries over unchanged:** the code path (one block, tied, hand-written pullback, gradchecked
  in Float64), and the specimen's stabilisers do what they did on the toy: the looped state
  contracts (centred step 103 at pass 1, 19.7 at pass 4, 0.83 at pass 16 on the "Good morrow,
  neighbour" passage; 78 of 83 predictions final by pass 8), and extra passes cost little (1.78 → 1.89
  over 12 extra passes). The stock tied loop, by contrast, is tuned to exactly four passes and falls
  apart beyond them (1.65 → 5.31): "overthinking", seen cleanly.
- **What does not:** the light cone, the exact floor and the extrapolation story have no counterpart
  with full attention on text. More passes at test time do **not** help: every model is best at its
  trained pass count. And the stabilisers cost loss at scale (1.784 against stock tied's 1.654;
  even the one-pass control, 1.698, beats it). Tying itself costs against Book 1 (1.654 vs 1.531) and
  buys only 0.044 over one block at equal parameters, for 4× the compute.
- **What a Shakespeare chapter could honestly claim:** one shared block of 215k parameters writes
  Shakespeare-like text; weight-tying trades parameters for compute at a measurable price; the loop's
  behaviour beyond its trained depth depends on how it is built (a contracting, settling loop versus a
  stock one that is tuned to one depth). **What it could not claim:** that looping improves the model,
  that thinking longer helps, or anything resembling the specimen's exact anchors.
- **Confounds and open questions:** one seed each; the looped run also carried over the toy's
  first-word-only start, which may explain part of its pass-1 weakness (2.49) and its higher loss; the
  separate effect of each change is untested; the window-of-two Shakespeare model (to set per-pass loss
  against (k+1)-gram statistics) was skipped. Wall times were measured under the laptop's Power Saver
  profile (one pass 24.9 min, looped 83.1, stock tied 75.1) and do not compare with Book 1's 36.2.

### Round 2: the confound removed, and the window of two on text

Two more runs, one seed each, same recipe (`loop/notes/shakespeare.md`, Round 2).

| model (215,233 params) | best held-out loss | pass 16 |
|---|---|---|
| A: specimen recipe, stream started from the full input, K ∈ 3…5 | 1.710 (pass 3) | 1.782 |
| B: as A with the **window of two**, K ∈ 1…8 | **1.626** (pass 6) | 1.639 |
| (round 1) stock tied ×4 / one pass / specimen recipe with first-word start | 1.654 / 1.698 / 1.784 | |
| interpolated Kneser–Ney n-gram of order k+2 (k+1 characters of context, as pass k sees) | 1.530 at order 7 | 1.540 at order 18 |

- **What the stabilisers cost, confound removed:** the first-word start was most of round 1's
  deficit (full-input start improves pass 4 by 0.070 nats). What remains is small: 1.710 against
  one pass's 1.698 (+0.012) and stock tied's 1.654 (+0.056), in exchange for a loop that does not
  collapse beyond its trained depth (1.78 at pass 16 against stock tied's 5.31).
- **The window of two is the best of the small models** (1.626 at pass 6, 0.029 better than stock
  tied with full attention, 0.094 above Book 1's 810k-parameter 1.531), and its **cone is real in
  the trained logits**: all 96 perturbations outside t−k…t left the pass-k logits exactly unchanged,
  all 48 at the cone's inner edge changed them. (The window and the longer unroll changed together
  from A to B, so their separate contributions are open.)
- **Against the n-gram ladder it lags, pass by pass, but follows its shape.** The gap is 0.019 nats
  at pass 1, 0.065 at pass 2, then a nearly constant 0.09–0.10 from pass 3 to 16; both curves drop
  steeply while the first five or six characters enter the context and are flat after pass 5–6. The
  ladder is a fitted reference line (discounts tuned on the last 5 % of the *training* split), not a
  floor; the transformer also sees absolute positions.
- **A sobering by-product:** the smoothed 7-gram (1.530) is as good as Book 1's four-block model at its
  5,000 steps (1.531). Book 1's Chapter 9 should know this.

**Reading for the book (revised).** Shakespeare gives a good closing chapter, not a second specimen.
With the window of two, the toy's central object survives on text exactly: pass k of the trained
language model provably and measurably sees k + 1 characters, so each pass can be set beside an
n-gram of matching order, and the reader watches the loop climb the n-gram ladder a rung per pass,
about 0.1 nats short of it, levelling where the ladder levels. What it cannot claim: that looping
beats untied depth (1.626 vs 1.531), that it beats a well-smoothed n-gram, or an exact floor. One
seed per model throughout.

### Revision of 22 Nov: the sandwich — Book 1's four blocks in Inkling's layout

Superseded by the revision of 22 Nov: the reading above as the *plan* for the Shakespeare chapter.
The 21 Nov runs in this section are kept, every number, as the evidence they are — one seed each,
215,233-parameter single-block models against Book 1's 810,049 — but the chapter is rebuilt around
the design below; the window-of-two model B and its n-gram ladder remain exhibits, not the closing
argument. What the revised chapter could claim is stated after the design.

**The design (proved accounting; untrained).** Keep Book 1's dense pre-LN blocks and exact
semantics — D = 128, H = 4, F = 512, T = 256, V = 65, full causal attention, fixed sinusoidal
positions — and arrange its four stored blocks as Inkling arranges its four six-layer blocks: the
first and last run once, the middle two three times with shared weights. Inkling's released model
is 24 stored layers in four blocks of six, applied 48 times per token (first and last blocks once,
middle 12 layers three times), decoded once after the last block; looping the middle rather than
the last 12 won on 27 of 28 of their evaluations, motivated by the authors' intuition that the outer layers
translate to and from the embeddings while the middle does more abstract work. Here, with B_i
Book 1's blocks as stored, E the token embedding and P the sinusoidal code, and one trainable gain
per channel shared by both restarts:

    U = B₁(E[:,x] + P),   M₀ = 0,   Y_k = U + g⊙M_{k−1},   M_k = B₃(B₂(Y_k)),   k = 1, 2, 3,
    Z = W_out·LN_f(B₄(M₃)) + b_out        (read out once, scored on Book 1's next characters).

U — the first block's output — is the standing restart, as the embeddings and prelude are standing
sources for every Inkling pass, and the previous core output M_{k−1} is the message, admitted
through the gate g. The design borrows Inkling's schedule and gated restart, and no more: Book 1's
block keeps its ordinary residuals, the sole message enters the first core block, and there is no
per-input mixture; production Inkling has no single residual stream, sums each block's layers from
zero, and mixes both previous middle-block sums separately at every Q, K, V and MoE input on the
second and third passes, through κ = 1 − tanh(Fz), γ = tanh(Gz). The announcement says that gate at first
blocks the message entirely; here the version of §6, item 6 is exact about it — with g = 0 the
sandwich is exactly the four-block checkpoint's predictor, for arbitrary existing weights. No
intermediate readout or per-pass loss is described for Inkling, and the sandwich likewise reads out
once; the specimen's deep supervision does not carry over.

| stored component | count |
|---|---:|
| embedding, final LN, output matrix and bias, 2VD + 2D + V | 16,961 |
| one block, 4D² + 2DF + 9D + F | 198,272 |
| Book 1's four blocks plus embedding/readout | 810,049 |
| shared 128-channel gate | 128 |
| total | **810,177** |

Depth is 1 + 2 × 3 + 1 = 8 block applications per token against Book 1's four — 2,048 token-block
applications per length-256 window, excluding embedding, gates and readout; eight *untied* blocks
at equal depth would store 1,603,137 parameters. Weight sharing does not share activations: each
causal attention application, including each repeated core pass, uses its own past K/V activations
once the gate opens. The pullback is finite unrolling, no equilibrium inverse: pull the single loss
through the readout and B₄; for k = 3, 2, 1 pull M̄_k through B₃ then B₂ to V_k = Ȳ_k, sum each
block's parameter cotangents over its three calls, accumulate Ū += V_k,
ḡ_i += Σ_n V_{k,in}·M_{k−1,in} and M̄_{k−1} += g⊙V_k; discard the cotangent of the fixed zero M₀;
call B₁'s pullback once with the summed Ū and scatter to the embedding; store one parameter vector
and one optimiser state per shared parameter. The six core block tapes, the prelude and coda tapes
and the preceding messages are retained or recomputed.

**Reading for the book (revision of 22 Nov).** The closing chapter now has a model the reader
already knows block by block, changed only in schedule and in one gated line, against a real
model's published layout. What it could honestly claim *before any new run* is the accounting
above and the closed-gate identity; the 21 Nov numbers (Book 1 1.5313 held out; stock tied ×4
1.654; the windowed specimen recipe 1.626 at six passes) are historical context. The latter two
used 215,233 parameters, and none of these losses estimates the sandwich's. What it
could claim after §13, run 4 is whatever that run measures against Book 1 at equal updates and at
matched block applications — alongside Inkling's reported results. In the source note: the variable-loop variant's
chained-lookup accuracy kept improving through ten passes and, on GPQA Diamond questions correct
at every pass count, mean written reasoning fell from 3,703 tokens at one pass to 1,661 at five
(1,731 at three), while closed-book factual recall barely moved; the released model always runs
three passes, with no halting, and the variant was trained at 1–5 passes in the
ratio 1:2:6:2:1 in later midtraining only — none of which licenses a claim that the sandwich's
extra passes will help on Shakespeare, only that "thinking longer" and "knowing more" are different
claims and must be measured separately. The 21 Nov open questions (one seed per model; window and
unroll changed together between A and B; the separate effect of each stabiliser) stay open for the
runs they belong to.

## 13. Runs for mains power

Every run the revision of 22 Nov needs, in the analysis's order (`loop/notes/written-vs-looped.md`
§7, "Tests to schedule on mains power"). Nothing here was run for this revision. Expected durations
are the analysis's conjectured allowances, built from the 21 Nov timings — specimen-size update loops of
about 10 s per seed (the frozen 30-seed study; 6.5–9.9 s of updates in the screen) and Shakespeare
timings of roughly 72–90 min under the laptop's Power Saver profile (updates and wall time;
stock tied and looped wall times of 75.1 and 83.1 min in §12, against 24.9 min for one pass;
Book 1's 36.2 min counts updates only).
Compilation, validation and the hardware power profile affect elapsed time.

1. **Written and latent parity** — allow 5–15 minutes. Fit the three end-only schedules of §1
   (one pass; written feedback; tied passes) on matched words and initial seeds. Give the written
   model gold pairs in training and its own argmax in evaluation. Record that extra parity input
   when comparing training signals. Report end-only cross-entropy,
   final accuracy, intermediate-token count and evaluated columns separately. Compare free rollouts
   with teacher forcing through lengths 32 and 64; inject independent bit flips as a positive
   control for the error law ½(1 − Π(1 − 2ε_i)). Retain the untrained hand-set witness of §1 as an
   exact reference. This run tests whether the trained seeds do the task both ways and whether their free-rollout
   errors follow the independent-flip law.
2. **Does the gate replace stream normalisation?** — allow 20–60 minutes with exhaustive
   validation. Cross gate enabled/bypassed with the final stream LayerNorm present/absent, with
   identical anchored blocks, seeds, pass schedule and objective; bypassed means fixed g = 1.
   Start with five seeds per cell; budget 30 fresh seeds per cell before making a reliability claim
   against the measured 25/30 of §1. Log the gain, the closed-gate gradient on short and long words,
   the reached front, centred state steps and estimates of the Jacobian norm. If the gate stalls at
   closure on the end-only objective — the population gradient there is proved zero under the
   conditions of §1 — compare an end-only run against prefix supervision. This run decides whether
   §6, item 6 joins the specimen or stays a proved alternative, and whether item 4 of §6 can come out.
3. **Adaptive hops and doubling** — allow 10–30 minutes for a small synthetic sweep. Probe an
   immutable-table query and a parallel writable-summary scan separately; sweep hop count and
   passes, permute addresses, and inspect the decoded intermediate addresses. A chain-length
   accuracy curve alone cannot diagnose doubling, and the same caution applies to reading the
   variable-loop variant's chained-lookup curve as a hop count. This run supports the
   steps-versus-distance chapter (revised §5, chapter 4) beyond the window's one-edge-per-pass case.
4. **The Shakespeare sandwich** — allow 2–3 hours per 5,000-update eight-application run, plus
   controls and hardware-dependent overhead (against the 75.1 min stock tied Power Saver
   run of §12; the allowance is conjectured in the analysis and is to be replaced by a measurement).
   First the checks, before any long run: exact closed-gate logits against the four-block weights,
   and finite differences through the core, the gate and the prelude on a tiny configuration. Then
   pair a scratch-trained sandwich with Book 1 at equal optimiser updates, and compare about 2,500
   sandwich updates against 5,000 four-block updates at the same batch shape. This matches block applications and gives
   the sandwich half as many training tokens. Record
   tokens seen under each budget, final-only validation cross-entropy on the same windows, and wall
   time. Trying other pass counts afterwards is a diagnostic; the shipped setting that motivates
   the design is three.

The analysis's own short checks (the witness margins, the exact error enumeration, the gate
identity and the parameter totals) rerun from the repository root with one Julia thread by the
command given at the end of `loop/notes/written-vs-looped.md`; they are checks, not training runs —
no duration for them is stated in the notes and none is claimed here. Runs 1–3 are specimen-sized
(written versus looped; the gate; steps versus distance); run 4 is the only Shakespeare-scale item
and the only one whose outcome can change what the closing chapter asserts rather than how it is
phrased.

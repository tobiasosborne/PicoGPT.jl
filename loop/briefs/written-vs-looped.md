# Brief: written versus looped — the analysis for the revised looped book (Astra)

## Context

Read `loop/CONCEPT.md` (the concept and its results, incl. §12 Shakespeare), `loop/notes/derivations.md`
(your own earlier derivations), `loop/notes/window.md` (the light-cone specimen: prefix parity over
{be, not}, window-2 attention, 270 params; the 2-gain register), `loop/notes/audit.md`, `BRIEF.md` §1–3
(notation). The user approved a revision after comparing our specimen with a real shipped looped model,
Inkling 2 Mini (Thinking Machines, 20 Nov 2026; source copies in `/tmp/claude-1000/-home-tobias-Projects-PicoGPT-jl/ffbe9658-9762-4fc0-a2bf-fb9b7fd59bdd/scratchpad/web/inkling/`; a separate agent is
writing `loop/notes/inkling.md` from them). In brief: Inkling has 24 layers in four blocks of six; first
and last run once; the middle two run three times with shared weights; every pass restarts from the first
block's output, and passes 2–3 also read the previous pass's output (the "message") through a gate that
starts closed (x' = κ⊙x + γ⊙m, κ = 1 − tanh(Fz), γ = tanh(Gz), per Goodfire's appendix: verify in the
source files). Looping buys serial steps (chained table lookups improve to ten passes), not knowledge,
and replaces written chain of thought.

Five approved changes to the concept; your analysis decides whether each holds up:
1. **Written versus looped** (must not be skipped). Our specimen labels every word, which is its own
   scratchpad: feed each answer back and one pass suffices. Show the negation task three ways: one pass
   with nothing written; one pass that writes intermediate answers and reads them back (a tiny chain of
   thought); the loop with nothing written.
2. **The window of two as a device**: in real models the per-pass limit is dependent steps, not distance;
   the window turns steps into distance. Chained lookup is the faithful task; relate parity on a line to it.
3. **The gate as the stabiliser**, in place of or beside the parameter-free normalised stream.
4. **Shrink and relabel the fixed-point material** (many passes, loss at every pass, (1 − Jᵀ)⁻¹) as the
   Huginn / small-recursive-model family, not what a shipped looped model does.
5. **Shakespeare on Inkling's layout**: Book 1's four blocks, first and last once, middle two three
   times, gated message. Design and accounting only.

## Deliverable: `loop/notes/written-vs-looped.md` (extend derivations.md by reference where it has the pieces)

- **The three-way comparison made exact.** Choose the cleanest task variant (e.g. the answer demanded only
  at the end, or at marked positions) and the token format for the written version, ideally one in which
  the window of two still suffices. For each way: the exact loss floor, a hand-built or provably existing
  solution in the specimen's block where one exists, and the cost in block applications and in tokens
  written. State the exchange rate between passes and written tokens; say honestly when writing is cheaper.
- **Error propagation.** In the written version run free (the model reads its own answers), with per-step
  error ε the parity is wrong after t steps with probability (1 − (1 − 2ε)^t)/2 if errors are independent;
  check it, and give the corresponding statement for the loop (Ising chain / binary symmetric channel).
- **The gate**: the smallest gate that fits the specimen (scalar or per-coordinate gain on the message, zero
  at initialisation); its pullback; proof that the closed gate is exactly the one-pass model; whether the
  gate's gradient is nonzero at initialisation; what the gate does to the Jacobian of the pass map
  (contraction for small gain; relation to the eigenvalue pinned at 1 by LayerNorm's symmetry, §5 of
  derivations.md).
- **Steps versus distance**: a precise statement of what the window does, and the general bound for chained
  lookup in hops per pass, including the pointer-doubling case you already have.
- **The sandwich on Book 1's Shakespeare model** (D=128, H=4, F=512, T=256, V=65): forward definition,
  parameter count, block applications per token, what the pullback needs. No run.
- A short list of the runs a coder should do later on mains power to test these claims, with rough durations.
- Mark every claim proven / checked (how) / conjectured. Say plainly if a change does not hold up.

## Rules

**Battery rule:** the laptop is on battery. Julia only for checks that finish in seconds on one or two
threads (exhaustive enumeration at small length, a gradcheck on a tiny model, verifying hand-built weights);
no training. Write checking code in `loop/scratch/wvl/`. Edit only those files and the deliverable, plus
`loop/briefs/written-vs-looped-report.md` (your summary). Do not edit CONCEPT.md, src/, or anything else.
Do not commit. Do not start Muse or Gemini agents. Do not change system settings.

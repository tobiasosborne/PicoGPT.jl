# Growing a transformer by renormalisation: synthesis of the first round

24 November 2026. From the thirteen reports on `origin/grow/*`, their scripts and saved outputs.
Two short controls were run for this synthesis (`grow/synthesis/`), marked "added here".

**Terms.** A *lift* makes a larger model (the child) from a trained small one (the parent),
computing the same function at birth. *Staged* training trains the parent, lifts, then trains the
child; *scratch* trains the final model directly. The *floor* is a world's exact minimum loss. The
*defect* of a coarse-graining π is I(πY; H | πH), what the fine history says about the next coarse
symbol beyond the coarse history; it is zero exactly when the coarse world is closed. The
*specimen* is the 1,007-parameter model of `to be or not to be`.

## 1. What is established

**Proved** (derivation notes with numerical checks; I verified some arguments by hand, not all).

- Exact lifts exist for a vocabulary split, MLP units, a block or pass behind a zero gate, longer
  context on old prefixes, and width by duplicating coordinates if the position code is duplicated
  too. Refuted: more heads at fixed width; width duplication with PicoGPT's own position code.
- Duplicated width is a symmetric point that training cannot leave. Another exact lift breaks it.
- A coarse world is closed exactly when the coarse history is predictively sufficient (strong
  lumpability for Markov chains). The price of forgetting is the defect.
- A block of attention positions merges exactly only into the pair (Σ eˢ, Σ eˢ·value), which one
  ordinary key reproduces only if the keys coincide.
- White's rule has a local counterpart, an eigenproblem only when stream covariance and output
  sensitivity separate. There is no canonical form and no two-site update. Bounded new information
  per level does not bound model size.

**Measured, adequate evidence** (5 to 8 seeds, exact floors, one optimiser for both arms; checked
against the output files).

- Width, specimen (8 seeds): growing 4 → 8 costs 1.0 to 2.4 times scratch. The child needs as many
  updates as a fresh model (109 to 174 against 124).
- Vocabulary, specimen (5 seeds): 3.4 times scratch. The coarse text has the higher floor (0.152
  against 0.059 nats) and trains more slowly.
- A synthetic two-level Markov world (5 seeds, 10 settings): the child beats scratch only when a
  class's two children are interchangeable as inputs (0 to 10 steps against 30); otherwise it
  needs 1.2 to 2.4 times scratch's steps. The parent's cost rises from 30 to 1,155 steps as the
  defect goes from 0 to 0.52 nats.
- None of the 877 partitions of the specimen's alphabet is closed (exhaustive).

**Measured, thin evidence.**

- Context (2 seeds, one world, one size): T = 8 then 32 reaches the floor in 1.6 to 1.7 million
  tokens; scratch at T = 32 is still 0.7 nats above it at 2 million (one seed) and 5 million (the
  other).
- Doors (9 seeds, bimodal, unequal caps): with a second pass behind a zero gate, 7 of 18 children
  reach 99 % against 1 of 9 scratch runs; Fisher's exact test gives p = 0.2.
- Truncation (one seed per width): weighting by output sensitivity beats PCA; the spectrum falls
  slowly, with no gap.
- Blocked attention in a language model (one seed); towers on Shakespeare (greedy, in-sample);
  prior art (abstracts only). `passes` trained in neither regime.

## 2. Agreement, contradiction, overstatement

Three reports agree that a coarse world which is not closed is *harder* than the fine one
(`tower`, `hierarchy`, `lumpable`). Two agree that the parent must be revised, not frozen. Two
agree that the internal condition R∘F₂ = F₁∘R fails (proved at LayerNorm; measured at 10 to 60 %
per half-block).

Corrections to the README and its status table:

1. Its closing paragraph says staging lost where "the coarse stage was no cheaper". In `lifts`,
   and in `hierarchy` at zero defect, the parent was cheap. Staging lost because the child then
   needed scratch's updates or more.
2. "Exactness can be a poor place to start" is not supported. The best width arm is exact. In
   `hierarchy` the noise that brings parity is as large as a fresh initialisation; it discards
   the inheritance.
3. "A closed gate has zero gradient on the end-only objective" holds for parity with a calibrated
   uniform readout. In `doors` the gradient at exact closure was nonzero in all 18 runs.
4. `lumpable`: "about ln 2 per doubling" is 0.14 to 0.72 nats, and a binary split cannot add more
   than ln 2 in any world. It is no evidence for an area law.
5. `truncate`: "rank 5 suffices" is 8 % and 19 % above the floor, and a native width-4 model
   reaches the floor.
6. `doors`: one counted success used more work than the scratch cap. Added here: scratch with the
   gate starting at zero, same weights and data, reaches 99 % in 0 of 9 seeds, so the gate's
   starting value is not the cause.
7. `context`: the numbers are right. Added here: the scratch plateau persists at half and at double
   the learning rate (one seed). But at birth the child is worse than uniform on the new positions.
   It does not "learn only what is new".
8. `passes` shows one thing: a lift that damages the parent's answers (birth loss 2 to 17 times
   the floor) is no better than a random start.

## 3. What the round says about the idea

Staging did not pay wherever scratch converges easily, in about 100 updates: width, vocabulary,
the Markov world. It paid in two worlds where scratch sits on a plateau. In `passes` scratch also
stalls, the lift damaged the parent, and staging did not help. One hypothesis covers all cases:
staging is a route round an optimisation barrier, and pays only if scratch has one and the lift
keeps the parent's competence on its old domain. This is the known curriculum effect. No
experiment separated it from the exact lift, and nothing ties it to renormalisation. It is
untested.

Two parts of the renormalisation picture survived. Coarse-graining generates couplings outside
the family, three times over: the log-partition score under blocking, a LayerNorm that is no
longer round under width truncation, longer memory in a coarse world. And the environment decides
what to keep: truncation weighted by output sensitivity beats PCA.

Four did not: the coarse world as a cheap warm-up, the child that learns only what is new, the
internal intertwiner, a fast-falling spectrum. A canonical form and a two-site update are proved
absent.

Never tested: sweeping back; the area law as a statement about work or parameters; more than two
levels; fixed points and relevance (no scale map is defined); growth above 10,000 parameters.

## 4. The owner's two principles

**Approximate growth.** Approximation is forced on the *worlds*: none is closed, so one measures
the defect. It is not needed for the *lifts*: exact ones are free. What must be kept is the
parent's competence on its old domain. The `context` child is exact there and poor elsewhere, and
wins; the `passes` child is damaged there and gains nothing.

**Access is earned.** Untested, but four observations bear on it.

- Waiting for the parent's floor is the wrong gate. In `lifts`, lifting after 50 updates beat
  lifting after 200; in `hierarchy` the wait made lineages cost 8 to 10 times scratch.
- Competence mattered in `doors`: no child of the four unconverged parents succeeded, against 7 of
  14 for the rest. The gate gradient at birth did not predict success.
- Local refinement has a cheap criterion: a class splits almost for free when its children are
  interchangeable as inputs.
- "One world, a moving cutoff" suggests coarse-graining the *targets*, not the *inputs*. On the
  specimen the class given the fine history has floor 0.041, given the coarse history 0.152.
  With fine inputs, H(Y|H) = H(πY|H) + H(Y|πY,H) has no defect term. This removes the main loss
  found today.

## 5. Recommendation

Continue for one more round, narrowly, and stop if experiment 1 fails. The bookkeeping is sound;
the claim that growing saves work rests on two small results.

1. **Are the two positive results more than a plateau in one small baseline?** Smallest version:
   `context` and `doors` as they stand, 10 seeds, equal budgets with the parent charged; scratch
   at three learning rates and twice the width; in `doors` also a child born with the gate open,
   to separate lift from curriculum. For: staged at least twice as cheap as the best scratch in 8
   of 10 seeds, both worlds, both sizes. Against: a tuned or wider scratch within 1.5 times.
   About 3 CPU-hours.
2. **Does earned, local access pay where the coarse level is the expensive one?** Smallest
   version: the two-level world with `context`'s lag-12 dependency on the class and a closed
   split; fine inputs, coarse targets first; 10,000 parameters, 10 seeds. Arms: scratch; all
   classes split on a schedule; each class split when its own loss stops falling; coarse inputs
   as today. For: whole-lineage work below 0.7 times scratch, the gated arm no worse than the
   schedule. Against: the child again needs scratch's steps; then close the vocabulary axis.
   About 4 CPU-hours.
3. **Passes behind a gate, on the model that trains.** Smallest version: the 270-parameter looped
   specimen unchanged, prefix parity, one pass and one position added at a time behind a zero
   gate up to 7, against scratch at 7 passes, 10 seeds. For: the floor is reached with less work
   or more often, and closing the last gate returns the parent. Against: no difference. About 1
   CPU-hour. Here the child gains capability and no parameters.

Stop: growth experiments on the specimen text; frozen-parent arms; coarse inputs as the parent's
world; searching text for closed partitions; a single global intertwiner; low-order expansions of
blocked attention on trained heads; work ratios from fewer than five seeds.

## 6. Open risks

- **The baseline.** One optimiser setting, one size, 2 and 9 seeds. In an unsaved `doors` run
  with four heads scratch reached 99 % in 2,600 steps; one head was kept because it is hard.
- **Curriculum, not growth.** No control separates the exact lift from easy-then-hard ordering.
- **Built-in structure.** Both winning worlds were designed with the two scales staging expects.
- **The work measure.** Parameter-updates or tokens; by the clock every grown width arm lost.
- **Scale.** Nothing above 10,000 parameters was grown; published savings at scale are 22 to 45 %.

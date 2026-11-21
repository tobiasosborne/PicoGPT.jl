# Brief: looped-transformer prototypes in Julia, round 1

## Context

PicoGPT (this repository) is a GPT written from scratch in Julia: no autodiff, every operation
returns its value and its pullback, gradients checked by `gradcheck`. Its book's specimen has 1007
parameters and trains on `to be or not to be`. We are exploring a **third book on the tiniest
looped transformer**: one PicoGPT block, the same parameters, applied K times to its own output.
Today's job is evidence, not polish: which tiny task makes the loop *load-bearing* and *visible*,
with numbers. In parallel, Astra (a mathematician agent) is writing `loop/notes/derivations.md`
(the loop pullback, hand-built weight solutions, impossibility arguments); you may read it if it
appears, but do not wait for it.

Read first: `BRIEF.md` §1–3, §13 (notation and recipe; binding), all of `src/` (≈900 lines),
`test/runtests.jl`, `jspace/README.md` and the top of `jspace/src/JLens.jl` and
`jspace/scripts/export_specimen_jlens.jl` (the model of how a supplement folder reuses the
package), and `loop/CANDIDATES.md` (the candidate specimens and the criteria).

## What to build (paths you own: `loop/src/`, `loop/test/`, `loop/scripts/`, `loop/artifacts/`, `loop/notes/prototypes.md`)

1. `loop/src/Loop.jl`, module `Loop`, `using PicoGPT`: the looped model, written in the style of
   `src/*.jl` (short functions, docstrings stating the maths, comments saying why), reusing
   `embed`, `block`, `layernorm`, `affine`, `crossentropy`, `carve`, `init`, `AdamW`, `clip!`,
   `warmup_cosine` and friends. It must support:
   - K passes of the *same* block (θ holds one block; build a Config with L = 1 and loop it);
   - optional input injection (X₀ re-added each pass; implement X_{k+1} = B(X_k) + X₀ and
     X_{k+1} = B(X_k + X₀)-type variants only if cheap; record which you used);
   - readout and loss at the last pass, or averaged over all passes (deep supervision);
   - a per-pass trace (the stream, the attention, the per-pass predictions) for analysis;
   - optional non-causal attention only if a candidate needs it (keep it additive; do not edit
     `src/`; if you need a variant op, write it in `loop/src/`);
   - the pullback through the loop by hand: X̄_k from X̄_{k+1}, θ̄ summed over passes. Test it with
     PicoGPT's `gradcheck`/`checkpullback` in `loop/test/runtests.jl` (relative error < 1e-6, Float64).
   - the controls: an unlooped model of the same θ size with one pass; an untied L = K model; an
     unlooped model of matched parameter count (e.g. wider F) where it makes sense.
2. Prototypes, each a script in `loop/scripts/` that trains in seconds to a minute with AdamW and
   the BRIEF §13 recipe as a starting point (tune if needed and say so), fixed seeds, several seeds
   for robustness, and writes its numbers to `loop/artifacts/<name>.json`:
   - **P1 prefix parity, "not not to be"** (candidate 1): bits (or the tokens `not` / a neutral
     word; a tiny vocabulary of 2–3 symbols), predict the running parity at every position, looped
     with injection. Report: parameter count; training curve; accuracy per position per pass (the
     light cone: after k passes, which positions are right?); behaviour when run with more passes
     than trained; behaviour on longer inputs than trained (the positional code `positions(D, T)`
     extends to any T; note that `gpt`'s context check uses c.T, so write your own forward); and the
     same numbers for the controls. Try D = 8, H = 1 or 2, F small; try to find the *smallest* model
     that works. Say whether the learned front moves like k (linear) or faster (doubling).
   - **P2 rule 60** (candidate 2): the same block without injection, trained end to end on k-step
     targets of the causal cellular automaton s_t ← s_t ⊕ s_{t−1}; compare with Lucas' theorem;
     run beyond the trained k (does the Sierpinski triangle keep growing?).
   - **P3 pointer chasing** (candidate 3), if time allows: follow a permutation k hops; control at
     one pass.
   Explore freely within this: if a variant (with or without LayerNorm drift, deep supervision,
   curriculum over K) is what makes a prototype work, report that, it is itself a finding.
3. `loop/notes/prototypes.md`: for each prototype, the exact setup, the numbers (tables are fine
   here, this is a lab notebook), what worked, what did not, and the one or two observations a book
   could be built on. Every number must come from a script in `loop/scripts/` that reruns it.

## Rules

- Do not edit anything outside the paths you own. Do not commit. Use Float64 for the specimen-sized
  models. Run `julia --project` from the repository root.
- Keep each script fast: the point is that the final specimen could train live in a browser.
- Report back (and copy to `loop/briefs/prototypes-report.md`): what you built, how you verified
  the pullback, the headline numbers per prototype and control, what failed, and your recommendation
  of which prototype is the specimen.

# Brief: the light-cone specimen — looped prefix parity with a two-token attention window

## The idea (orchestrator's, to be tested)

Round 1 (`loop/notes/prototypes.md`) showed that with full causal attention the loop is not forced:
at small T one pass can read every bit, the optimiser finds shortcuts, and no durable recurrence
is learned. Astra (`loop/notes/derivations.md` §6, row "Conditional information under a strict
local rule", and §7 last paragraph) notes that if attention is restricted to a local window, the
receptive field gives an **exact Bayes floor** for the loss. This brief turns that into a candidate
specimen with an anchor of the same kind as the first book's entropy floor.

**The one change to PicoGPT:** the causal mask M[i,j] = −∞ for j > i becomes a *window* mask,
M[i,j] = −∞ unless i − w < j ≤ i, with window w = 2 (each token reads itself and its
predecessor; position 1 reads only itself). ("Sliding-window attention" is real ML jargon.) Then
information travels at most one position per pass: after k passes the readout at position t can
depend only on x_{t−k}, …, x_t. That is a light cone (physics: a Lieb–Robinson bound for a
nearest-neighbour lattice). The task stays prefix parity ("not not to be": token 2 = negation,
token 1 = neutral word; the target at t is the parity of negations up to t), looped with input
injection X_{k+1} = B(X_k) + X₀ (use round 1's `Loop.jl` modes; say which you use).

**The anchor.** For uniform independent bits, the target at t after k passes is determined if
t ≤ k + 1 and exactly fair (posterior ½) otherwise. So the cross-entropy at pass k has the Bayes
floor ln 2 · #{t ≤ T : t > k + 1}/T, and with deep supervision (loss averaged over passes 1…K) the
training loss has the floor

    ℓ* = (ln 2 / (T K)) Σ_{k=1}^{K} max(T − k − 1, 0)     (e.g. T = 8, K = 8: 21 ln 2 / 64),

reached only by a model that is right everywhere inside the cone and *exactly undecided* outside it.
**Verify this formula yourself by exhaustive enumeration** before using it (check the offset: does
the first pass already see x_{t−1}? Count carefully what the readout after pass k can see given
where the embedding enters; correct the formula if needed and say so). Controls with exact floors
of their own: one pass (floor ln 2 · (T − 2)/T at T, any width), untied L = K (can reach the same ℓ*
but with K× the block parameters), and the looped model at test time with more passes and longer
inputs (the prediction: a learned *local* rule works at every length and pass count, with front
t = k + 1).

## What to do

1. Implement the window mask as an additive option in `loop/src/` (a new file, e.g.
   `loop/src/window.jl`, included by or alongside `Loop.jl`; a variant of PicoGPT's `attention`
   with mask argument, the same pullback, gradchecked in a new test file `loop/test/window.jl`).
   Do not edit `src/`. Round 2 of the prototypes (another Sol agent) is concurrently editing
   `loop/src/Loop.jl`, `loop/scripts/*` it created, and `loop/notes/prototypes.md`: **create only new
   files** (`loop/src/window.jl`, `loop/scripts/window_*.jl`, `loop/test/window.jl`,
   `loop/artifacts/window-*.json`, `loop/notes/window.md`); if you need a change in `Loop.jl`, copy
   what you need into your new file instead.
2. Train from random initialisation (BRIEF §13 init; AdamW; tune steps, η, batch; fresh random words
   each step or the full enumeration at T = 8, which is 256 words). Find the **smallest** model that
   works: try D = 8, 6, 4; H = 1; F small. Deep supervision vs final-pass loss: report both.
3. Success means: across 5 seeds, the training loss reaches ℓ* to within ~1 % (report the gap), and
   per pass the model is right inside the cone and ≈ ½ outside it; then, *without retraining*,
   at T = 32 and 64 with K = T passes it is right at every position (accuracy ≥ 99 %), and extra
   passes (to 2T) do not degrade it. Report seeds that fail, and why if you can tell.
4. For the best model record: parameter count; the training curve against ℓ*; accuracy and mean
   predicted probability per position per pass (the cone); whether the attention is a clean
   previous-token/self split; |X_{k+1} − X_k| per pass after the front has passed (does the state
   reach a fixed point, or does the stream drift? centre each column too, see derivations §5); and
   the pullback seen as a covector travelling back one position per pass (seed a logit difference
   at one site, terminal loss, save X̄ per pass).
5. The same with rule 60 (autonomous X_{k+1} = B(X_k), window 2, train on k-step targets for k
   in 1…K with deep supervision, compare with Lucas' theorem beyond K) only if P1 succeeds and time
   remains.
6. Time each training run. The specimen must train in seconds.

## Rules

Create only the new files named above. Do not commit. Do not start Muse or Gemini agents. Every
number in your notes comes from a script that reruns it. Report back (copy to
`loop/briefs/window-report.md`): does the light-cone specimen train, the smallest working size, ℓ*
and the gap, length/pass extrapolation, the controls, and anything that surprised you.

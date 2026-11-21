# Brief: the mathematics of the tiniest looped transformer (derivations, hand-built solutions, impossibility arguments)

## Context

PicoGPT (this repository) is a GPT written from scratch in Julia with hand-derived pullbacks, and a
book for BSc physics and maths students. Its specimen has 1007 parameters (V=7, D=8, H=2, d=4,
F=32, L=1, T=8), trains on `to be or not to be`, and has an exact loss floor (2 ln 2 + 3 ln 3)/80.
We are exploring a **third book on the tiniest looped transformer**: one block B_θ of exactly the
PicoGPT kind, applied K times with the same θ. We need a specimen whose loop is load-bearing, whose
computation is visible pass by pass, and which has an **exact anchor** (a number or structure the
theory predicts and the trained model reaches). Your job is the mathematics; a coder (Sol) is in
parallel building the looped model in Julia and running prototype experiments in `loop/`.

Read first: `BRIEF.md` §1–3 and §12–13 (notation is binding), `src/block.jl`, `src/attention.jl`,
`src/embedding.jl`, `src/model.jl`, `src/parameters.jl`, and `loop/CANDIDATES.md` (the
orchestrator's first list of candidate specimens, with guesses that you should test, correct or
kill). The block is exactly:

    Â = LN₁(X);  Q,K,V = Wq Â + bq, Wk Â + bk, Wv Â + bv;  per head A = softmax_rows(QᵀK/√d + M)
    X₁ = X + Wo (V Aᵀ) + bo;   X₂ = X₁ + W2 relu(W1 LN₂(X₁) + b1) + b2

with causal mask M, LN with ε = 1e-5 and learned γ, β, sinusoidal positions P added once at the
embedding (positions(D, T) in src/embedding.jl), readout Z = Wout LN_f(X) + bout.

## What to derive (write each as a short, checked section; index form and matrix form side by side)

1. **The loop and its pullback.** Define the looped model precisely in two or three variants:
   (a) autonomous, X_{k+1} = B(X_k); (b) with input injection (choose the simplest form that keeps
   the Euler-step reading of the block; say why, e.g. X_{k+1} = B(X_k) + X₀ vs X_{k+1} = B(X_k + X₀)
   vs others, and what each does to the growth of |X|); (c) readout at every pass with a loss averaged
   over passes ("deep supervision"). Derive the pullback: X̄_k = J_kᵀ X̄_{k+1} (+ injected terms),
   θ̄ = Σ_k θ̄_k. Then the fixed-point case: if X_k → X*, the gradient of a loss at X* is the
   implicit-function formula with (1 − J*ᵀ)⁻¹, and the truncated loop gives its Neumann series. State
   exactly when the series converges and what it is when J* is nilpotent.
2. **Prefix parity ("not not to be", candidate 1).** (i) A hand-built solution: explicit numbers for
   θ in a PicoGPT block (D as small as possible, ideally D=8 or less, H=1 or 2, small F) that, looped
   with injection, computes s_t = x_1 ⊕ … ⊕ x_t at every position, correct for t ≤ (number of passes)
   and stable thereafter (the correct state is a fixed point). Handle LayerNorm honestly (it rescales
   each column; show the construction survives it, or say what must change). Handle softmax honestly
   (attention to the previous token is only approximate; bound the error, or say how large the
   scores must be, and whether sinusoidal positions with D=8 allow a sharp previous-token head at all
   T up to, say, 64). If the full block cannot do it exactly, give the smallest honest deviation.
   (ii) The impossibility/control argument: why one pass of the same block cannot compute prefix
   parity at lengths beyond some bound, ideally in one sentence of mathematics (e.g. counting via
   uniform attention plus a ReLU MLP with F units has ≤ F+1 linear pieces, so ≤ ~F/2 parity flips).
   Make it rigorous or say exactly what it does and does not prove (fixed T vs growing T; softmax
   attention with bounded weights; Hahn 2020-type sensitivity arguments).
   (iii) Does a log-depth (pointer-doubling, Wyllie-style) solution exist with *shared* weights in
   this block, the pointer being stored as data in the stream? If so, which accuracy-vs-pass curve
   distinguishes it from the linear light-cone solution (front at t ≈ k vs t ≈ 2^k)?
3. **Rule 60 (candidate 2).** Confirm: autonomous loop of s_t ← s_t ⊕ s_{t−1} gives Lucas' theorem
   s_t^{(k)} = Σ_j C(k,j) s_{t−j} mod 2; the relation to windowed parity at k = 2^m − 1; whether the
   *same* hand-built block as in 2(i), without injection, realises it.
4. **Pointer chasing (candidate 3).** A hand-built or near-hand-built solution for following a
   permutation k hops in k passes, and the argument that one pass cannot do two hops (cite the
   result you use; say precisely what is proven for one softmax attention layer).
5. **Linearisation and the light cone.** For the parity solution, compute (or characterise) the
   Jacobian J* = ∂B/∂X at the fixed point: is it (close to) a shift / nilpotent? Then the pullback
   travels one position per pass: say what the reader would see if the covector X̄_k were drawn pass
   by pass.
6. **Exact anchors.** List every exact number or structure the theory predicts that a trained model
   could be seen to reach (e.g. a loss floor if the task has ambiguity, the front speed, a fixed
   point, a spectrum, a parameter count). For each, say how a coder should measure it.
7. **Audit** `loop/CANDIDATES.md`: which claims there are wrong, and any candidate it missed that
   is better (state it and its anchor).

## Rules

- Correctness first; mark every claim proven, checked numerically (say how), or conjectured. You may
  write small Julia scripts in `loop/scratch/astra/` to check numbers (use the package: `using
  PicoGPT`; run with `julia --project`), but do not edit `src/`, `book/`, `web/`, `jspace/` or any
  other file under `loop/` than those named here. Sol is writing `loop/src/`, `loop/test/`,
  `loop/scripts/`: do not touch them.
- Prefer exact constructions whose weights a reader could check by hand. If a construction needs
  large weights (sharp softmax), give the numbers and the error bound.

## Deliverable

`loop/notes/derivations.md` (the mathematics, sections 1–7, in BRIEF.md notation, with every
hand-built θ given as explicit arrays, also saved as Julia code in `loop/scratch/astra/handbuilt.jl`
returning a θ vector in PicoGPT's layout for a stated Config so that the coder can load it), and a
final report copied to `loop/briefs/derivations-report.md`: what you proved, what you checked
numerically, what failed, and the three things the orchestrator most needs to know.

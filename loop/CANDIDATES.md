# Candidate specimens for a book on the tiniest looped transformer

First list, written by the quest orchestrator before any prototype ran (21 Nov 2026). It is a
working document: the prototypes and derivations decide. Criteria (from the quest brief):
**load-bearing loop** (a control fails, ideally provably), **visible per pass**, **specimen-sized**
(hundreds to low thousands of parameters, trains in seconds), **exact anchor**, **shows what looping
buys** (more passes at test time, longer inputs, fewer parameters).

Notation: the looped model applies one PicoGPT block B_θ (same θ) K times. "Input injection" means
the embedded input X₀ is added back at every pass; without it the loop is autonomous,
X_{k+1} = B(X_k), a discrete dynamical system.

## 1. "not not to be": prefix parity as negation  (lives: first choice)

Line: a sequence of words/bits, each either a negation (`not`) or neutral; at each position predict
whether what has been said so far affirms or denies (parity of the negations so far). Prefix XOR
written as double negation, with Hamlet in the background.
- Loop: input injection; natural learned algorithm s_t ← x_t ⊕ s_{t−1}, a previous-token head plus
  an XOR in the MLP; information travels one position per pass. A light cone: after k passes,
  positions t ≤ k (+ offset) are right.
- Control: one pass can compute parity at a *fixed* length by counting (uniform attention gives
  the fraction of nots, the MLP must fold it into a sawtooth), but a ReLU MLP with F hidden units has
  at most F+1 linear pieces, so it cannot fold beyond ~F teeth: an unlooped model fails at lengths
  beyond training, a looped one need not. One-sentence mathematics candidate.
- Anchor: the light cone (front speed 1 per pass); the linearised loop is a shift (nilpotent), so
  the pullback through the loop also travels one position per pass and its Neumann series
  terminates. Hand-built weights likely writable.
- Physics: product of signs ±1 = an Ising chain's domain-wall count; transfer matrices.
- Open: does training find the linear-depth algorithm or a log-depth pointer-doubling one?
  Accuracy front vs k (linear vs 2^k) would show which: itself an exhibit.

## 2. Rule 60, the Sierpinski triangle  (lives: pairs with 1)

Same block, no injection: s_t ← s_t ⊕ s_{t−1} is the causal elementary cellular automaton rule 60
(left-neighbour XOR). Iterating it draws Pascal's triangle mod 2.
- Anchor: Lucas' theorem, s_t^{(k)} = Σ_j C(k,j) s_{t−j} mod 2, exact for every k; at k = 2^m − 1
  all binomials are odd, so the CA computes windowed parity: candidates 1 and 2 are the same block
  with and without input injection.
- Visible: the spacetime diagram is a trace in the sense of BRIEF §11.
- Risk: if trained one step at a time the loop is not load-bearing in training; train on k-step
  targets end to end.

## 3. Following the cycle: pointer chasing in a permutation  (lives)

Context writes a permutation as pairs (`a→c c→b b→a`), then a start symbol; predict the k-th
successor. One hop per pass.
- Control: composing two lookups needs two attention layers (known one-layer limits on
  composition / induction); one pass does one hop.
- Anchor: hop count = pass count; cycle structure of π (orbits) visible.
- Cost: needs content-addressed attention over pairs; vocabulary n+2; T grows with n.

## 4. The odometer: the block as successor function  (doubtful)

Each pass adds 1 to a binary number held in the stream; looping counts. Carries visible.
- Dies probably: one pass computes increment with a prefix-AND (attention can average), so the loop
  is a clock, not a computation; unless the task is "add k" with k passes. Weak control.

## 5. The line as an attractor: denoising "to be or not to be"  (doubtful, charming)

Corrupt the line; a non-causal looped block repairs it pass by pass; the true line is a fixed point
with a basin of attraction; stability = spectral radius of the Jacobian at the fixed point < 1.
- Needs bidirectional attention (a change to PicoGPT). Loop load-bearing only for heavy corruption.
  The anchor (fixed point, Jacobian spectrum) is genuine but the "why one pass fails" is weak.

## 6. Relaxation: harmonic interpolation, attention as diffusion  (dies?)

Attention is a row-stochastic matrix, so a looped attention is a Markov chain; relaxation to a
harmonic function at rate set by the spectral gap. Beautiful physics, DEQ gradient (1 − Jᵀ)⁻¹.
- Dies: one pass of global attention can compute the answer directly (uniform average); the task has
  no reason to be local. Could be resurrected as a didactic chapter on the loop's spectrum.

## 7. Compass turns: state tracking in Z₄ or S₃  (lives as extension of 1)

`L R R L L …` → heading at each step: prefix product in a group. Z₂ is candidate 1; Z₄ is a
"rotation" (physics: phase); S₃ is non-abelian (word problem, the hard case in circuit
complexity). Transfer matrices literally. Probably a chapter, not the specimen.

## 8. Shidoku: 4×4 Sudoku  (doubtful: what the user has likely heard about)

The recent tiny recursive models (HRM, TRM) are famous for Sudoku. 4×4 has 288 solutions; constraint
propagation per pass; anchor = propagation depth computed by a solver. Needs non-causal attention,
2-D positions, a dataset, and probably >5k parameters. Too big for the specimen; maybe an epilogue.

## 9. The liar and the truth-teller  (prologue toy, not a specimen)

"this sentence is false" has no fixed point (a 2-cycle); "this sentence is true" has two. A looped
boolean map shows fixed points and limit cycles. Charm only.

## 10. Odd–even transposition sort  (dies)

n passes sort n items, but one pass of attention can sort by ranking (count smaller items).

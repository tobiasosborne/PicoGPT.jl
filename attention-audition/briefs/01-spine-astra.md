# Brief 01 — the derivation spine (Astra)

You are the mathematician on a small writing project. Your deliverable is a short, exact document,
`attention-audition/notes/spine.md`, that a prose writer will turn into a book chapter. You write
no prose for the book itself.

## Context

PicoGPT is a GPT written from scratch in Julia plus a book about it for BSc physics/maths students.
The repository root is your working directory. Read, in this order:

1. `BRIEF.md` (binding: §2 notation, §3 exact model semantics, §4 the specimen, §12 findings).
2. `book/chapters/05-attention.md` — the current attention chapter. It is correct but builds
   attention bottom-up (an average, then a score, then a mask). It stays as it is.
3. `book/chapters/03-energies.md`, `04-surprise.md`, `06-block.md` (what the reader already has, and
   what comes after: LayerNorm, MLP, the residual stream).
4. `book/notes/derivations.md` (the verified maths reference), and `src/attention.jl`, `src/model.jl`.
5. `/home/tobias/Projects/PicoGPT.jl/book/sources/attention-as-ansatz.md` — a distillation of a
   conversation in which the book's author finally understood attention, top-down, as an ansatz
   for the conditional P(next symbol | context). Read it closely. Its §2 (the five steps) is the
   new route of the chapter. Two remarks are marked **[added]**: they are not from the
   conversation and are unverified. (Read only; it lives outside this checkout. Do not copy it.)
6. Skim `/home/tobias/Projects/PicoGPT.jl/book/sources/attention-as-ansatz.transcript.md`, turns 6
   and 7 (the answer the author accepted, and their question about the enlarged sample space).

## What the spine must fix

Write it in the book's notation (BRIEF §2: reader $i$, sources $j\le i$, $\hat x_j$ the
LayerNorm'd column, $W_q,W_k,W_v,W_o$ with per-head row blocks $W_q^{(h)}$ etc., $A_{ij}$, $o_i$,
$d$, $\sqrt d$, $V$ for vocabulary size). For each step: **the exact statement; what is exact;
what is assumed; when the assumption holds; when it fails**. Short proofs where a proof is a few
lines; a precise reference to `derivations.md` where it already exists.

1. **Step 1, the uncertain index.** The identity, and the precise sense in which it is empty: the
   extension $\tilde P(y,j\mid c)$ is not determined by $P$ (any $r(j\mid y,c)$ works; the uniform
   one returns the original conditional in every term). State exactly what the first real
   assumption is (that an extension exists in which both factors are simple), and that nothing
   guarantees it. This answers the author's question in turn 7; get it airtight.
2. **Step 2, one shared small rule on vectors attached to positions.** What exactly the model's
   $\hat x_j$ is in the specimen ($L=1$: LayerNorm of embedding plus positional code) and in
   general (after earlier blocks it already carries gathered information). Where the reader's own
   column enters (the source set includes $j=i$; the residual stream also carries $x_i$ past the
   attention).
3. **Step 3, a Gibbs distribution over positions with a bilinear energy.** Gibbs form alone is no
   restriction (any positive distribution); the restriction is the energy. Include the biases
   exactly: expand $q_i\cdot k_j$ with $b_q,b_k$ and show which terms depend on $j$ (and so why
   $b_k$ has identically zero gradient, BRIEF §12). No built-in decay with distance; positions
   enter only through the positional code inside $\hat x_j$. The temperature $\sqrt d$ and the
   variance argument (at initialisation only). The causal mask as the statement that the index
   ranges over $j\le i$ — every position is a reader, and the $T$ rows give the $T$ conditionals of
   the chain rule in one pass (lower-triangular Jacobian).
4. **Step 4, low rank.** $W_q^{(h)\T}W_k^{(h)}$ is a $D\times D$ matrix of rank $\le d$, not
   symmetric; its two factors are all that "query" and "key" mean. Parameter count of the
   factored form against the full form, for the specimen.
5. **Step 5, mean field.** Replace $\sum_j A_{ij}\,g(\cdot,W_v\hat x_j)$ by $g(\cdot,\sum_jA_{ij}W_v\hat x_j)$.
   Exact when $g$ is affine in its second argument. State the size of the gap in general
   (second order in the spread of the selected vectors: a Jensen gap, with the Hessian) and the
   two regimes where it is small (vectors close together; one source holds nearly all the weight).
   Note the honest point that in a trained network the weights are fitted to the mean-field model,
   not to the mixture, so the mixture is a *different* model, not the "true" one.
6. **The [added] remark — verify it independently and fix its scope.** Claim: with a softmax
   readout whose logits are affine in $m$, the mean-field prediction is the normalised geometric
   mixture $\prod_jp_j(y)^{\pi_j}/\sum_a\prod_jp_j(a)^{\pi_j}$ of the per-source predictions. Prove or
   refute. Then state precisely what survives in the real model, where $o_i$ passes through $W_o$,
   the residual add, LayerNorm, the MLP, the final LayerNorm and $W_\text{out}$ before the softmax.
   (For instance: does it hold to first order in the spread? Is there any exact statement at all?)
   Write the conclusion as one sentence the chapter could use with its true scope, or say plainly
   that it should be left out.
7. **Heads.** With $H$ heads there are $H$ indices, one per head; say what their joint law is in
   the mixture picture (product over heads, given the context), how the mean field is taken head by
   head, and why the output is a direct sum $\R^D=\R^d\oplus\cdots$ recombined by $W_o$.
8. **What the steps leave out**, in one table (as in the distillation): restriction vs what it
   replaces.
9. **The MPS comparison and the one-particle picture** (distillation §5): which statements are
   exact and could stand in a short aside or closing section for a reader who may not know matrix
   product states; which are analogy only. Keep this brief; the quantum generalisations (§7) and
   the modern decoder (§6) are out of scope.

## The specimen numbers

A colleague (Sol) is computing, in parallel, on the trained specimen
(`artifacts/specimen.json`, `train.theta_final`, the 10 training windows, 80 readers), the
following, and will write `attention-audition/notes/numbers.md` and
`attention-audition/artifacts/mixture.json`:

- for every reader $i$ (each window, each position): the model's actual prediction $p_i$ (mean
  field); the **mixture** prediction, obtained by replacing head $h$'s output $o^{(h)}_i$ by the
  value $v^{(h)}_{j_h}$ of a single source, for every pair $(j_1,j_2)$, running the rest of the
  (column-wise) model on column $i$, and averaging the resulting distributions with weights
  $A^{(1)}_{ij_1}A^{(2)}_{ij_2}$; and the normalised **geometric** mixture of the same per-pair
  predictions with the same weights;
- KL divergences and total-variation distances between these three, top-1 agreement, the loss of
  each on the training targets, the entropy of each attention row and the weighted spread
  $\sum_jA_{ij}\lVert v_j-o_i\rVert^2$ per head; also the single-head variants (mix one head, keep
  the other at its mean).

In section 10 of your spine, say whether this experiment is well defined and measures what the
chapter would claim; if the definitions should change, say exactly how. If `numbers.md` already
exists when you finish, read it and audit it against your statements. Do not compute the numbers
yourself.

## Rules

- Write only `attention-audition/notes/spine.md`. Do not edit anything else; do not commit.
- Be exact and short: this is a reference, not an essay. Aim for 3–6 pages of markdown with LaTeX.
- Every claim either proved in a few lines, referenced to `derivations.md`, or marked as an
  analogy/interpretation.
- Final report (your last message): where the file is, the verdict on the [added] remark, any
  place where the distillation or the current chapter is wrong, and open doubts.

# DMRG and transformer growth

**Verdict — needs more.** The local truncation criterion is proved. Whether it produces a smaller executable PicoGPT at a useful error and total training cost remains open. [Derivation](dmrg.md)

| Question | Result |
|---|---|
| Which directions should survive a cut? | **Proved.** Minimize the quadratic teacher-to-student KL over stream projectors. A separable activation–Fisher moment gives weighted PCA and a discarded eigenvalue sum. With sample-dependent sensitivity, the activation and Fisher marginals can agree in two worlds with disjoint optimal projectors. |
| What does an area bound buy? | **Proved.** Conditional mutual information is the Bayes log-loss price of ignoring a refinement. It can be arbitrarily small while an exact PicoGPT predictor needs width at least the fine alphabet size. This bound concerns exact width. |
| When must the parent move? | **Proved.** Independent marginal and conditional fits on compatible data can be trained separately. A single-history, three-outcome flat-softmax split needs a change in the inherited bias to attain the joint optimum. A two-block transformer update has no general isometric two-site tensor to split by SVD. |
| Does staging save work? | **Proved.** For two separable Bernoulli fits, staging saves one factor evaluation/update at identical final logits against the specified dense gradient-descent baseline. A direct optimizer that skips the converged factor matches that work. **Open.** A whole-lineage saving for PicoGPT at matched final loss. |
| Is there a canonical stream? | **Proved.** Positive-epsilon LayerNorm preserves orthogonal changes on the centered plane up to reparameterization of its affine readers. Fixed positional differences restrict the global gauge. The separable local experiment has canonical weighted eigenspaces at a gap. Its canonical coordinates need not define a network gauge. |

**Checked numerically.** Julia verifies the finite-difference derivatives, KL curvature including cross-position terms in a causal attention block, discarded norm, gauge identities on the model, exact-width examples and optima. The smallest split examined has a frozen excess of 0.002238133166642009 nats. These are small algebraic checks. The strict work saving uses counted factor updates under the stated cost model. [Check script](check_dmrg.jl)

From the repository root, run

```sh
OPENBLAS_NUM_THREADS=1 julia -t1 --compile=min --project=. grow/notes/check_dmrg.jl
```

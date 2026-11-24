# Definitions and exact statements

**Verdict — promising.** Several growth operations preserve the parent predictor exactly at birth. Treating the parent as a quotient of an independently trained child requires a further condition on every fibre of the proposed coarse map. The [derivation](definition.md) gives that condition for stream dynamics, sequence laws and controlled returns.

| Proposed operation | Proved scope |
| --- | --- |
| Split vocabulary | Copy parent embeddings and readout rows, add log split weights to biases. Predictions sum and surprise decomposes. Arbitrary history-dependent split weights require representable child log odds. |
| Add depth or a loop pass | A zero gate on the increment is exact. In stock PicoGPT a fresh untied block can also have both residual outputs zero. Shared parameters retain gradients from earlier passes. |
| Add heads | A muted extra head works if the old head shapes are preserved. At fixed D a parent attention increment of affine rank three rules out the proposed lift with just one active head of width two. A sufficient score and value rank condition permits some parents to fit. |
| Add MLP units | Zero outgoing columns preserve the map. Incoming gradients start at zero; outgoing gradients can be nonzero immediately. |
| Duplicate residual coordinates | With copied positions, identical LayerNorm parameters across copies, and both queries and keys scaled by r⁻¹ᐟ⁴, the blocks and readout agree on the duplicated subspace. Averaging arbitrary child streams fails at LayerNorm. Stock width-dependent position frequencies can change even the sign of a logit. |
| Increase context | Every old prefix and its derivatives are preserved. Agreement with a sliding parent window requires the correct suffix states and position indices. |
| Block positions | Exact aggregation gives a log-sum-exp score and query-dependent value. The log-odds log cosh(q) refutes general closure under a single effective key per block. Replicated keys at fully visible, equal-size blocks give a restricted exact case. |

**Proved for worlds.** A pushforward path law always exists. Equality of fine-conditioned coarse predictions requires predictive sufficiency. For a Markov state partition this is the row-sum lumpability criterion. For a recursively updated history statistic the same condition applies to the history kernels. A three-state fine chain already has a projection with unbounded suffix memory. Under the corresponding joint reward/observation condition the return distribution is preserved for lifted policies. Total variation errors add along the tower. Conditional KL errors add across generated time steps by the path chain rule.

**Checked numerically.** From the repository root run the following [Julia script](check_definition.jl).

```sh
OPENBLAS_NUM_THREADS=1 julia -t1 -O0 --project=. grow/notes/check_definition.jl
```

| Float64 check | Observed result |
| --- | ---: |
| Largest absolute discrepancy among the tested identities | 4.55 × 10⁻¹⁵ |
| Largest absolute error over 2,036 coordinate finite differences | 2.69 × 10⁻⁹ |
| Width counterexample, parent first logit at position two | 0.99977957 |
| Same logit after duplication with stock positions | −0.45617464 |

**Open.** The counterexamples rule out the specified lifts. A different universal stock width reparameterization, or a reimplementation using all narrower heads, falls outside those proofs. The tower specifies no common rescaling operator, so it has no defined RG spectrum. Bounded predictive complexity and a training-cost advantage remain unproved.

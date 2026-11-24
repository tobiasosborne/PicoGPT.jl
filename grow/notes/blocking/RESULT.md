# Blocking positions in one causal attention head

**Proved.** Exact marginalisation preserves the read through two additive functions of the query, the block partition sum and its value-weighted numerator. For a block beside an untouched source, one affine score closes exactly when its projected keys coincide. The count supplies a bias. The quadratic correction is the within-block key covariance, and the linear value correction is the value–key covariance. The [derivation](../blocking.md) gives the simultaneous softmax closure criterion, uniform error bounds, iteration and mask treatment.

**Checked numerically.** The [Julia check](check.jl) reproduced all causal head reads to Euclidean error below $10^{-15}$ on four random settings and the saved trained specimen. For narrow random triples at perturbation scale 0.125, first- and second-degree RMS read errors were 0.00589 and 0.00102. On the specimen the same truncations gave 1.35 and 3.15. The bound holds throughout, but the second-degree approximation fails to improve accuracy on the specimen.

**Conjectured.** An adaptive grouping criterion could make truncation useful at trained query scales. The experiment leaves the attainable compression at a given downstream loss open.

**Verdict — needs more.**

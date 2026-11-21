# Derivations report

## Three findings for the book

1. **One ordinary block can do both jobs.** A four-feature PicoGPT block computes prefix parity with $X_{k+1}=B(X_k)+X_0$ and Rule 60 with $X_{k+1}=B(X_k)$. All words through four bits passed with the same weights for both tasks at that length. The three-bit model has 936 parameters, the four-bit model 2,160. This construction uses a finite ReLU lookup table. For a visible previous-token head, a second model uses **446 parameters**, the repository's sinusoidal positions and softmax, and a fixed one-register refresh between stock block calls. Its derived margin supports 64 data bits plus a neutral boundary token. Its $10^8$ embedding carrier makes it a reference for the computation rather than evidence about training.
2. **The light cone is visible to a bit flip, but need not appear in the gradient.** Each stock pre-LN block transports a constant feature shift at each token unchanged, so its full Jacobian has at least one unit eigenvector per token. The implicit inverse on the full stream is singular. In the 446-parameter construction, the refreshed state's Jacobian is zero on the Boolean plateaux. The signed-shift Jacobian belongs to another smooth realization of the same digital update. For the book, draw the actual VJP next to a single-bit intervention.
3. **Rule 60 has a finite, exact control at two stopping times.** Score the same fair bit word at passes 1 and 2 against the corresponding Rule 60 rows. A one-pass model with one readout and no horizon input has Bayes CE infimum $(T-1)\ln2/(2T)$ and accuracy ceiling $1-(T-1)/(4T)$. The recurrent reference can classify both targets perfectly. The pass count supplies the horizon. Adding it to a one-pass baseline changes the comparison. For ordinary prefix parity there is also a genuine asymptotic one-layer classification lower bound from average sensitivity, with no useful numerical cutoff for these small specimens. [Kozachinskiy, Steifer and Wałęga, Theorem 3 and Corollary 4](https://arxiv.org/html/2602.05896v2#S3)

## Proofs and checks

- Derived the shared-block, embedding, and shared-readout pullbacks for autonomous, injected, and deeply supervised loops. The stationary Neumann series requires spectral radius below one. A transient rollout contains ordered products of varying Jacobians.
- Verified the stock constant-shift adjoint identity and the vanishing refreshed-register VJP (maximum absolute entry below $2.8\times10^{-16}$). Exhausting all eight-bit words gave the predicted two-horizon disagreement fraction $7/16$.
- Derived all Rule 60 taps, their Lucas submask test, the full parity window at $2^m-1$, and the finite-width period $2^{\lceil\log_2T\rceil}$ of the update operator.
- Ran the actual Julia block on every word at $T=2,3,4$ for 32 passes in both ordinary modes. All decoded states agreed. The largest full-state deviation was $2.41\times10^{-9}$ or less. The transparent previous-token model agreed on all 64 six-bit words and on eight 64-bit words for 67 passes. Maximum state-register error was below $3.3\times10^{-14}$. A 256-bit enumeration checked every permissible pair of positions. The softmax bound on total misplaced weight is $4.3\times10^{-7}$.
- Constructed a stock tied pointer-doubling block with addresses stored in the stream and a stock permutation-chasing block. Each pays one preparation pass. All 16 four-bit strings passed the doubling test, including their internal addresses. All six three-label permutations and all starts passed eight hops, including unchanged memory. These explicit witnesses cost 27,980 and 16,398 parameters respectively.
- Derived a permutation-specific one-pass communication bound. On a disjoint bipartite encoding with $2n$ labels, the memory half must convey at least $\log_2(n!)$ bits under the paper's finite-summary assumption. The published two-function theorem and this reduction both allow tiny-domain shortcuts. [Peng, Narayanan and Papadimitriou, Theorem 1](https://openreview.net/pdf?id=KidynPuLNW)

## Limits

The finite interpolation tables grow with length. The compact previous-token construction uses a different outer recurrence from raw input addition, and its attention proof ends at 65 total columns. No constant-width, length-uniform pointer-doubling weights or small-length universal parity cutoff were obtained. ReLU piece counting applies to the restricted scalar counting control. None of the constructed weights demonstrates that optimization finds the same rule.

The [derivations](../notes/derivations.md) give the arrays, certificates, candidate-by-candidate audit and measurements. The [Julia constructors](../scratch/astra/handbuilt.jl) return weights in PicoGPT's flat layout. Run the [verification script](../scratch/astra/check.jl) from the root with the following command.

~~~sh
julia --project loop/scratch/astra/check.jl
~~~

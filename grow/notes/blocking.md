# Attention under blocking of positions

## Exact decimation of a source block

**Proved (conditional expectation).** Fix a head of width $d=D/H$, a query at position $i$ in one sequence, and $x=q_i/\sqrt d$. In the row convention of [the implementation](../../src/attention.jl), $A_{ij}=e^{x^\top k_j}/Z$ for $j\le i$. Partition the visible positions into nonempty blocks. For a block $B$, define
$$
Z_B(x)=\sum_{j\in B}e^{x^\top k_j},\qquad N_B(x)=\sum_{j\in B}e^{x^\top k_j}v_j.
$$
The coarse score, probability and read are
$$
s_B=\log Z_B,\qquad p_B=\exp(s_B)/\sum_C\exp(s_C),\qquad v_B=N_B/Z_B.
$$
Indeed $p_B=\sum_{j\in B}A_{ij}$ and $p_B v_B=\sum_{j\in B}A_{ij}v_j$. Summing over blocks preserves the head output at this query. The pair $(Z_B,N_B)$ is uniquely determined if the replacement must preserve mass and weighted value against arbitrary untouched neighbours. Equality of one final read alone is weaker, e.g. if every value is the same then any scores give that read.

**Proved (one block against an unchanged exterior).** Ask for closure at every query in a linear space $U$ (the whole head by default), and project keys onto $U$. Write $m=|B|$. A score $b+x^\top k_*$ is exact iff all projected keys in $B$ coincide and $b=\log m$. In that case the block value is their arithmetic mean of values. To see necessity, evaluate at zero for the count, then take the second derivative along any $u\in U$ at zero. Differentiating the finite sum gives $\partial_u^2\log Z_B=\operatorname{Var}_{B,x}(u^\top k)$ under the within-block Gibbs weights. At zero the variance vanishes in every direction iff the projected keys coincide. Substitution proves sufficiency. A bare dot product against untouched singleton scores therefore works only for $m=1$. A constant query coordinate or an explicit log-count bias implements the required offset.

**Proved (fixed value alone).** Group sources by distinct projected keys $a$ with multiplicity $n_a$ and mean value $w_a$. A constant $w$ reproduces $v_B(x)$ iff every $w_a=w$, since the condition is $\sum_a n_a(w_a-w)e^{x^\top a}=0$ identically. Restrict to a line with pairwise distinct slopes $u^\top a$. Taking successive derivatives at zero gives an invertible Vandermonde system componentwise. In particular repeated keys may carry different individual values.

**Proved (simultaneous closure modulo a row shift).** For a fixed partition the observable mass ratios admit one affine score per block iff their *normalised projected key distributions are translates of one common distribution*. For proof, choose a reference block $C$. Equality of probabilities is equivalent to $Z_B/Z_C=e^{b_B-b_C+x^\top(a_B-a_C)}$. At zero this fixes $e^{b_B-b_C}=|B|/|C|$. After removing counts, the right side translates the reference empirical distribution by $a_B-a_C$, by the exponential independence just proved. Conversely translates supply exactly this ratio. Bare scores additionally require equal block sizes. For fixed values one must also apply the preceding conditional-mean criterion in each block. This includes cancellation of a common nonlinear score even with dispersed keys. These are all-query statements for a fixed source configuration. On a finite list of observed queries, affine agreement is only a finite interpolation condition.

## Cumulants and controlled truncation

**Proved (local expansion).** Let $E$ be uniform averaging within $B$, $\mu=Ek$, $w=Ev$, $h=k-\mu$, $g=v-w$, $Y=x^\top h$ and $C=E[hh^\top]$. The joint generator $L(x,t)=\log E\exp(x^\top k+t^\top v)$ gives $s_B=\log m+L(x,0)$ and $v_B=\nabla_t L(x,0)$. Expanding at zero yields
$$
\begin{aligned}
s_B^{(1)} &= \log m+x^\top\mu, &
s_B^{(2)} &= s_B^{(1)}+\tfrac12 x^\top C x,\\
v_B^{(1)} &= w+E[gY], &
v_B^{(2)} &= v_B^{(1)}+\tfrac12 E[gY^2].
\end{aligned}
$$
The neglected terms after degree two begin at degree three in $x$ for fixed keys and values. This follows by expanding $\log E e^Y$ and $E[g e^Y]/E[e^Y]$, using $Eh=Eg=0$. In general the order-$n$ tensors are the key cumulant of order $n$ for the score and the mixed cumulant with one value and $n$ keys for the read, each contracted with $x^{\otimes n}/n!$. This is a local Taylor series since the generating function is analytic near zero.

**Proved (generated couplings).** In the original query units the generated score is $q^\top C q/(2d)$. A symmetric quadratic score per block absorbs it, or equivalently one may lift the query to include its symmetric pair products. Already at first order the value requires a block-specific matrix $E[gh^\top]$ acting on $x$. Matching the score alone can leave a leading linear read error. A pair of equally weighted subkeys $\mu\pm a$ sharing the block's total count generates $\log\cosh(x^\top a)$ in the score, whose quadratic tensor is $aa^\top$. Two subkeys with arbitrary positive weights still have covariance of rank at most one. Independent ordinary heads have separate normalisations. Adding such a head supplies no within-block log-partition operation by itself.

**Proved (uniform first-order error bounds).** Write $W=\max_B Y-\min_B Y\le \|q\|\operatorname{diam}_B(k)/\sqrt d$ and $D_v=\max_{j,l\in B}\|v_j-v_l\|_2$. Then for all real queries
$$
0\le s_B-s_B^{(1)}\le W^2/8,\qquad
\|v_B-w\|\le D_v W/4,\qquad
\|v_B-v_B^{(1)}\|\le D_v W^2/8 .
$$
For proof tilt the uniform law by $e^{tY}$ for $0\le t\le1$. The second derivative of the log-partition function along this path is $\operatorname{Var}_tY\le W^2/4$ (bound the second moment about the range midpoint). Integrating with weight $1-t$ gives the score bound. The first derivative of $E_t v$ is $\operatorname{Cov}_t(v,Y)$, bounded in norm by $D_v W/4$ by applying Cauchy–Schwarz to each unit projection. Its second derivative is $E_t[(v-E_t v)(Y-E_t Y)^2]$, of norm at most $D_v\operatorname{Var}_tY$. Integration gives the other two inequalities.

**Proved (error in the head output).** Let $W_*$ be the maximum block width $W$ in the visible row, and let $D_{v,\mathrm{row}}$ be the diameter of all visible values. With first-order scores and values, the Euclidean read error is at most
$$
D_{v,\mathrm{row}} W_*^2/32+\max_B D_{v,B} W_B^2/8 .
$$
Hold exact conditional values fixed while interpolating from approximate to exact scores. The read derivative is their covariance with the block score errors. The conditional values lie in the original convex hull and the errors range over an interval of length at most $W_*^2/8$. The same projection bound gives the first term. At the approximate probabilities, replacing the values costs at most the worst block value error. Using frozen mean values replaces the second term by $\max_B D_{v,B}W_B/4$.

## Repeated blocking and causality

**Proved (flow).** For a parent $P$ with disjoint children, $(Z_P,N_P)=\sum_{B\subset P}(Z_B,N_B)$, where the sum runs over its immediate children. Thus every tree over the same leaves gives the same answer. Equivalently carry the counting measure $\sum_j\delta_{k_j}$ and its value-weighted companion $\sum_j v_j\delta_{k_j}$; the pair above consists of their Laplace transforms. This representation can grow with the number of distinct keys. With $\rho_B=|B|/|P|$, the first two key couplings obey
$$
\mu_P=\sum_B\rho_B\mu_B,\qquad
C_P=\sum_B\rho_B\left[C_B+(\mu_B-\mu_P)(\mu_B-\mu_P)^\top\right].
$$
Expand $k-\mu_P=(k-\mu_B)+(\mu_B-\mu_P)$ under each child law to prove the recursion. Even point-mass children generate a variance when their means differ. More generally raw key moments and moments weighted by one value add before normalisation. Keeping these up to degree $n$ therefore gives exactly the parent's Taylor coefficients through degree $n$ after any number of merges.

**Proved (fixed profiles and scaling).** Divide the pair by block size. Merging identical normalised pairs fixes their profile at every query, while an equal-arity merge of $b$ such children adds $\log b$ to the score. Every empirical profile can be replicated, so blocking alone selects no universal one. For nested blocks of $m$ i.i.d. key/value pairs, at each fixed $x$ satisfying $E[(1+\|v\|)e^{x^\top k}]<\infty$, the strong law gives almost surely
$$
s_B(x)=\log m+\log E e^{x^\top k}+o(1),\qquad
v_B(x)\longrightarrow \frac{E[v e^{x^\top k}]}{E e^{x^\top k}} .
$$
Apply the law separately to $Z_B/m$ and $N_B/m$ and use positivity of the limit denominator. The limiting profile retains the population cumulants wherever the generator exists near zero. A Gaussian limit for sums of independent keys concerns a different operation, since blocking here mixes their empirical measures.

**Proved (mask).** At row $i$ use $B_i=B\cap\{1,\ldots,i\}$. An empty intersection has zero mass and numerator and can be skipped; a partial block needs its own prefix count, moments or exact pair. For contiguous blocks only the current block can be partial. A prefix decomposes into completed subtrees and visible leaves, so the addition rule implements the mask at every level. Using an entire diagonal block would give unseen positions positive weight. At a query on the last site of a block, all visible blocks are complete. Keeping only those queries preserves the corresponding head reads on the original keys and values. Coarsening the queries or propagating this equality through further transformer layers would require additional identities.

## Connections

**Proved (tree factorisation).** The probability of reaching a leaf factors into conditional child masses $Z_{\rm child}/Z_{\rm parent}$ along its path, by telescoping. This gives an exact reference for a multiscale attention tree. Zhu and Soricut construct hierarchical attention using a hierarchy of low-rank matrix blocks. Here the exact conditional child masses can still require examining all leaves for each new query. [H-Transformer-1D](https://aclanthology.org/2021.acl-long.294/)

**Proved (kernel features).** Given a separable approximation $e^{x^\top k}\approx\phi(x)^\top\psi(k)$, substitute it in both sums. The block summaries become $\sum_j\psi(k_j)$ and $\sum_j v_j\psi(k_j)^\top$, which add exactly for the chosen kernel. The ratio incurs the kernel approximation error. Performer constructs positive random features for this softmax kernel. [Choromanski et al., *Rethinking Attention with Performers*](https://arxiv.org/abs/2009.14794)

**Proved (source moments).** Factor $e^{x^\top\mu}$ from $Z_B,N_B$ and expand $e^{x^\top(k-\mu)}$. The degree-$n$ coefficients are centred source moments with charges $1$ and $v_j$, divided by $n!$. These two moment expansions illustrate the multipole idea directly. There is even a Gaussian-kernel form $e^{x^\top k}=e^{\|x\|^2/2}e^{-\|x-k\|^2/2}e^{\|k\|^2/2}$. The common target factor cancels in the read. Greengard and Strain group sources for the Gaussian transform using local expansions. Contiguity in sequence position gives no upper bound on source spread in key space, as adjacent keys can be arbitrarily far apart in the head model. [Greengard and Strain, *The Fast Gauss Transform*](https://doi.org/10.1137/0912004)

## Numerical check

**Checked numerically (Julia, Float64).** The [script](blocking/check.jl) runs from the repository root with `OPENBLAS_NUM_THREADS=1 julia -t1 --project grow/notes/blocking/check.jl`. It groups contiguous triples, intersects each with each query's causal prefix, and compares the resulting head output with PicoGPT's own attention and mix. RMS means $\sqrt{\operatorname{mean}_i\|r_i^{(n)}-r_i\|_2^2}$ over all head rows, before the output projection. Each order truncates *both* score and value at that degree. The width shown is the largest within-block score range in the sample.

| Sample | Head rows | Max exact read error | RMS, degree one | RMS, degree two | Max width |
|---|---:|---:|---:|---:|---:|
| Random, $\sigma=0.125$ | 768 | $7.81\times10^{-16}$ | 0.00589 | 0.00102 | 0.958 |
| Random, $\sigma=0.25$ | 768 | $6.96\times10^{-16}$ | 0.0228 | 0.00796 | 1.92 |
| Random, $\sigma=0.5$ | 768 | $9.09\times10^{-16}$ | 0.0851 | 0.0598 | 3.83 |
| Random, $\sigma=1$ | 768 | $9.22\times10^{-16}$ | 0.296 | 0.537 | 7.66 |
| Trained specimen | 160 | $7.92\times10^{-16}$ | 1.35 | 3.15 | 12.9 |

**Checked numerically (sampling and interpretation).** Each random sample uses the same 64 draws at $d=4$, $T=12$, with independent standard Gaussian query, value, block centre and key perturbation coordinates. The key equals the block centre plus $\sigma$ times its perturbation. The trained run uses both heads and every training-window prefix of the saved step-200 specimen. The script asserts the first-order bounds for every row and checks the Taylor coefficients by centred finite differences. Degree two improved the read in RMS at the three smallest random spreads. It increased the error at the largest random spread and on the trained specimen. The specimen comparison uses head-output units and measures no language-model loss.

**Conjectured (open).** Choosing which source blocks to compress from their projected spread over the queries that actually visit them could make a low-order truncation useful in a trained model. The remaining test is whether the head errors and downstream loss stay small at a meaningful reduction in stored or evaluated keys.

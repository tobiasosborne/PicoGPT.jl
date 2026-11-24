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

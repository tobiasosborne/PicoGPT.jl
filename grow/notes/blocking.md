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

**Proved (error in the head output).** Let $W_*$ be the maximum block width $W$ in the visible row, and $D$ here denote the diameter of *all visible values* (in this bound only). With first-order scores and values, the Euclidean read error is at most
$$
D W_*^2/32+\max_B D_{v,B} W_B^2/8 .
$$
Hold exact conditional values fixed while interpolating from approximate to exact scores. The read derivative is their covariance with the block score errors. The conditional values lie in the original convex hull and the errors range over an interval of length at most $W_*^2/8$. The same projection bound gives the first term. At the approximate probabilities, replacing the values costs at most the worst block value error. Using frozen mean values replaces the second term by $\max_B D_{v,B}W_B/4$.

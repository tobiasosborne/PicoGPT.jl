# One block, many passes

A fixed point of the decoded bits, a fixed point of the real residual stream, and the derivative at that point are three different measurements. This matters even on four bits. The ordinary PicoGPT block below can compute prefix parity and Rule 60 with the same weights. Its full derivative still has eigenvalue one.

**Evidence convention.** “Proved” refers to an algebraic result under the hypotheses given. “Checked” gives the finite domain and numerical test. The finite-table constructions have an algebraic interpolation proof and a numerically checked separation condition. None of the weight sets below is a training result. All arrays use column tokens, row-query attention, and $\varepsilon=10^{-5}$.

## 1. A tied block and its pullback

**Proved.** Write $U=X_0=\operatorname{embed}(x,E)$, $p$ for the parameters of the single block, $r=(W_{\rm out},b_{\rm out},\gamma_f,\beta_f)$ for the shared readout, and $B(X)=X+f_p(X)$. In the derivative, a single stream index $i$ ranges over feature and token. The batch dimension contributes independent sequences. Define $X_{k+1}=G(X_k,U,p)$ and $Z_k=R_r(X_k)$. Four choices of $G$ clarify which state the loop remembers.

| Loop | Matrix update | Component update |
|---|---|---|
| Autonomous | $B(X)$ | $X_i+f_i(X)$ |
| Add input after the block | $B(X)+U$ | $X_i+f_i(X)+U_i$ |
| Add input before the block | $B(X+U)$ | $X_i+U_i+f_i(X+U)$ |
| Refresh a state register | $U+C[B(X)-X]$ | $U_i+\sum_j C_{ij} f_j(X)$ |

The second rule gives an ordinary residual (Euler) step with constant forcing $U$. It needs one addition and matches the raw-addition witness below. In the third rule the same drift enters before the two normalized residual branches. At fixed finite weights the branch updates are bounded per column because LN has bounded output, attention takes convex combinations, and the MLP receives bounded inputs. All three residual variants therefore grow at most linearly in $k$. With zero branches the autonomous stream stays at $U$, and either additive loop gives $(k+1)U$. Cancellation can give a fixed point for particular weights and inputs. The fourth rule reanchors the stream each pass and is bounded independently of $k$. In §2, $C$ copies just feature 7 into the same feature of each column. It introduces no learned parameter, but changes the recurrence.

Let $w_K=1$ for last-pass loss (all other $w_k=0$), or $w_k=1/K$ for $1\le k\le K$ for deep supervision. Take $w_0=0$ and $\mathcal L=\sum_{k=1}^K w_k\ell(R_r(X_k),y_k)$, allowing different targets at different passes. Set $g_k=w_k R_{X,k}^{\mathsf T}\nabla_Z\ell_k$ and abbreviate $J_k=G_{X,k}$, $H_k=G_{U,k}$, $D_k=G_{p,k}$ with the other arguments held fixed.

| Quantity | Matrix pullback | Component pullback |
|---|---|---|
| State | $\bar X_K=g_K,\quad\bar X_k=g_k+J_k^{\mathsf T}\bar X_{k+1}$ | $\bar X_{k,i}=g_{k,i}+\sum_j(\partial G_j/\partial X_i)\bar X_{k+1,j}$ |
| Block | $\bar p=\sum_{k=0}^{K-1}D_k^{\mathsf T}\bar X_{k+1}$ | $\bar p_a=\sum_{k,j}(\partial G_j/\partial p_a)\bar X_{k+1,j}$ |
| Input | $\bar U=\bar X_0+\sum_k H_k^{\mathsf T}\bar X_{k+1}$ | $\bar E_{ia}=\sum_{n:x_n=a}\bar U_{in}$ |
| Readout | $\bar r=\sum_{k=1}^K w_k R_{r,k}^{\mathsf T}\nabla_Z\ell_k$ | $\bar r_a=\sum_{k,j}w_k(\partial R_j/\partial r_a)(\partial\ell_k/\partial Z_j)$ |

For the first two variants $J=B_X$, $D=B_p$ and $H=0,I$ respectively. For pre-addition $J=H=B_X(X+U)$ and $D=B_p(X+U)$. For the refreshed register $J=C(B_X-I),\, H=I,\, D=CB_p$. Use one saved block pullback per pass. Sum each named parameter's cotangent and flatten **once** in the repository's order. An independent parameter vector per pass changes the model.

**Fixed-point derivative, proved with conditions.** If $X^*=G(X^*,a)$ for independent data or parameters $a$, $G$ is differentiable locally (for the usual implicit function theorem, continuously differentiable near the point), and $I-J_*$ is invertible, then
\[
(I-J_*^{\mathsf T})\lambda=\nabla_X\ell,\qquad
\nabla_a\mathcal L=\partial_a\ell+G_a^{\mathsf T}\lambda.
\]
In components $\lambda_i=\ell_{X_i}+\sum_j(\partial G_j/\partial X_i)\lambda_j$. A ReLU kink requires a separate directional analysis. In finite dimension the operator series $\sum_{h\ge0}(J_*^{\mathsf T})^h$ converges exactly when $\rho(J_*)<1$. Invertibility alone suffices for the implicit derivative. If $J_*^m=0$, the inverse is the sum through $h=m-1$. At the stationary linearization, $K$ passes with final loss give the first $K$ terms for a repeated parameter forcing, plus the initial-state term $(J_*^{\mathsf T})^K g$ if differentiating the initialization. Along a transient trajectory the factors are ordered products of **different** Jacobians and parameter derivatives. Convergence of the forward values alone does not justify taking a gradient limit. The constant-feature symmetry in §5 rules out the full inverse for the first three variants.

## 2. Prefix parity

**Proved digital recurrence.** Let $S_{tj}=[t=j+1]$ be the left-neighbour shift with a zero boundary. Write $\oplus$ for addition in $\mathbb F_2$. With $s^{(0)}=0$,
\[
s^{(k+1)}=x\oplus S s^{(k)},\qquad
s_t^{(k)}=\bigoplus_{h=0}^{\min(k,t)-1}x_{t-h}.
\]
The empty XOR is zero. Thus every $t\le k$ is correct and stays correct. With $s^{(0)}=x$ replace $k$ by $k+1$ in the window length. The unique digital fixed point is $(I-S)^{-1}x$ over $\mathbb F_2$ since $S^T=0$ (here the exponent $T$ denotes sequence length). For independent fair bits the expected correctness outside the window is exactly $1/2$. One string can agree by chance beyond the front.

### A four-feature witness for the ordinary post-addition loop

**Construction and finite certificate.** There is no special mask, sentinel, extra input port, or change to the embeddings. The same $\theta$ also runs §3 autonomously. This is a finite-length witness. It uses a uniform head and a table encoded in the ReLU slopes. Use the second witness below for a head that actually selects the previous token.

For bits $x,s\in\{0,1\}$ let $e_i$ be a coordinate vector. Define
\[
 U_t(x)=P_t+3xe_3+3(2x-1)e_4,\qquad C_t(s)=P_t+12(2s-1)e_4.
\]
Initially the bit is $x_t$, so the front starts at one. In either loop the first $B$ produces $C_t(x_t\oplus x_{t-1})$. At subsequent calls the autonomous input is $C(s)$ and the post-additive input is $C(s)+U(x)$. Prescribe
\[
 B(C(s))_t=C_t(s_t\oplus s_{t-1}),\qquad
 B(C(s)+U(x))_t=C_t(x_t\oplus s_{t-1}).
\]
A sign readout $Z_{2t}-Z_{1t}=(e_4-e_2)^\mathsf{T}\operatorname{LN}_f(X_t)$ decodes all three stream types. The bit amplitude exceeds the largest competing positional contrast (at most 2 for $U,C$ and 4 for their sum).

**Explicit arrays.** Set $D=4,H=L=1,V=2$. Everywhere in these constructions unspecified arrays or entries are zero. All three LN gains are ones, all LN offsets zero. The columns of $E$ are $U_t(x)-P_t$. Take $W_v=e_1(1,-0.7,0.3,0.6)$, $W_o=e_1e_1^{\mathsf T}$, $W_q=W_k=0$ and $W_{\rm out}[2,:]=(e_4-e_2)^{\mathsf T}$. The following finite sums specify every MLP entry, including its biases, without fitting any parameter.

1. For each prefix length $t=1,\ldots,T$ and each of its $2^t$ bit strings, include calls on $U(x)$ and $C(x)$ with last-column target $c=C_t(x_t\oplus x_{t-1})$. Also include calls on $C(s)+U(x)$, starting at $s=x\oplus Sx$ and updating $s\leftarrow x\oplus Ss$ for $t$ calls. For each such call $c=C_t(x_t\oplus s_{t-1})$ uses the incoming $s$. All boundary bits are zero.
2. For each call the last mid-block column is $y=X_t+e_1\sum_{j\le t}(1,-0.7,0.3,0.6)\operatorname{LN}_1(X_j)/t$. Set $h=v^\mathsf T\operatorname{LN}_2(y)$, $v_i=\sin(i^2+1)$ and $d=c-y$. Remove duplicate identical entries and sort as $(h_i,d_i)$. The finite check verifies that equal keys demand the same correction.
3. Put $b_2=d_1$. For each consecutive pair with $\delta=h_{i+1}-h_i$, use two rows $W_1=v^{\mathsf T}$, with biases $-(h_i+\delta/4),-(h_{i+1}-\delta/4)$. Their $W_2$ columns are $+2(d_{i+1}-d_i)/\delta$ and its negative. The pair contributes zero below the first knot, interpolates linearly, and contributes $d_{i+1}-d_i$ above the second. The corrections telescope at every key, so the residual gives $y+d=c$. Induction covers all passes in exact arithmetic, conditional on the finite separations.

| Maximum $T$ | $F$ | Parameters | Minimum distinct key gap, Float64 | Largest stream error over 32 passes, both loops |
|---:|---:|---:|---:|---:|
| 2 | 34 | 432 | 0.02049 | $1.1\,10^{-11}$ |
| 3 | 90 | 936 | 0.0002672 | $6.0\,10^{-10}$ |
| 4 | 226 | 2160 | 0.0001834 | $2.5\,10^{-9}$ |

**Checked.** All words at each listed length, every output column, both raw and autonomous modes, and the full predicted residual against its prescribed value. For $T=4$ the largest weight magnitude is about $2.18\,10^5$. The table size increases with length and gives no efficient extrapolation claim. A shorter prefix is already a row in the table. The additive stream becomes the fixed $U(x)+C(s^*)$.

### A 446-parameter previous-token head for 64 data bits

**Construction.** Here $D=8,H=L=1,F=4,V=2$. Prepend one neutral token (id 1) for the zero boundary. Data ids are $1+x_t$. The single change outside the stock block is the register refresh $G(X,U)=U+e_7e_7^\mathsf T[B(X)-X]$. Thus the block receives exactly $X_j=P_j+A e_5+100x_j e_6+100 s_j e_7$, with $A=10^8$ and $s^{(0)}=0$. The sentinel always reads itself and remains zero. Keeping the workspace out of the next source makes the full register state stationary as soon as all bits settle.

**Explicit arrays.** Let $\sigma_0=A\sqrt7/8$, let row $i$ of $C_4$ be $(e_i-e_8)^\mathsf T$ for $i=1,\ldots,4$, and let $R$ be block diagonal, with blocks $\left(\begin{smallmatrix}\cos w&-\sin w\\ \sin w&\cos w\end{smallmatrix}\right)$ at $w=1,0.1$. With $e=(1,1,1,1)^\mathsf T$ the nonzero entries are
\[
\begin{aligned}
 E[:,1+x]&=Ae_5+100xe_6,\\
 W_k[1:4,:]&=\sigma_0 C_4,& b_k[1:4]&=e,\\
 W_q[1:4,:]&=1024\sqrt8\,R\sigma_0 C_4,& b_q[1:4]&=1024\sqrt8\,Re,\\
 W_v[1,:]&=\sigma_0(e_7-e_8)^\mathsf T,& W_o&=e_1e_1^\mathsf T,\\
 W_1[i,:]&=(\sigma_0/100)(e_6+e_1-2e_8)^\mathsf T,&
 b_1&=-(1,3,5,7)^\mathsf T/4,\\
 W_2[7,:]&=(200,-200,-200,200),\\
 W_{\rm out}[2,:]&=(\sigma_0/100)(e_7-e_8)^\mathsf T,& b_{\rm out}[2]&=-1/2 .
\end{aligned}
\]
The key bias could be zeroed with identical attention weights since it shifts each score row by a constant. Its written value makes the trigonometric proof easier.

**Score margin and LN.** Centering cancels exactly in every contrast. If $\sigma$ is the actual LN scale, the triangle inequality applied to the augmented vector $(\Pi X/\sqrt8,\sqrt\varepsilon)$ gives $|\sigma-\sigma_0|<51$ at the first LN, since $\|P+100xe_6+100se_7\|\le2+100\sqrt2$. Therefore $|\sigma_0/\sigma-1|<1.6\,10^{-6}$. At unit gain ratio the four addressing features are $v_j=P_{1:4,j}+(1-P_{8j})e$. The score before sharpness scaling is $(Rv_i)^\mathsf T v_j$. The rotation asks for position $i-1$ using the **stock** two fast sinusoidal pairs. In the actual LN the feature error has norm below $6.4\,10^{-6}$, since each raw contrast has magnitude at most 2. For $i\le65$, $\|v_i\|<1.42$, hence each unscaled pair score changes by less than $1.9\,10^{-5}$.

**Checked finite trigonometric minimum.** Over every allowed wrong column in every row $2,\ldots,T$, the minimum preferred-minus-runner-up gap at unit gain ratio is 0.2134538759528262 at $T=64$ and 0.0184494569067582 at $T=65$, from 256-bit evaluation of the explicit pairs. This leaves a conservative actual gap $>0.0184$ through 64 data positions and the sentinel. The total wrong softmax mass is bounded by $64\exp(-1024\cdot0.0184)<4.3\,10^{-7}$. A single score scale for unbounded $T$ does not follow. At the sentinel the only causal weight is one.

**Boolean gate, proved given the gap.** Each value differs from $100s_j$ by at most $2.001$ from the gain and $P_{7j}-P_{8j}$. Its mixture $h$ differs from $100s_{i-1}$ by less than $2.01$. Attention adds $he_1$. At LN₂ the input perturbation off $Ae_5$ has norm at most $2+100\sqrt2+103$, so the gain ratio differs from one by less than $3\,10^{-6}$. The common MLP scalar $m=(\sigma_0/\sigma_2)(100x_i+h+P_{6i}+P_{1i}-2P_{8i})/100$ is within 0.061 of $x_i+s_{i-1}$. The four hinges write exactly $0,100,0$ on the three corresponding plateaux. Refreshing the register gives the stated Boolean recurrence. The margin in the final LN readout also exceeds the positional error. The large carrier and sharp scores make this an existence witness. Trainability at these scales has not been established.

**Checked.** The stock Julia block with this outer update agrees at every pass on all 64 six-bit words, plus zero, ones, alternating, and five seeded random 64-bit words. The longest test runs for 67 passes. The maximum absolute register error against $100s$ is below $3.3\,10^{-14}$. A further update at the fixed point changes the stream by less than $10^{-9}$.

### What a one-pass control can rule out

**Proved restricted counting bound.** Fix a length $t$. Suppose the only count-dependent input to a one-hidden-layer ReLU logit difference is a scalar $u=m/t$ in an affine line, where $m$ is the number of ones. Then $g(u)=a+bu+\sum_{i=1}^F c_i[d_i u+e_i]_+$ has at most $F+1$ affine pieces. Strict alternation on $m=0,\ldots,t$ requires $t$ sign changes, hence $F\ge t-1$. One way to count is to perturb the bias infinitesimally while preserving all the strict sample signs and avoiding roots at knots or on flat pieces. Each distinct root then costs an affine piece. If the last bit is fixed and only $m=0,\ldots,t-1$ vary the bound becomes $F\ge t-2$. A full triangular tooth contains two sign changes. For a fixed $t$ a spline can interpolate the alternating labels. An unknown $t$ also requires a way to distinguish lengths. Uniform averaging alone can give identical fractions at different lengths with different parities.

**Scope.** Position-dependent values, nonuniform softmax scores, multiple heads, residual features and both subsequent LayerNorms invalidate the scalar affine-spline hypothesis in general. The four-feature witness above actually exploits position-dependent normalized values under uniform attention. There is no resulting universal classification cutoff at $F+2$ for PicoGPT.

**Proved one-pass bound for the actual block, in margin.** Keep all weights and widths fixed. Flip one bit at $i<t$ and evaluate at $t$. The receiving query stays fixed. Values have uniformly bounded norm and the scaled dot-product scores lie in $[-M,M]$ for a constant $M$ because LN₁ is bounded. The old and new weight of the changed key are each at most $e^{2M}/t$. Conditional on choosing any other key, the value mixture is unchanged. The difference of the two unconditioned mixtures is consequently at most $4\max_j\|v_j\|e^{2M}/t$ per head. The subsequent residual, MLP, readout and LNs at $t$ have a length-independent Lipschitz bound ($\varepsilon>0$). Thus the change in the binary logit difference is at most $C(\theta)/t$. Required labels flip. To realize margin $\eta>0$ on both members of every pair requires $t\le C/(2\eta)$. This permits correct argmax with vanishing margin. For fair bits, pairing such strings also gives an expected CE lower bound $\log(1+\exp(-C/(2t)))$ by convexity. The analogous soft-attention sensitivity obstruction and its distinction from a classification lower bound appear in Hahn, Lemma 5. [Hahn, 2020](https://doi.org/10.1162/tacl_a_00306)

**Proved exact classification obstruction at growing lengths.** Kozachinskiy, Steifer and Wałęga (v2, Theorem 3 and Corollary 4) bound the average number of single-bit flips changing a one-layer classifier by $O(\sqrt n(\log n)^{O(1)})$. Parity changes on all $n$ flips at every word. Their proof transfers to PicoGPT's pre-LN and final LN as follows. Fix the last query bit. Each earlier position has just two possible local normalized embeddings. Its unnormalized softmax weight and weighted value coordinates thus have two possible constant values each, hence are affine functions of that one bit. Each head output is a ratio of affine sums, with a positive denominator. All remaining tests at the query (including $z=\sqrt{r+\varepsilon}$ specified by $z>0,\ z^2=r+\varepsilon$, ReLU, and comparing logits) form a fixed-size semialgebraic formula in these sums. This is exactly the hypothesis used between Lemma 2 and Theorem 3. With $n=t-1$ free bits, a fixed-size one-pass PicoGPT cannot classify all prefixes for all sufficiently large $t$, even allowing vanishing margins. This proof gives no practical cutoff at $t=8$ or $64$. A $\mathsf{TC}^0$ upper bound on its own would be insufficient because threshold circuits can compute parity by testing each possible odd count. Hsu's separate heads-times-degree bound assumes rational postprocessing, and its ReLU extension has a margin hypothesis. [Kozachinskiy, Steifer and Wałęga, §3, v2](https://arxiv.org/html/2602.05896v2#S3) [Hsu, 2026](https://arxiv.org/abs/2605.12171)

### Shared-weight pointer doubling

**Proved algorithm.** Store a bit $s_t$ and an address $a_t$. Initialize $s=x$, $a_t=t-1$, with $a_0=s_0=0$. In parallel use the same instruction every round,
\[
(s'_t,a'_t)=(s_t\oplus s_{a_t},\,a_{a_t}),\qquad
s'=s\oplus M(a)s,\quad M(a)_{tj}=[a_t=j],\quad a'=M(a)a .
\]
Here matrices omit column zero, so a missing neighbour contributes zero. After $h$ rounds $a_t=\max(0,t-2^h)$ and $s_t$ is the parity of the last $\min(t,2^h)$ bits. The address and bit must be read from the **old** state. There is no external pass counter.

**Stock-block existence at fixed maximum length.** The address-register arrays in §4 implement exactly this program on a bounded domain, with one tied preparation pass to install the initial registers from the ordinary embedding. For each larger finite $T$ one can widen the address registers, increase the score gap to $\log(4T)$ or greater, and compile the finite MLP as below. This gives a family of tied blocks with $1+\lceil\log_2 T\rceil$ passes. The width and the MLP table can grow with $T$. A single small fixed $\theta$ working at arbitrarily large $T$ is open here. Indeed any fixed head after pre-LN has bounded scores and cannot put a fixed positive mass on one prescribed key for every length.

**Checked.** All 16 four-bit words, including every stored address and every partial parity, through five doubling rounds after preparation. This stock witness uses $D=48,F=186$ and 27,980 parameters. It answers the shared-weights existence question, at a cost well above specimen size. At pass $k\ge1$ its guaranteed front is $2^{k-1}$. The raw additive parity table has front $k+1$, and the refreshed model has front $k$. To distinguish algorithms, count data positions and specify the initialization and preparation offset. On balanced bit strings, compare whole window predictions or flip individual inputs and map the resulting changes. A heatmap that only shows a wider region of high accuracy does not identify a particular computation.

## 3. Rule 60

**Proved.** With zero left boundary and arbitrary seed $u$, the row at pass $k$ is
\[
 s^{(k)}=(I+S)^k u\quad\text{over }\mathbb F_2,\qquad
 s_t^{(k)}=\bigoplus_{0\le j\le\min(k,t-1)}\bigl({k\choose j}\bmod2\bigr)u_{t-j}.
\]
Since $(I+S)^{2^r}=I+S^{2^r}$, expand the product over the set bits of $k$. A coefficient is one precisely when all the set bits of $j$ are also set in $k$ (the binary case of Lucas' theorem). At $k=2^m-1$ all $0\le j<2^m$ contribute, giving an output that is length-$2^m$ windowed parity, zero padded at the left. At $k=2^m$ there are just two taps before boundary truncation. On finite width $T$ the operator returns to the identity once $k=2^{\lceil\log_2 T\rceil}$. This is its order, since a smaller power of two leaves a nonzero shift for $T>1$. Individual seeds may have shorter periods. A single leftmost impulse gives the Pascal triangle modulo two, cropped at $T$. Row $k$ has $2^{\operatorname{popcount}(k)}$ lit cells before cropping.

**Same weights question.** The explicit four-feature table in §2 really does compute both maps. Its two reachable stream encodings distinguish the modes and its MLP was built for both. Simply removing the input from the 446-parameter wrapper gives a different result. In that construction the local operand is read from feature 6, which holds the input bit, while the old propagated bit resides in feature 7. One can refresh feature 6 from the current bits to study Rule 60 with that gate, but it would be another outer recurrence. A learned prefix-parity block is under no algebraic obligation to implement the CA when switched to autonomous mode.

## 4. Following a permutation

**Proved algorithm and encoding.** Give one entry $(j,\pi(j))$ per causal memory token, followed by one query token $a$. Keep the memory entries immutable. At each loop call the query addresses the unique entry with key $a$, copies $\pi(a)$ into its new address and thus recovers $\pi^k(a)$ after $k$ hops. In components $a'=\sum_j[j=a]\pi(j)$. As one-hot columns, $u'=P_\pi u$, $(P_\pi)_{ij}=[i=\pi(j)]$. Only the query must see all entries. If keys and values arrive as *separate* word tokens, an additional binding operation or a different head is needed. The example encodes each edge as a single token, uses $n^2+n$ token ids, and costs $T=n+1$ columns.

**Explicit arrays for chasing and doubling.** These are ordinary autonomous PicoGPT blocks. In both cases set $m=n+1$, $D=2(4m+2)+4$, $H=L=1$, and $u_j=(e_{2j-1}-e_{2j})/\sqrt2$. Form four $D\times m$ matrices $Q_c,K_c,V_c,S_c$ whose successive columns are the consecutive groups of $m$ vectors $u_j$ starting at $j=1,m+1,2m+1,3m+1$. Index columns by addresses $0,\ldots,n$. Put $b=u_{4m+1}$ and $h=u_{4m+2}$. An address column is
\[
 C(q,k,v,s)=Q_c[:,q]+K_c[:,k]+V_c[:,v]+(2s-1)b .
\]
It is centered, has norm 2, and has exact LN scale $\sigma_c=\sqrt{4/D+\varepsilon}$. The embedding column for raw token id $j$ is $j e_{D-2}$, in addition to stock $P_t$. All unspecified weights are zero, all LN gains one. The nonzero attention arrays are
\[
 W_q[1:m,:]=4\sqrt D\,\sigma_c Q_c^\mathsf T,\quad
 W_k[1:m,:]=\sigma_c K_c^\mathsf T,\quad
 W_v[1:m,:]=\sigma_c V_c^\mathsf T,\quad
 W_v[m+1,:]=\sigma_c b^\mathsf T,\quad
 W_o[:,1:m]=S_c,\quad W_o[:,m+1]=h .
\]
Thus on canonical columns $S_{ij}=4[q_i=k_j]$. A unique match among at most four columns has weight at least $e^4/(e^4+3)>0.94$. The receiver sees the old local registers and, in orthogonal scratch features, a mixture of value-address one-hots and signed bits. The leading address coordinate and the sign recover the matched registers. At address zero the scan ignores this mixture. For a larger domain replace the 4 in $W_q$ by a sharpness at least $\log(4T)$.

- **Doubling.** With $V=2$, raw token id $1+x_t$ prepares $C(t-1,t,t-1,x_t)$. Each subsequent target is $C(a'_t,t,a'_t,s'_t)$, with the simultaneous updates in §2. Set $W_{\rm out}[2,:]=b^\mathsf T$.
- **Chasing.** Raw edge id $(j-1)n+\pi(j)$ prepares $C(0,j,\pi(j),0)$. Query id $n^2+a$ prepares $C(a,0,0,0)$. The target is the same memory for each edge and $C(\pi(a),0,0,0)$ at the query. Set $W_{\rm out}[n^2+a,:]=Q_c[:,a]^\mathsf T$ for $a=1,\ldots,n$. After preparation the query readout gives its starting address before any hops.

**MLP arrays.** Use precisely the two-hinge formula of §2 with $v_i=\sin(i^2+1)$ and the actual middle column $y=X_t+W_o\sum_{j\le t}A_{tj}W_v\operatorname{LN}_1(X_j)$. Take the finite pairs $(v^\mathsf T\operatorname{LN}_2(y),c-y)$ for the following inputs and targets $c$ at every position. For scan, enumerate every bit word, its preparation pass, and its $\lceil\log_2 n\rceil+1$ canonical rounds. For chasing, enumerate every permutation and starting query, its preparation pass and one canonical hop. Subsequent hops are already represented. This specifies $F$, $W_1,b_1,W_2,b_2$ by finite sums and ordering, just as for the small parity table. The constructors return a flat vector in the repository's layout.

**Why finite compilation extends.** LN with $\varepsilon>0$ is injective on centered vectors since $r/\sqrt{r^2/D+\varepsilon}$ increases strictly. On arbitrary vectors it forgets only the mean. The scratch write is centered and orthogonal to all canonical registers. Thus canonical columns that become identical after the second LN have identical $y$ and identical target. For raw columns the unused final sin/cos pair is untouched by embeddings and scratch. Its contrast identifies the position because for distinct integer times equality of $\sin(\omega t)-\cos(\omega t)$ would require a nonzero algebraic multiple of an integer to equal a nonzero rational multiple of $\pi$, where $\omega$ is the algebraic last frequency. The contrast is nonzero at every integer time and is zero on canonical columns. The spare raw-id coordinate and another untouched coordinate then distinguish ids at that position. Equal centered raw middle columns have the same original mean and hence the same $y$. Consequently a finite list of distinct normalized points has consistent targets. Choose a scalar projection outside the finite union of hyperplanes perpendicular to any difference of distinct points. The stated $v$ is checked for the two supplied configurations. A new finite bound may require a different $v$. This existence proof spends width on storing addresses and can spend exponentially many MLP knots on restoring the stream.

**Checked.** With three labels, all six permutations and three starts, all eight successive hops after preparation and every memory register. The stock block has $D=40,F=106$, 16,398 parameters. Maximum real-state error was below $6.6\,10^{-12}$. The four-bit doubling witness had maximum real-state error below $1.3\,10^{-11}$. One preparation pass is charged in both models. The gap certificates were $1.28\,10^{-4}$ for chasing and $8.40\,10^{-4}$ for doubling.

**One-layer lower bound and its assumptions.** For the two separately supplied arbitrary functions $f,g:[n]\to[n]$, Peng, Narayanan and Papadimitriou, Theorem 1, assume $H$ heads, their vector dimension $d_{\rm paper}$, and $p$-bit numerical communication. If $H(d_{\rm paper}+1)p<n\log_2 n$, a single softmax layer cannot answer $f(g(a))$ correctly on all inputs. Their stated error lower bound for uniform functions and queries is $(n\log_2 n-H(d_{\rm paper}+1)p)/(3n\log_2 n)$. Separate local preprocessing and arbitrary final combination are accommodated. The theorem concerns the paper's finite-precision summary model. [Peng, Narayanan and Papadimitriou, COLM 2024, Theorem 1](https://openreview.net/pdf?id=KidynPuLNW)

**Permutation-specific consequence, proved under the same summary assumption.** Take $2n$ labels $A_i,B_i$. Restrict one permutation to $\pi(A_i)=B_{g(i)}$, $\pi(B_j)=A_{f(j)}$, for independent permutations $f,g$. Fix the query at $A_1$. The single layer's query is fixed before it reads either half. For every head the $B$-half can send its unnormalized weighted value sum and its scalar normalizer to reconstruct the full query output from the other half. If that message took fewer than $\log_2(n!)$ bits, some $f\ne f'$ would have identical messages. Choose $g(1)=j$ at an index where they differ. Then $\pi^2(A_1)$ must differ but the receiver has identical information. Hence $H(d_v+1)p\ge\log_2(n!)$ is necessary if each head sends $d_v+1$ summaries of $p$ bits. Here $d_v$ is the head's value width, $D/H$ in PicoGPT. This is a worst-case size tradeoff for separately supplied entries. It supplies neither an infinite-precision impossibility nor a prohibition for a three-label permutation or a wide one-layer model. Exact partial-sum message sizes have to be counted if a different numerical model is used.

## 5. Linearization and the light cone

**Proved stock symmetry.** Let $\mathbf1\in\mathbb R^D$ and let $c$ have one scalar per token. LN erases the independent translations $\mathbf1c^\mathsf T$. Both residual branches therefore take the same values before and after this translation:
\[
 B(X+\mathbf1c^\mathsf T)=B(X)+\mathbf1c^\mathsf T,\qquad
 \sum_j\frac{\partial B_{it}}{\partial X_{ju}}=\delta_{tu}.
\]
The component identity holds at differentiable points, and for the selected repository pullback along these directions at ReLU kinks. Autonomous, pre-additive, and post-additive loops each have at least $N$ independent unit right eigenvectors at every differentiable state. Every fixed point belongs to a family translated by those vectors. Even if a particular initialized trajectory settles exactly, $\rho(J_*)<1$ and nilpotence of the **full** Jacobian are impossible for these three loops. The final LN ignores this family. One may center per column and study the induced dynamics on the quotient. Stability and invertibility there must be checked anew. Causality makes the positional Jacobian block lower triangular. Its spectrum is the union of the spectra of its diagonal token blocks. Global causal attention allows entries anywhere below that diagonal.

**Three experiments with different answers.**

| Perturbation | Matrix statement | What moves |
|---|---|---|
| Flip one input or old bit in the digital recurrence | $\delta s'=S\,\delta s$ for old-state flips in $\mathbb F_2$. $\delta s^{(k)}=(I+\cdots+S^{k-1})\delta x$ for injected input flips | An old-bit flip advances one site. An injected input flip changes every reached site of its forward cone. |
| Differentiate the signed-product extension | $a_t=(-1)^{x_t}$, $u'_t=a_tu_{t-1}$ with boundary $u_0=1$, $J=\operatorname{diag}(a)S$ | $\bar u_t^{(k)}=a_{t+1}\bar u_{t+1}^{(k+1)}$. A terminal single-site covector walks left by one per pass, flipping sign at negations, and vanishes at the boundary. $J^T=0$ with superscript $T$ denoting the $T$-th power. |
| Differentiate the actual 446-parameter recurrence | $J=C(B_X-I)=0$ locally on each strict Boolean plateau | The MLP writes a locally constant bit and the refresh discards the scratch. A state covector vanishes after one step. There are still direct input/readout paths. |

**Checked with the package pullback.** At the tabulated four-column equilibrium a column-mean shift changed the output by the same shift to within $2.5\,10^{-11}$, and its adjoint pairing agreed to displayed Float64 precision. For a seeded register covector at a refreshed fixed point, the maximum absolute entry of $J^\mathsf T\bar X$ was $2.8\,10^{-16}$ or less.

The raw additive table gives a fourth measurable case. At each tabulated fixed point the MLP's correction is locally constant by construction. Its local stream derivative is that of the first residual, $I+D_X[W_o\,\mathrm{mix}(A,V)]$. In this example $A$ is uniform and $W_v$ reads LN₁. In particular
\[
 J_{tu}=[t=u]I_4+[u\le t]\,e_1(1,-0.7,0.3,0.6)\,D\operatorname{LN}_1(X_u)/t .
\]
Even its off-diagonal response can reach any earlier token in one backward pass. The full stream, the gauge-fixed stream, the register derivative, and a one-bit flip answer different experimental questions. For a trained specimen, seed a logit difference at one site, save its VJP per pass, and **also** compare pairs of inputs differing at one site. Plot the actual gradients and digital differences separately. With deep supervision each pass injects a new covector $g_k$, so use terminal-only loss to isolate transport.

## 6. Exact anchors and how to measure them

The following are predictions of the specified rules. A training run reaching an anchor is a further empirical claim.

| Anchor | Prediction | Measurement |
|---|---|---|
| Tied parameter count | $2VD+V+2D+4D^2+2DF+9D+F$, independent of $K$ | Count the flat vector. An untied $L=K$ model pays the block term $K$ times. The refreshed specimen has 446 parameters. |
| Gradients of the repeated block | $\bar p=\sum_k\bar p^{(k)}$. The shared key bias has zero adjoint for every loss because it shifts each score row uniformly | Compare to a directional finite difference, log per-pass contributions and their sum, and measure the largest absolute entry of $\bar b_k$. |
| Prefix front and intervention | Window lengths $w_k=k,k+1$ or $2^{k-1}$ under the three initializations in §2. Unit accuracy for $t\le w_k$, $1/2$ outside for independent fair bits following these exact rules | Exhaust all short words or use balanced pairs. Compare predicted bits with the whole window XOR. Use per-site flips to recover the window edges. |
| Conditional information under a strict local rule | A predictor that sees only the last $w$ bits has Bayes CE infimum $\max(T-w,0)\log 2/T$ for fair inputs. Its average over pass weights $w_k^{\rm loss}$ is the corresponding weighted sum | Compare soft probabilities with the posterior (one half beyond the front). A hard window-parity predictor can have high CE on unreached sites despite the same accuracy. This floor requires the information restriction. Global softmax has access to the whole prefix. |
| Prefix equilibrium | $s^*=(I-S)^{-1}x$ over $\mathbb F_2$. After the front reaches $T$ every bit stays fixed | Save additional passes. Report class flips and $\lVert G(X,U)-X\rVert_{\rm max}$ separately, and center each stream column to diagnose mean drift. |
| Digital versus real spectrum | Signed-shift extension is nilpotent of index $T$ and has only eigenvalue zero. The refresh plateau has $J=0$. Ordinary pre-LN residual loops have at least $N$ independent unit eigenvectors | Perturb column means exactly. On a small example build $J^\mathsf T$ by basis VJPs, compare to the formula in §5, project to $\mathbf1^\perp$ if quoting a quotient spectrum, and test powers against a localized covector. |
| Rule 60 | Tap offsets are the submasks of $k$. Untruncated impulse weight $2^{\operatorname{popcount}(k)}$, all taps at $k=2^m-1$, finite-width operator period $2^{\lceil\log_2T\rceil}$ | Compare every seed or every impulse row and superpositions to the bitwise oracle. The impulse alone cannot establish that a learned nonlinear model superposes. |
| Address routing | Stock prepared-chase $q_k=\pi^{k}(q_0)$ after $k$ hops. For a cycle of length $d$, readouts and prepared canonical stream repeat after $d$. Each scan round doubles the covered interval up to the left boundary | Read the address, the matched score and the memory entries per pass. Test fixed points and nontrivial cycles on unseen permutations. Probe data-held predecessor pointers in doubling. |
| CE for known deterministic targets | Conditional entropy is zero. Verified correct argmax does not require CE zero. For a strictly separated finite data set, increasing the logit scale sends CE to zero. Finite logits with $V>1$ give positive CE | Report CE, accuracy and the worst margin separately. The original text specimen's $(2\ln2+3\ln3)/80$ comes from different ambiguous contexts. |
| Two possible stopping times | For the Rule 60 experiment in §7 the horizon-blind infimum is $(T-1)\log2/(2T)$ and the best accuracy is $1-(T-1)/(4T)$ | Duplicate each input at both requested horizons with equal weight. Check ambiguous labels, one-pass probabilities and both loop readouts. |

## 7. Audit of the candidate list

| Candidate | Verdict and scope |
|---|---|
| 1. Prefix parity | The linear front is a property of the local update. Doubling and global count shortcuts have different fronts. The ReLU piece bound needs a scalar affine counting path. The full residual derivative cannot be nilpotent. A four-feature exact-arithmetic finite table and a transparent 64-data-bit refreshed construction give two distinct controls. |
| 2. Rule 60 | Lucas, windowed parity, and the cropped triangle all hold with the stated boundary. On a finite line the operator recurs. Shared weights for the two tasks require compatible encodings or explicit training. The post-addition table supplies an existence example. |
| 3. Pointer chase | Composition lower bounds are dimension/precision tradeoffs. A single hop per *algorithmic* round is realizable. It gives no unrestricted two-hop lower bound at small $n$. Pair tokens, ordered value-only slots and split key/value tokens have different vocabulary, length and preparation costs. |
| 4. Odometer | On least-significant-bit-first data the carry into bit $t$ is the AND of all lower bits. $b'_t=b_t\oplus\bigwedge_{j<t} b_j$. Carry is visible with causal attention in that order. A different significance order changes causal access. For a fixed width the period $2^T$ is exact. Another control is needed to establish depth necessity. |
| 5. Denoising | A pattern can be a fixed point, but an attractor in all real directions conflicts with §5 for the stock residual loop. On a differentiable centered map $\rho(J)<1$ suffices for local attraction. Equality is inconclusive. Bidirectionality is relevant if future constraints disambiguate a missing earlier character. A memorized constant line needs no recurrent repair. |
| 6. Harmonic interpolation | A frozen attention matrix acting directly on values can be a Markov operator. PicoGPT also normalizes, projects values, takes residual sums, applies an MLP, and usually changes $A$ with $X$. Even a uniform average only solves particular boundary conditions. A harmonic extension generally depends on geometry and boundary-specific weights. |
| 7. Group tracking | Prefix products and the doubling algorithm extend to finite groups, with the older interval multiplied **on the left** for a left-to-right product. $S_3$ is noncommutative but solvable ($A_3$ is cyclic and the quotient is $C_2$). Treating it as the standard non-solvable-group circuit-complexity example would be incorrect. |
| 8. Shidoku | The check enumerated 288 labeled completed boards. “Propagation depth” additionally requires a specified puzzle, rule set and parallel update schedule. A different serialization can query the whole puzzle causally at the end. The estimated parameter minimum and architectural necessities have no proof here. |
| 9. Liar/truth-teller | On one Boolean bit, negation has one two-cycle and identity has two fixed points. A continuous real extension can have further behavior. For example, $f(u)=1-u$ has the real fixed point $1/2$. |
| 10. Odd–even sorting | Rank by counts needs the entire multiset, a tie convention, and a way to put values into output slots. In-place globally sorted output at an early causal position is impossible when a later unseen symbol can change that output, for **any** number of causal passes. Separate output query positions or bidirectional attention change the task. |

**New candidate with a rigorous one-pass control using Rule 60 at two stopping times.** Provide a uniformly random bit word $x$ and score $(I+S)x$ at pass 1 and $(I+S)^2x$ at pass 2 with equal weight. On the control, use the same one-pass readout for both observations of the same $x$. It has no horizon input. The conditional label at site 1 always agrees. At each $t\ge2$ the two labels differ precisely if $x_{t-1}\ne x_{t-2}$, with $x_0=0$. Its probability is $1/2$. Given a disagreement the posterior is $(1/2,1/2)$. Otherwise the label is certain. Averaging yields the two exact anchors in §6. At $T=8$ exhaustive enumeration gives disagreement on $7/16$ of word/site pairs, a CE infimum of $(7/16)\ln2$, and accuracy ceiling $25/32$. This is a lower bound on CE even for a one-pass control of unlimited width, and the same recurrent block can have zero classification errors at both stopping times. The experiment tests whether one state update can be reused at different requested horizons. The number of passes conveys the horizon. A one-pass control explicitly given the horizon or separate readouts for the two targets faces a different question. For an architecture-enforced **spatial** separation, restrict all communicating layers to one left neighbour and self. Its receptive field gives the conditional entropy bound in §6, at the cost of changing the attention mask.

**Reproduction.** From the repository root run

~~~sh
julia --project loop/scratch/astra/check.jl
~~~

The script calls PicoGPT's actual embedding, block, LayerNorm and readout, and uses independent bit and pointer recurrences as oracles. To inspect or use a flat weight vector, load the constructors as follows.

~~~julia
using PicoGPT
include("loop/scratch/astra/handbuilt.jl")
c, θ, gap = Handbuilt.stock(4)                     # B(X)+U or B(X)
c8, θ8 = Handbuilt.compact_config(), Handbuilt.compact()
cs, θs, _ = Handbuilt.addressed(:scan, 4)          # stock autonomous block
cp, θp, _ = Handbuilt.addressed(:chase, 3)
p = carve(θ8, c8)
U, _ = embed(reshape([1, 2, 1, 2], :, 1), p.E)  # neutral sentinel, then 1,0,1
X = Handbuilt.register_step(U, U, only(p.blocks))
~~~

The constructors expose the table formulas as arrays. The helpers `stock_theta` and `addressed_theta` return just the vector. The `register_step` helper expects one sequence at a time.

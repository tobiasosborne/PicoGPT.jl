# Attention {#ch-attention}

::: claim
Attention is an ansatz for the next-character conditional: treat it as a mixture over an uncertain source $j\le i$, weight sources by a low-rank bilinear Gibbs law (query and key), and average source vectors instead of predictions.
:::

::: names
| maths | physics | ML |
|---|---|---|
| row-stochastic lower-triangular $A$ | causal Gibbs weights | attention pattern |
| $S_{ij}=-E_{ij}/\tau=q_i\cdot k_j/\sqrt d$ | negative energy / $\tau$ | attention score, logit |
| $B=W_q^\T W_k$ ($\operatorname{rank}\le d$); $q_i,k_j$ | low-rank bilinear interaction | query, key |
| $v_j=W_v\hat x_j+b_v$ | vector averaged | value |
| $o_i=\sum_{j\le i}A_{ij}v_j$ | expectation $\langle v\rangle_i$ | attention output |
| $\sqrt d$ | temperature (Gibbs analogy) | scale factor |
| $A_{ij}=0$ for $j>i$ | causality | causal mask |
| $\R^D=\R^d\oplus\cdots\oplus\R^d$ | direct sum | multi-head attention |
:::

## The table and the uncertain index {#sec-index}

The chain rule (@ch-task) is exact yet gives no recipe for its factors: predicting $Y$ from $c=(a_1,\dots,a_i)$ uses only $1,\dots,i$, and an unrestricted table $P(y\mid c)$ has $V^i(V-1)$ free entries. Every other map—embedding $Ee_{a_t}+P_{:,t}$ (@ch-vectors), the column-wise maps of @ch-block and the readout (@ch-energies)—acts column by column, $X\mapsto(f(x_1),\dots,f(x_T))$, a bigram model with a clock. Positions interact here and nowhere else.

::: aside
A matrix product state writes a table across a cut as $\sum_{r=1}^{\chi}L_r(u)R_r(v)$: the bond $\chi$ bounds cut rank. Attention stores up to $T$ columns with a shared addressing rule; $d$ bounds only a pair score.
:::

Five steps replace the table: Steps 2–4 *restrict* the conditional family—approximations since $P$ may lie outside it—and Step 5 replaces one predictor by another.

*Step 1: an uncertain index.* Fix $c$ with $P(c)>0$ and adjoin $J\in\{1,\dots,i\}$. Any distribution $r(j\mid y,c)$ on $j\le i$ defines $\tilde P(y,j\mid c)=P(y\mid c)\,r(j\mid y,c)$; with $\pi_j=\sum_y\tilde P(y,j\mid c)$ and $p_j(y)=\tilde P(y,j\mid c)/\pi_j$ (arbitrary if $\pi_j=0$),
$$P(y\mid c)=\sum_{j=1}^{i}\pi_j\,p_j(y).$$
Forced at $i=1$, this is free for $i>1$ ($P$ never fixes $r$); uniform $r=1/i$ (ignoring $y$) leaves $p_j=P(y\mid c)$ intact. Exact and empty: the first restriction assumes a mixture approximating $P$ with simple shared rules for both $\pi_j$ and $p_j$ across prefixes, excluding characters fixed only by joint multi-position patterns.

*Step 2: a shared rule on position vectors.* Position $j\le i$ carries $x_j\in\R^D$ (initially $Ee_{a_j}+P_{:,j}$, positional code $P_{:,j}$) and a rescaled copy $\hat x_j$ (@ch-block). In one $d$-dimensional *head* (@sec-heads; two in the specimen, $h$ dropped), score $(\hat x_i,\hat x_j)$ over $j\le i$ for $\pi_\theta(j\mid c)$, set $v_j=W_v\hat x_j+b_v\in\R^d$, and share $g_\theta(x_i,v_j)\in\Delta^{V-1}$:
$$p_{\rm mix}(y\mid c;\theta)=\sum_{j=1}^{i}\pi_\theta(j\mid c)\,g_{\theta,y}(x_i,v_j).$$
Here $g_\theta(x_i,m)$ adds $W_o m+b_o$ to $x_i$ and runs the remaining column-wise maps of @ch-block, so $g_\theta(x_i,v_j)$ predicts from $v_j$ alone (other head fixed). Reader $i$ enters the score via $\hat x_i$, may choose itself ($j=i$), and passes $x_i$ to $g_\theta$. Equal $v_j$ give identical components, so $g_\theta$ cannot separate sources collapsed by $W_v\hat x_j+b_v$ (erased features can still affect scores). ML calls $v_j$ the *value*.

## A low-rank Gibbs energy {#sec-energy}

*Step 3: a bilinear Gibbs energy.* Writing positive weights as $\pi_j=Z_i^{-1}e^{-E_{ij}/\tau}$ ($j\le i$, @ch-energies) is exact ($E_{ij}=-\tau\log\pi_j$); finite energies keep $\pi_j>0$. The restriction is one shared bilinear energy on $(\hat x_i,\hat x_j)$ (plus offsets), with no distance penalty—position enters only via $P_{:,i},P_{:,j}$ and earlier reads.

*Step 4: low rank.* Factor $B=W_q^\T W_k$ ($W_q,W_k\in\R^{d\times D}$, one head's row blocks) with offsets $b_q,b_k\in\R^d$:
$$q_i=W_q\hat x_i+b_q,\qquad k_j=W_k\hat x_j+b_k,\qquad S_{ij}=\frac{q_i\cdot k_j}{\sqrt d},\qquad A_{ij}=\frac{e^{S_{ij}}}{\sum_{j'\le i}e^{S_{ij'}}}\quad(j\le i),$$
so $E_{ij}=-q_i\cdot k_j$, $\tau=\sqrt d$, $\pi_\theta(j\mid c)=A_{ij}$. *Query* $q_i$ and *key* $k_j$ are simply these factors; $B$ need not be symmetric. Expanding separates the offsets:
$$q_i\cdot k_j=\underbrace{\hat x_i^\T W_q^\T W_k\hat x_j+b_q^\T W_k\hat x_j}_{\text{depends on }j}+\underbrace{(W_q\hat x_i+b_q)^\T b_k}_{\text{constant in row }i}.$$
Thus $b_q$ adds a source-linear bias, whereas $b_k$ shifts row $i$ equally, cancels out of $A$ and has zero gradient—answering the puzzle of @ch-energies. A positive row fits across contexts if and only if $\log(\pi_j/\pi_{j'})=(W_q\hat x_i+b_q)^\T W_k(\hat x_j-\hat x_{j'})/\sqrt d$. Before softmax, $\operatorname{rank} B\le d$; $2dD$ entries beat $D^2$ when $2d<D$, while the specimen ($D=8,d=4$) stores 64 against 64 per head (plus 8 biases), imposing rank $\le 4$ without saving entries.

### Why $\sqrt d$ {#sec-sqrt-d}

For independent coordinates of $q_i,k_j$ with mean zero and variance one, $\operatorname{Var}(q_i\cdot k_j)=d$. Large score gaps concentrate a row and shrink its softmax Jacobian (@ch-pulling-back); dividing by $\sqrt d$ sets marginal score variance to one at initialisation, neither frozen nor uniform.

::: aside
This concerns initialisation only; after training, scale and entropy follow the loss. Normalisation is row by row (each with its own $Z_i$); nothing equilibrates between readers.
:::

@fig-attention-reads shows both heads.

::: {.figure #fig-attention-reads window=10 reader=8}
Gibbs weights $A_{ij}$ over $j\le i$ in both heads (@sec-heads), head 1 above and head 2 below. Arrows leave the reader for its sources; width is $A_{ij}$, loops are self-reads, and each head sums to one. [Drag the window.]{.web}
:::

## The mean-field step {#sec-meanfield}

*Step 5: replace the selected vector by its expectation.* Average values first and decode once:
$$o_i=\sum_{j=1}^{i}A_{ij}\,v_j,\qquad A_{ij}=\frac{e^{q_i\cdot k_j/\sqrt d}}{\sum_{j'\le i}e^{q_i\cdot k_{j'}/\sqrt d}},\qquad O_h=V_h A_h^\T\in\R^{d\times T},$$
with $(q_i,k_j,v_j)=(W_q\hat x_i+b_q,\,W_k\hat x_j+b_k,\,W_v\hat x_j+b_v)$, replacing $\sum_{j\le i}A_{ij}g_{\theta,y}(x_i,v_j)$ by $g_{\theta,y}(x_i,o_i)$. Forming $A$ and $o_i=\E_{J\sim A_{i,:}}[v_J]$ (in the convex hull of $v_1,\dots,v_i$) is *attention*; moving an expectation inside a nonlinear map is a *mean-field* step.

::: aside
Rows are readers and columns are sources in $S,M,A$. Since tokens are columns of $V_h\in\R^{d\times T}$, $O_h=V_h A_h^\T$ needs the transpose.
:::

::: {.figure #fig-attention-pipeline window=10 reader=8 head=1}
One reader and head, top to bottom: keys $k_j$, query $q_i$, scores $S_{ij}=q_i\cdot k_j/\sqrt d$, weights $A_{ij}$, and values $v_j$ (muted) with shares $A_{ij}v_j$ (solid), summing to $o_i$. Lower $\tau$ concentrates on top scores; higher $\tau$ flattens towards uniform. [Drag $\tau$.]{.web}
:::

Training fits $g_\theta(x_i,o_i)$ directly, so comparing it with $\sum_j A_{ij}g_\theta(x_i,v_j)$ measures distance from the selection picture, not network error. Affine $g_y(m)=g_{\theta,y}(x_i,m)$ makes them coincide. Where $g_y$ is smooth on segments from $o_i$ to $v_j$, linear terms cancel and the gap is second order in $s^2=\sum_j A_{ij}\|v_j-o_i\|^2$ (small for clustered values or near-copies); a ReLU kink (@ch-block) between them can make it first order.

Let $z(m)\in\R^V$ be the logits and $p_j=\softmax(z(v_j))$ the prediction from $v_j$ alone. Because $\log p_j(y)=z_y(v_j)-\lse(z(v_j))$ shifts every character equally, the normalised geometric mixture is
$$p^{\rm geo}_y=\frac{\prod_{j\le i}p_j(y)^{A_{ij}}}{\sum_a\prod_{j\le i}p_j(a)^{A_{ij}}}=\softmax\!\left(\sum_{j\le i}A_{ij}\,z(v_j)\right)_y.$$
Hence $p^{\rm geo}=\softmax(z(o_i))$ if and only if $z(o_i)-\sum_j A_{ij}z(v_j)\in\operatorname{span}(\one)$: exact for affine logits, while PicoGPT's nonlinear readout (@ch-block) agrees through first order where $z$ is smooth on those segments, and can differ at first order across a ReLU kink. Finite logits keep $p_j(y)>0$: no source literally vetoes a character.

At the trained parameters, swapping both heads' outputs independently with weights $A^{(1)}_{ij_1}A^{(2)}_{ij_2}$ (@sec-heads; $x_i$, rows, values fixed) over all {{value:mixtureReaderCount}} readers of the ten overlapping training windows gives equal-weight mean cross-entropy {{value:mixtureLossActual}} nats (actual, near the floor of @ch-surprise), {{value:mixtureLossGeometric}} (geometric) and {{value:mixtureLossArithmetic}} (arithmetic); mean total variation $\tfrac12\sum_y|p_y-q_y|$ from $p$ is {{value:mixtureTvGeometric}} vs {{value:mixtureTvArithmetic}}. @fig-attention-meanfield isolates head 1 across the 70 multi-source readers.

::: {.figure #fig-attention-meanfield}
Reader 't' at position 6 of `or not t`, head 1 (head 2 at its usual output). Under each source $j$ is $p_j$ for $v_j$ alone; below, three combinations on one scale. Arithmetic $\sum_jA_{ij}p_j$ adds probability shares (sources only add); geometric adds weighted surprises $A_{ij}(-\ln p_j)$ and renormalises, pricing out 'o' where a source finds it nearly impossible, and the model's $p$ sits with it. The strip repeats both total-variation distances over all 70 multi-source readers: geometric is closer at every one.
:::

## Every reader at once: the causal mask {#sec-mask}

One pass predicts $a_{i+1}$ from $a_{\le i}$ for all $T$ positions (@ch-task). For $Q_h,K_h\in\R^{d\times T}$ and $S_h=Q_h^\T K_h/\sqrt d$, the range $j\le i$ becomes the *causal mask*:
$$M_{ij}=\begin{cases}0,&j\le i,\\ -\infty,&j>i,\end{cases}\qquad A_h=\softmax_{\text{rows}}(S_h+M).$$
Since $e^{-\infty}=0$, $A_h$ is lower-triangular, rows sum to one over $j\le i$, and $A_{11}=1$. Column $i$ of $O_h=V_h A_h^\T$ uses only columns $1,\dots,i$; induction over column-wise maps gives a block lower-triangular Jacobian where differentiable, yielding all $T$ factors $q_\theta(a_{i+1}\mid a_{1:i})$, multiplying to $q_\theta(a_{2:T+1}\mid a_1)$.

::: {.figure #fig-attention-triangle window=10}
Scores, mask and weights of both heads for one window (rows: reader $i$; columns: source $j$). Left: $S_{ij}$ for all pairs—dot area grows with $|S_{ij}|$, filled positive, open negative. Middle: $S+M$, upper triangle removed. Right: weights $A_{ij}$, area $\propto A_{ij}$, rows summing to one. [Run the computation.]{.web}
:::

::: aside
$S_h$ is computed for all pairs (quadratic in $T$); masking discards $T(T-1)/2$ entries (28 of 64 per head and window). Causal implementations can skip upper-triangular work.
:::

## Heads: one index per subspace {#sec-heads}

PicoGPT splits $\R^D$ into $H$ subspaces of width $d=D/H$: in the specimen, $\R^8=\R^4\oplus\R^4$ ($D=8,H=2,d=4$), rows $1$–$4$ of $Q,K,V$ in head 1 and $5$–$8$ in head 2. The split is in the *outputs*: each $W_q^{(h)},W_k^{(h)},W_v^{(h)}\in\R^{d\times D}$ reads all $D$ coordinates of $\hat x_i$.

Each head gets an index $J_h\le i$. Independent indices ($\prod_h A^{(h)}_{ij_h}$) on $m_{\mathbf j}=\bigoplus_h v^{(h)}_{j_h}$ define the joint mixture of @sec-meanfield; by linearity, the mean-field output uses only marginals:
$$\E[m_{\mathbf J}]=\bigoplus_{h=1}^{H}o_i^{(h)},\qquad O_h=V_h A_h^\T\in\R^{d\times T},\qquad O=\begin{pmatrix}O_1\\[-2pt]\vdots\\[-1pt]O_H\end{pmatrix}\in\R^{D\times T}.$$
Any joint law with those marginals gives the same $O$: attention cannot correlate heads' reads. Then $W_o\in\R^{D\times D}$ recombines blocks into the stream (@ch-block):
$$X\;\longleftarrow\;X+W_oO+b_o\one^\T.$$
At fixed $D$, more heads allow more distinct patterns (rank $\le d$ each); in deeper blocks, $x_j$ includes earlier $W_oO+b_o\one^\T$, so later scores depend on earlier reads. Together $W_q,W_k,W_v,W_o$ and biases hold $4(D^2+D)$ numbers—{{value:attentionParams}} specimen parameters, replacing the $V^i(V-1)$ table for any $T$.

## In the code {#sec-code-attention}

Two functions implement the chapter. First, Steps 3–4 (with the mask of @sec-mask) for all heads and windows:

::: {.code file="src/attention.jl" fn="attention"}
:::

Second, Step 5, $O_h=V_h A_h^\T$, head by head:

::: {.code file="src/attention.jl" fn="mix"}
:::

Each returns its value and pullback `back`, mapping output to input adjoints (@ch-pulling-back).

## Exercises {.unnumbered}

::: exercise
Show that equal scores in row $i$ make $o_i$ the plain average of $v_1,\dots,v_i$. Which reader sees no difference between masked and unmasked softmax? For which reader do arithmetic, geometric and mean-field predictions coincide identically?
:::

::: exercise
Show that adding a constant to row $i$ of $S$ leaves row $i$ of $A$ unchanged. From $q_i\cdot k_j$ in @sec-energy, deduce that $b_k$ never affects $A$ and has zero gradient, whereas $b_q$ can learn.
:::

::: exercise
In a one-block model with $P=0$, $\hat x_j$ depends only on the character at $j$. Show that $o_i$ is invariant under permuting sources $1,\dots,i-1$: which prefixes with the same last character give the same prediction? Which step fails for a second causal block?
:::

::: exercise
For bits $a,b\in\{0,1\}$, let $o(a,b)=(a+3^b b)/(1+3^b)$, weights $(\tfrac12,\tfrac12)$ for $b=0$ and $(\tfrac14,\tfrac34)$ for $b=1$ (e.g. scores $0,b\log 3$). Verify $ab=2a+3b-4\,o(a,b)$. Give the decoder $(a,b,o(a,b))$; for any conditional with $0<P(Y=1\mid a,b)<1$, write log-odds as $c_0+c_1a+c_2b+c_3ab$, substitute the identity, and apply binary Gibbs readout.
:::

::: exercise
Treating $v_j$ and raw scores $S_{ij}$ as independent arguments of one head, compute $\partial o_i/\partial v_j$ and $\partial o_i/\partial S_{ij}$. Express the second via $A_{ij},v_j,o_i$: raising one score pulls $o_i$ towards $v_j$ by how much?
:::

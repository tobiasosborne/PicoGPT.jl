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

The chain rule of probability (@ch-task) is exact, yet gives no recipe for its factors. Predicting the next character $Y$ from a prefix $c=(a_1,\dots,a_i)$ uses only positions $1,\dots,i$, and an unrestricted table $P(y\mid c)$ has $V^i(V-1)$ free entries. Every other map—the embedding $Ee_{a_t}+P_{:,t}$ (@ch-vectors), the column-wise maps of @ch-block, and the readout (@ch-energies)—acts one column at a time:
$$X=(x_1,\dots,x_T)\;\longmapsto\;\bigl(f(x_1),\dots,f(x_T)\bigr).$$
That alone is a bigram model with a clock. Positions interact here and nowhere else.

::: aside
A matrix product state writes a table across a cut as $\sum_{r=1}^{\chi}L_r(u)R_r(v)$, so its bond dimension $\chi$ bounds cut rank. Attention keeps up to $T$ columns with a shared addressing rule, and $d$ bounds only a pair score.
:::

Five steps replace the table. Step 1 is an exact identity. Steps 2–4 *restrict* the conditional family—approximations, since $P$ may lie outside it—and Step 5 replaces one predictor by another.

*Step 1: an uncertain index.* We rewrite $P(y\mid c)$ as a mixture over an uncertain earlier position. Fix $c$ with $P(c)>0$ and adjoin a random index $J\in\{1,\dots,i\}$, enlarging the sample space. The original law $P$ never fixes the enlarged distribution: we *choose* an extension. Pick any distribution $r(j\mid y,c)$ on $j\le i$ and set $\tilde P(y,j\mid c)=P(y\mid c)\,r(j\mid y,c)$. With marginal weights $\pi_j=\sum_y\tilde P(y,j\mid c)$ and components $p_j(y)=\tilde P(y,j\mid c)/\pi_j$ (arbitrary if $\pi_j=0$),
$$P(y\mid c)=\sum_{j=1}^{i}\pi_j\,p_j(y).$$
At $i=1$ the index is forced, while for $i>1$ the choice of $r$ is completely free. Uniform $r(j\mid y,c)=1/i$, which ignores $y$, leaves $p_j(y)=P(y\mid c)$ intact: the identity alone is exact and empty. The first real assumption is that some choice makes both $\pi_j$ and $p_j$ follow simple shared rules across prefixes, approximating $P$. Nothing guarantees such a choice exists. It holds when single earlier positions carry the cue, and fails for characters fixed only by joint multi-position patterns.

*Step 2: a shared rule on position vectors.* We replace prefix-specific tables by one shared rule on position vectors. Position $j\le i$ carries a stream vector $x_j\in\R^D$ and its rescaled copy $\hat x_j$ (@ch-block); initially $x_j=Ee_{a_j}+P_{:,j}$, with positional code $P_{:,j}$. We work inside one $d$-dimensional *head* (the specimen has two, @sec-heads) and drop the head label $h$ until then. A shared rule scores $(\hat x_i,\hat x_j)$ over $j\le i$ to produce the weights $\pi_\theta(j\mid c)$. A shared affine map forms the *value* $v_j=W_v\hat x_j+b_v\in\R^d$, and a shared predictor $g_\theta(x_i,v_j)\in\Delta^{V-1}$ produces each source's prediction:
$$p_{\rm mix}(y\mid c;\theta)=\sum_{j=1}^{i}\pi_\theta(j\mid c)\,g_{\theta,y}(x_i,v_j).$$

Here $g_\theta(x_i,m)$ places $m$ in this head's slot of the attention output, keeps the other head's output as it is, applies $W_o$ and $b_o$, adds the result to $x_i$, and runs the remaining column-wise maps of @ch-block, so $g_\theta(x_i,v_j)$ predicts from $v_j$ alone. Reader $i$ enters the score via $\hat x_i$, may choose itself ($j=i$), and passes $x_i$ to $g_\theta$. Nothing here is exact for a general $P$: sharing one rule is a genuine restriction. It holds when a source's predictive information fits in $v_j$, and fails whenever $W_v\hat x_j+b_v$ collapses distinct sources onto equal values $v_j$, since equal values give identical components $g_\theta(x_i,v_j)$.

::: aside
Features erased by $W_v\hat x_j+b_v$ cannot change $g_\theta(x_i,v_j)$, though they can still affect the scores $\pi_\theta(j\mid c)$, which see $\hat x_j$.
:::

## A low-rank Gibbs energy {#sec-energy}

*Step 3: a bilinear Gibbs energy.* We give $\pi_j$ a Gibbs form driven by one shared bilinear energy. Writing strictly positive weights $\pi_j>0$ over $j\le i$ as a Gibbs law (@ch-energies),
$$\pi_j=\frac{1}{Z_i}\,e^{-E_{ij}/\tau},\qquad Z_i=\sum_{j'\le i}e^{-E_{ij'}/\tau}\qquad(j\le i),$$
is exact: set $E_{ij}=-\tau\log\pi_j$, and finite energies keep $\pi_j>0$. The restriction is demanding one shared bilinear energy on $(\hat x_i,\hat x_j)$, plus offsets, across all pairs and prefixes. There is no distance penalty: position enters only through $P_{:,i}$, $P_{:,j}$, and earlier reads. Distant sources can therefore take most of the weight, whereas non-log-bilinear rules or exact zeros on earlier positions lie outside this family.

*Step 4: low rank.* We restrict the $D\times D$ interaction matrix $B$ to rank at most $d$ by factoring $B=W_q^\T W_k$, where $W_q,W_k\in\R^{d\times D}$ are one head's row blocks, with offsets $b_q,b_k\in\R^d$. Define the *query* $q_i\in\R^d$ and *key* $k_j\in\R^d$:
$$q_i=W_q\hat x_i+b_q,\qquad k_j=W_k\hat x_j+b_k.$$
Their scaled inner product gives the score $S_{ij}$ and weights $A_{ij}$:
$$S_{ij}=\frac{q_i\cdot k_j}{\sqrt d},\qquad A_{ij}=\frac{e^{S_{ij}}}{\sum_{j'\le i}e^{S_{ij'}}}\qquad(j\le i).$$
Thus $E_{ij}=-q_i\cdot k_j$, $\tau=\sqrt d$, and $\pi_\theta(j\mid c)=A_{ij}$. Query and key are simply these factors, and $B$ need not be symmetric.

Expanding separates the offsets:
$$q_i\cdot k_j=\underbrace{\hat x_i^\T W_q^\T W_k\hat x_j+b_q^\T W_k\hat x_j}_{\text{depends on }j}+\underbrace{(W_q\hat x_i+b_q)^\T b_k}_{\text{constant in row }i}.$$
Thus $b_q$ adds a source-linear bias, whereas $b_k$ shifts row $i$ equally, cancels out of $A$, and has gradient identically zero—answering the puzzle of @ch-energies.

Before softmax, $\operatorname{rank} B\le d$. Given a strictly positive row of weights for every context, one shared choice of $W_q$, $b_q$ and $W_k$ produces all of them if and only if, in every context,
$$\log\frac{\pi_j}{\pi_{j'}}=\frac{(W_q\hat x_i+b_q)^\T W_k(\hat x_j-\hat x_{j'})}{\sqrt d}$$
for all $j,j'\le i$, failing whenever scoring requires rank above $d$. The factors store $2dD$ matrix entries, beating $D^2$ when $2d<D$. In the specimen ($D=8, d=4$), each head stores $64$ matrix entries (plus $8$ biases) against $64$ for an unfactored $B$, imposing $\operatorname{rank} B\le 4$ without saving entries.

### Why $\sqrt d$ {#sec-sqrt-d}

Suppose the coordinates of $q_i$ and $k_j$ are independent with mean zero and variance one. Then $\operatorname{Var}(q_i\cdot k_j)=d$. Large score gaps concentrate a row onto its top score and shrink its softmax Jacobian (@ch-pulling-back). Dividing by $\sqrt d$ sets marginal score variance to one at initialisation, neither frozen nor uniform.

::: aside
This concerns initialisation only; after training, score scale and entropy follow the loss. Normalisation is row by row, with each reader carrying its own $Z_i$; nothing equilibrates between readers.
:::

@fig-attention-reads shows the trained weights $A_{ij}$ in both heads.

::: {.figure #fig-attention-reads window=10 reader=8}
Gibbs weights $A_{ij}$ over $j\le i$ in both heads (@sec-heads), head 1 above and head 2 below. Arrows leave the reader for its sources; width is $A_{ij}$, loops are self-reads, and each head sums to one. [Drag the window.]{.web}
:::

## The mean-field step {#sec-meanfield}

*Step 5: replace the selected vector by its expectation.* We average the values first and decode once, instead of decoding each source separately:
$$o_i=\sum_{j=1}^{i}A_{ij}\,v_j,\qquad A_{ij}=\frac{e^{q_i\cdot k_j/\sqrt d}}{\sum_{j'\le i}e^{q_i\cdot k_{j'}/\sqrt d}},\qquad O_h=V_h A_h^\T\in\R^{d\times T},$$
with $q_i=W_q\hat x_i+b_q$, $k_j=W_k\hat x_j+b_k$, and $v_j=W_v\hat x_j+b_v$. This replaces the mixture of predictions by one prediction at the mean value:
$$\sum_{j=1}^{i}A_{ij}\,g_{\theta,y}(x_i,v_j)\;\longrightarrow\;g_{\theta,y}(x_i,o_i).$$
Forming $A$ and $o_i=\E_{J\sim A_{i,:}}[v_J]$, a point in the convex hull of $v_1,\dots,v_i$, is *attention*. Moving an expectation inside a nonlinear map is a *mean-field* step.

::: aside
Rows are readers and columns are sources in $S$, $M$, and $A$. Since tokens are columns of $V_h\in\R^{d\times T}$, $O_h=V_h A_h^\T$ needs the transpose.
:::

::: {.figure #fig-attention-pipeline window=10 reader=8 head=1}
One reader and head, top to bottom: keys $k_j$, query $q_i$, scores $S_{ij}=q_i\cdot k_j/\sqrt d$, weights $A_{ij}$, and values $v_j$ (muted) with shares $A_{ij}v_j$ (solid), summing to $o_i$. Lower $\tau$ concentrates on top scores; higher $\tau$ flattens towards uniform. [Drag $\tau$.]{.web}
:::

Training fits $g_\theta(x_i,o_i)$ directly, so comparing it with $\sum_j A_{ij}g_\theta(x_i,v_j)$ measures distance from the selection picture, not network error. Write $g_y(m)=g_{\theta,y}(x_i,m)$. The replacement is exact whenever $g_y$ is affine in $m$. Where $g_y$ is smooth on the segments from $o_i$ to $v_j$, linear terms cancel and the gap is second order in $s^2=\sum_j A_{ij}\|v_j-o_i\|^2$, staying small for clustered values or near-copies. A ReLU kink (@ch-block) between $o_i$ and some $v_j$ can make it first order.

The model sits much closer to a *geometric* mixture of single-source predictions. Let $z(m)\in\R^V$ be the logits from $m$, and $p_j=\softmax(z(v_j))$ the prediction from $v_j$ alone. Because $\log p_j(y)=z_y(v_j)-\lse(z(v_j))$ shifts every character equally, the normalised geometric mixture is
$$p^{\rm geo}_y=\frac{\prod_{j\le i}p_j(y)^{A_{ij}}}{\sum_a\prod_{j\le i}p_j(a)^{A_{ij}}}=\softmax\!\left(\sum_{j\le i}A_{ij}\,z(v_j)\right)_y.$$
Hence $p^{\rm geo}=\softmax(z(o_i))$ if and only if $z(o_i)-\sum_j A_{ij}z(v_j)\in\operatorname{span}(\one)$, which holds exactly for affine logits. PicoGPT's nonlinear readout (@ch-block) agrees through first order where $z$ is smooth on those segments, and can differ at first order across a ReLU kink. Finite logits keep $p_j(y)>0$, so no source literally vetoes a character.

At the trained parameters, hold each reader's $x_i$, attention rows, and values fixed, and swap both heads' outputs independently with weights $A^{(1)}_{ij_1}A^{(2)}_{ij_2}$ (@sec-heads). Over all {{value:mixtureReaderCount}} readers of the ten overlapping training windows, equal-weight mean cross-entropy is {{value:mixtureLossActual}} nats for the actual model (near the floor of @ch-surprise), {{value:mixtureLossGeometric}} nats for geometric, and {{value:mixtureLossArithmetic}} nats for arithmetic. Mean total variation $\tfrac12\sum_y|p_y-q_y|$ from $p$ is {{value:mixtureTvGeometric}} for geometric against {{value:mixtureTvArithmetic}} for arithmetic. @fig-attention-meanfield isolates head 1 across the 70 multi-source readers.

::: {.figure #fig-attention-meanfield}
Reader 't' at position 6 of `or not t`, head 1 (head 2 at its usual output). Under each source $j$ is $p_j$ for $v_j$ alone; below, three combinations on one scale. Arithmetic $\sum_jA_{ij}p_j$ adds probability shares, so sources only add; geometric adds weighted surprises $A_{ij}(-\ln p_j)$ and renormalises, pricing out 'o' where a source finds it nearly impossible, and the model's $p$ sits with it. The strip repeats both total-variation distances over all 70 multi-source readers: geometric is closer at every one.
:::

## Every reader at once: the causal mask {#sec-mask}

One forward pass predicts $a_{i+1}$ from $a_{\le i}$ for all $T$ positions (@ch-task). Stack head $h$'s queries and keys as $Q_h,K_h\in\R^{d\times T}$, with score matrix $S_h=Q_h^\T K_h/\sqrt d$. The range $j\le i$ becomes the *causal mask* $M\in\R^{T\times T}$:
$$M_{ij}=\begin{cases}0,&j\le i,\\ -\infty,&j>i,\end{cases}\qquad A_h=\softmax_{\text{rows}}(S_h+M).$$
Since $e^{-\infty}=0$, $A_h$ is lower-triangular, rows sum to one over $j\le i$, and $A_{11}=1$. Column $i$ of $O_h=V_h A_h^\T$ uses only columns $1,\dots,i$. By induction through the column-wise maps, the whole map has a block lower-triangular Jacobian where differentiable. One pass thus yields all $T$ factors $q_\theta(a_{i+1}\mid a_{1:i})$, whose product is $q_\theta(a_{2:T+1}\mid a_1)$.

::: {.figure #fig-attention-triangle window=10}
Scores, mask and weights of both heads for one window (rows: reader $i$; columns: source $j$). Left: $S_{ij}$ for all pairs—dot area grows with $|S_{ij}|$, filled positive, open negative. Middle: $S+M$, upper triangle removed. Right: weights $A_{ij}$, area $\propto A_{ij}$, rows summing to one. [Run the computation.]{.web}
:::

::: aside
$S_h$ is computed for all pairs, quadratic in $T$; masking discards $T(T-1)/2$ entries (28 of 64 per head and window). Causal implementations can skip upper-triangular work.
:::

## Heads: one index per subspace {#sec-heads}

PicoGPT splits $\R^D$ into $H$ subspaces of width $d=D/H$. In the specimen, $D=8$, $H=2$, and $d=4$, so $\R^8=\R^4\oplus\R^4$: rows $1$–$4$ of $Q,K,V$ belong to head 1 and rows $5$–$8$ to head 2. The split is in the *outputs* only: each $W_q^{(h)},W_k^{(h)},W_v^{(h)}\in\R^{d\times D}$ reads all $D$ coordinates of $\hat x_i$.

Each head gets its own random index $J_h\le i$. Choosing the indices independently—so a tuple $\mathbf j=(j_1,\dots,j_H)$ gets weight $\prod_h A^{(h)}_{ij_h}$—and stacking the selected values into $m_{\mathbf j}=\bigoplus_h v^{(h)}_{j_h}$ defines the joint mixture of @sec-meanfield. By linearity, the mean-field output uses only the marginals:
$$\E[m_{\mathbf J}]=\bigoplus_{h=1}^{H}o_i^{(h)},\qquad O_h=V_h A_h^\T\in\R^{d\times T},$$
stacked into the full output
$$O=\begin{pmatrix}O_1\\[-2pt]\vdots\\[-1pt]O_H\end{pmatrix}\in\R^{D\times T}.$$
Any joint law with those marginals gives the same $O$: attention cannot correlate heads' reads. Then $W_o\in\R^{D\times D}$ recombines blocks into the stream (@ch-block):
$$X\;\longleftarrow\;X+W_oO+b_o\one^\T.$$
At fixed $D$, more heads allow more distinct patterns, each of rank at most $d$. In deeper blocks, each $x_j$ already includes earlier updates $W_oO+b_o\one^\T$, so later scores depend on earlier reads. Together, $W_q,W_k,W_v,W_o$ and biases hold $4(D^2+D)$ numbers—{{value:attentionParams}} specimen parameters, replacing the $V^i(V-1)$ table for any $T$.

## In the code {#sec-code-attention}

Two functions implement the chapter. First, `attention` computes Steps 3–4 (with the mask of @sec-mask) for all heads and windows:

::: {.code file="src/attention.jl" fn="attention"}
:::

Second, `mix` performs Step 5, $O_h=V_h A_h^\T$, head by head:

::: {.code file="src/attention.jl" fn="mix"}
:::

Each returns its value and pullback `back`, mapping output adjoints to input adjoints (@ch-pulling-back).

## Exercises {.unnumbered}

::: exercise
Show that equal scores in row $i$ make $o_i$ the plain average of $v_1,\dots,v_i$. Which reader sees no difference between masked and unmasked softmax? For which reader do arithmetic, geometric, and mean-field predictions coincide identically?
:::

::: exercise
Show that adding a constant to row $i$ of $S$ leaves row $i$ of $A$ unchanged. From $q_i\cdot k_j$ in @sec-energy, deduce that $b_k$ never affects $A$ and has zero gradient, whereas $b_q$ can learn.
:::

::: exercise
In a one-block model with $P=0$, $\hat x_j$ depends only on the character at position $j$. Show that $o_i$ is invariant under permuting sources $1,\dots,i-1$. Which prefixes with the same last character give the same prediction? Which step of your argument fails in a second causal block?
:::

::: exercise
For bits $a,b\in\{0,1\}$, define $o(a,b)=(a+3^b b)/(1+3^b)$, with weights $(\tfrac12,\tfrac12)$ for $b=0$ and $(\tfrac14,\tfrac34)$ for $b=1$ (for instance, scores $0$ and $b\log 3$). Verify $ab=2a+3b-4\,o(a,b)$. Now suppose a decoder receives $(a,b,o(a,b))$: for any conditional with $0<P(Y=1\mid a,b)<1$, write its log-odds as $c_0+c_1a+c_2b+c_3ab$, substitute the identity for $ab$, and apply a binary Gibbs readout.
:::

::: exercise
Treating $v_j$ and raw scores $S_{ij}$ as independent arguments of one head, compute $\partial o_i/\partial v_j$ and $\partial o_i/\partial S_{ij}$. Express the second in terms of $A_{ij}$, $v_j$, and $o_i$: raising one score pulls $o_i$ towards $v_j$ by how much?
:::

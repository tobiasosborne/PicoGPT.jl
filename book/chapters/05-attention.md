# Attention {#ch-attention}

::: claim
Attention is the one map in the model through which positions talk to each other: column $i$ of its output is an average of value vectors from positions $j\le i$, weighted by a Gibbs distribution over query–key scores.
:::

::: names
| maths | physics | ML |
|---|---|---|
| row-stochastic, lower-triangular $A\in\R^{T\times T}$ | causal transition kernel | attention pattern |
| $o_i=\sum_j A_{ij}v_j$, a convex combination | expectation value $\langle v\rangle_i$ | attention output |
| $S_{ij}=q_i\cdot k_j/\sqrt d$ | negative energy | attention score, logit |
| $\sqrt d$ | temperature | scale factor |
| $A_{ij}=0$ for $j>i$ | causality | causal mask |
| $\R^D=\R^d\oplus\cdots\oplus\R^d$ | direct sum of subsystems | multi-head attention |
:::

## Where columns meet {#sec-meet}

Every map we have met so far acts on one column at a time. The embedding sends the character at position $t$ to the column $Ee_{x_t}+P_{:,t}$. LayerNorm, the MLP and the final linear map, which we meet in @ch-block, also act column by column, with the same parameters for every column. Each of them has the form
$$X=(x_1,\dots,x_T)\;\longmapsto\;\bigl(f(x_1),\dots,f(x_T)\bigr).$$
A model assembled only from such maps predicts the next character from the current character and its position, and from nothing else. It is a bigram model with a clock.

Attention is the exception. It is the only map in the model whose output column $i$ depends on input columns other than $i$. Tokens interact here and nowhere else. We assemble it in three steps: an average, a score, and a mask.

## An average of earlier columns {#sec-average}

Let $v_1,\dots,v_T\in\R^d$ be vectors, one per position, written as the columns of $V\in\R^{d\times T}$. Let $A\in\R^{T\times T}$ be *row-stochastic*: $A_{ij}\ge 0$, and every row sums to one. Then set
$$o_i=\sum_{j=1}^{T}A_{ij}\,v_j,\qquad\text{in matrix form}\qquad O=VA^\T\in\R^{d\times T}.$$
Row $i$ of $A$ is a probability distribution on the positions, and $o_i$ is the expectation of $v$ under it. Geometrically, $o_i$ lies in the convex hull of the $v_j$. The indices read as a sentence: *reader* $i$ reads *source* $j$ with weight $A_{ij}$.

::: aside
Rows are readers and columns are sources, everywhere in this book. Since tokens are columns, $O=VA^\T$ is the natural order of the factors; the transpose is the price of writing tokens as columns.
:::

@fig-attention-reads shows this for the trained specimen. Each character of the window is a reader. Its two heads, which we explain in @sec-heads, read with two different distributions: one is drawn above the line, the other below.

::: {.figure #fig-attention-reads window=11 reader=8}
What each character reads. The arcs leave the selected reader and land on the earlier characters it reads; the width of an arc is the weight $A_{ij}$, and the arcs of one head sum to one. Head 1 is drawn above the line and head 2 below. A loop is a character reading itself. [Drag the window along the line to change the text.]{.web}
:::

## Scores, and why a softmax {#sec-scores}

Where do the weights come from? Every position emits three vectors. Each is an affine image of the position's column $\hat x_i$ of the normalised stream:
$$q_i=W_q\hat x_i+b_q,\qquad k_i=W_k\hat x_i+b_k,\qquad v_i=W_v\hat x_i+b_v.$$
The *query* $q_i$ says what reader $i$ is looking for, the *key* $k_j$ says what source $j$ offers, and the *value* $v_j$ is what source $j$ hands over when it is read. A pair is scored by a scaled inner product, and the weights are the softmax of the scores over the sources:
$$S_{ij}=\frac{q_i\cdot k_j}{\sqrt d},\qquad A_{ij}=\frac{e^{S_{ij}}}{\sum_{j'\le i}e^{S_{ij'}}}.$$
In the language of @ch-energies, $-q_i\cdot k_j$ is an energy, $\sqrt d$ is a temperature, and row $i$ of $A$ is the Gibbs distribution
$$A_{ij}=\frac{1}{Z_i}\,e^{-E_{ij}/\tau},\qquad E_{ij}=-q_i\cdot k_j,\quad \tau=\sqrt d.$$
A source is read in proportion to the exponential of how well its key matches the query.

The score is a bilinear form on the stream. Write $W_q^{(h)},W_k^{(h)}\in\R^{d\times D}$ for the rows of $W_q,W_k$ that belong to one head (@sec-heads). Up to the bias terms, $q_i\cdot k_j=\hat x_i^\T\,W_q^{(h)\T}W_k^{(h)}\,\hat x_j$. The matrix $W_q^{(h)\T}W_k^{(h)}$ is a general $D\times D$ matrix of rank at most $d$, and nothing makes it symmetric: "what does $i$ look for in $j$" and "what does $j$ look for in $i$" are different questions.

::: {.figure #fig-attention-pipeline window=11 reader=8 head=1}
One reader and one head, from top to bottom: the keys $k_j$ of the sources, the reader's query $q_i$, the scores $S_{ij}=q_i\cdot k_j/\sqrt d$, the weights $A_{ij}$, and the values, each scaled by its weight; their sum is $o_i$. Cold rows read one source; hot rows read all of them equally. [Drag the temperature $\tau$ away from $\sqrt d$.]{.web}
:::

## Why $\sqrt d$ {#sec-sqrt-d}

Suppose the components of $q$ and $k$ are independent, with mean zero and variance one. Then $q\cdot k=\sum_{r=1}^{d}q_rk_r$ is a sum of $d$ independent terms of variance one, so its variance is $d$. Unscaled scores therefore grow like $\sqrt d$ as heads get wider. The softmax of numbers with spread $\sqrt d$ is a Gibbs distribution at inverse temperature of order $\sqrt d$: as $d$ grows it freezes onto its largest entry, and its derivative, which is all that learning ever sees (@ch-pulling-back), goes to zero. Dividing by $\sqrt d$ keeps the scores of order one whatever the head width. In the physical reading, $\sqrt d$ is the temperature at which the distribution is neither frozen nor uniform.

::: aside
The variance argument concerns initialisation, when $q$ and $k$ really are close to independent. After training, the scale of the scores is whatever the loss wanted; the $\sqrt d$ just sets a sensible starting temperature.
:::

## The causal mask {#sec-mask}

A language model predicts $x_{t+1}$ from $x_{\le t}$ (@ch-task). One forward pass makes all $T$ predictions of a window at once, and to keep them honest, position $i$ must not read any position $j>i$. We add the mask
$$M_{ij}=\begin{cases}0, & j\le i,\\ -\infty, & j>i,\end{cases}\qquad A=\softmax_{\text{rows}}(S+M).$$
Since $e^{-\infty}=0$, the upper triangle of $A$ vanishes, and $A$ is lower-triangular. Every row still sums to one, now over $j\le i$. The first reader can read only itself: $A_{11}=1$.

Lower-triangularity is causality written as a matrix. Column $i$ of the output depends on columns $1,\dots,i$ of the input and on nothing later, so the Jacobian of the map $X\mapsto O$ is block lower-triangular. One pass therefore yields $T$ conditional distributions, each conditioned on its own past. These are exactly the factors of the chain rule of probability, and the loss uses all of them.

::: {.figure #fig-attention-triangle window=11}
The attention matrices of the specimen's two heads for one window. Row $i$ is reader $i$ and column $j$ is source $j$; the area of a dot is $A_{ij}$. Nothing is drawn above the diagonal. The forward computation: scores for every pair, then the mask, then the softmax of each row. [Run the computation.]{.web}
:::

::: aside
The scores $S_{ij}$ are computed for every pair, the future included, and the trace stores them before the mask is applied. Half of them are then thrown away. For long contexts that is wasteful, and fast implementations never compute the upper triangle.
:::

## Heads: a direct sum {#sec-heads}

The specimen has $D=8$ and $H=2$ heads of width $d=D/H=4$. Split the space as $\R^8=\R^4\oplus\R^4$: rows $1$–$4$ of $Q$, $K$ and $V$ belong to head 1 and rows $5$–$8$ to head 2. Each head computes its own scores and its own pattern $A^{(h)}$ from its own block, and the outputs are stacked:
$$O^{(h)}=V^{(h)}A^{(h)\T}\in\R^{d\times T},\qquad O=\begin{pmatrix}O^{(1)}\\ O^{(2)}\end{pmatrix}\in\R^{D\times T}.$$
The output map $W_o$ then mixes the blocks, and the result is added to the residual stream,
$$X\;\longleftarrow\;X+W_oO+b_o\,\mathbf 1^\T.$$
Two heads let a position read two different things at once, and each pays for it with a smaller $d$. The four matrices $W_q,W_k,W_v,W_o$ and their biases hold $4(D^2+D)$ numbers, which is $288$ of the specimen's parameters.

## In the code {#sec-code-attention}

In the code the chapter is two functions. The first computes the patterns $A$ of every head and every window from $Q$ and $K$:

::: {.code file="src/attention.jl" fn="attention"}
:::

The second applies them, $O=VA^\T$, head by head:

::: {.code file="src/attention.jl" fn="mix"}
:::

Each returns its value together with its pullback, the function `back` that turns the adjoint of the output into adjoints of the inputs. The bars are the covectors of @ch-pulling-back, where these pullbacks are derived.

## Exercises {.unnumbered}

::: exercise
Show that if all the scores in row $i$ are equal, then $o_i$ is the plain average of $v_1,\dots,v_i$. Which reader sees no difference between the masked and the unmasked softmax?
:::

::: exercise
Show that adding the same constant to every score in a row leaves that row of $A$ unchanged. Deduce that the key bias $b_k$ has no effect on $A$, and hence that its gradient is exactly zero. (The model carries $d$ parameters per head that it cannot use.)
:::

::: exercise
Remove the positional code, so that $\hat x_j$ depends only on the character at position $j$. Show that $o_i$ is then unchanged by any permutation of the sources $1,\dots,i-1$. What does that say about a model without positions?
:::

::: exercise
Compute the Jacobian $\partial o_i/\partial v_j$ and $\partial o_i/\partial S_{ij}$ for one head. Write the second one in terms of $A_{ij}$, $v_j$ and $o_i$, and interpret it: raising one score pulls the output towards that source's value, by how much?
:::

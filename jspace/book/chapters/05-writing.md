# Reading and writing {#ch-writing}

::: claim
A J-lens vector can be read, as the single number $\inner{v_a}{x}$, or written, by adding a multiple of it to a column of the stream; the model follows the writing, later positions see it only through attention still ahead, and the J-space part of a concept moves the output more than the rest of it, for a reason that is partly arithmetic.
:::

::: names
| maths | physics | ML |
|---|---|---|
| $v^\sharp$, the vector with $\inner{v^\sharp}{y}=v(y)$ for every $y$ | raising an index with the metric | steering vector |
| $x\mapsto x+\alpha\,v_a^\sharp$ | an applied field of strength $\alpha$ | steering |
| $x\mapsto x-\inner{\hat v_a}{x}\,\hat v_a$, an orthogonal projection | removing one component | ablation |
| $c=V^+x$, least-squares coordinates in a plane | coordinates in a non-orthogonal basis | lens coordinates |
| $x\mapsto x+V(\sigma(c)-c)$ | exchanging two occupation numbers | patching in lens coordinates |
| $m_a=\E[x\mid\text{next}=a]-\E[x]$ | a conditional shift of the mean | concept vector |
:::

## One covector, two uses {#sec-two-uses}

@ch-jlens built, at every depth $\ell$, one covector per character: $v_a$, row $a$ of $W_{\text{out}}J_\ell$, the averaged gradient, with respect to the stream at depth $\ell$, of the score $s_a=\inner{w_a}{y}$ that row $a$ of the readout computes before the final LayerNorm (@ch-jlens). The model's logit of $a$ is that score after the LayerNorm, and its gradient carries the LayerNorm's derivative as well. A covector is a linear map $\R^D\to\R$. It eats a column and returns a number, and that is its first use: *reading*. Its second use, *writing*, needs a conversion first, and the conversion is a choice. This chapter examines the choice @ch-jspace made silently, writes with it on the specimen, and then asks the paper's question: is what we wrote special?

Everything here is an intervention on one column of one depth. We replace a column $x$ of $X_\ell$ by an edited $x'$ and run the rest of the model forward from depth $\ell$, exactly, through every LayerNorm, softmax and rectifier. Nothing in the rest of the model is linearised. The averaged Jacobian only chooses the direction of the edit; the model decides what the edit does.

## Reading: a probe {#sec-probe}

The lens of @ch-jlens reads all characters at once:
$$\operatorname{lens}_\ell(x)=\softmax\bigl(W_{\text{out}}\LN(J_\ell x)+b_{\text{out}}\bigr).$$
A *probe* reads one of them, $x\mapsto\inner{v_a}{x}=\sum_q(W_{\text{out}}J_\ell)_{aq}\,x_q$. It is the lens with the LayerNorm and the bias taken away, and what is taken away can be written down. LayerNorm centres its input with $C=I-\tfrac1D\one\one^\T$, divides by one scale $s(u)=\sqrt{\lVert Cu\rVert^2/D+\varepsilon}$, and applies the affine map $\gamma,\beta$ of the final norm:
$$\LN(u)=\gamma\odot\frac{Cu}{s(u)}+\beta .$$
So the lens's logit for $a$ is
$$z_a(x)=\frac{\inner{w_a}{x}}{s(J_\ell x)}+\bigl(W_{\text{out}}\beta+b_{\text{out}}\bigr)_a,$$
with the modified covector $w_a=J_\ell^\T C\diag(\gamma)\,W_{\text{out}}^\T e_a$.
The lens ranks characters by the probes of the modified covectors $w_a$, all divided by one positive number, each shifted by its own constant. The probe $\inner{v_a}{x}$ drops the centring, the gains $\gamma$, the common scale and the constants. It is cruder and simpler: linear in $x$, one character at a time, with a sign.

::: aside
In the paper, the Methods section "Technical details of J-lens use cases" uses both forms of reading: the full readout when it wants a ranked list of tokens, and the probe "when measuring whether a specific concept is present above a threshold". Its formula for the lens writes the normalisation as `norm` and omits the output bias; ours keeps the specimen's $b_{\text{out}}$.
:::

## Writing: steering {#sec-steer}

To add $v_a$ to a column we must first make it a column. The stream $\R^D$ has an inner product, and an inner product turns covectors into vectors: $v^\sharp$ is the vector with $\inner{v^\sharp}{y}=v(y)$ for every $y$. In coordinates nothing happens; the numbers of the row become the numbers of a column, which is why the step is easy to miss. It is nonetheless a choice. A different inner product on the stream would turn the same covector into a different vector: the paper's own extension to words of several tokens, the *template lens*, works in the inner product defined by the inverse covariance of the activations. With the plain one, $v_a^\sharp$ is the direction in which the averaged score of $a$ rises fastest per unit length, and the edit
$$x\;\longmapsto\;x+\alpha\,v_a^\sharp$$
is one step of gradient ascent on that averaged score, of length $\lvert\alpha\rvert\,\lVert v_a\rVert$. From here on we write $v_a$ for $v_a^\sharp$ when it is added to a column, as the paper does. The figures keep the distinction in their inks.

To first order, and in the averaged model, the step moves every score, not only the score of $a$:
$$ds_b=\alpha\,\inner{v_b}{v_a}=\alpha\sum_{q}(W_{\text{out}}J_\ell)_{bq}(W_{\text{out}}J_\ell)_{aq},$$
or, all characters at once, $ds=\alpha\,W_{\text{out}}J_\ell J_\ell^\T W_{\text{out}}^\T e_a$. These are first-order changes of the scores before the final LayerNorm; the model's logits are the scores after it, so their changes are these passed through the LayerNorm's derivative, which centres and rescales them.
Steering towards $a$ also raises every character whose lens vector makes an acute angle with $v_a$, in proportion to their inner product: the Gram matrix of the lens vectors is the matrix of these side effects. In the real model there is in addition everything the first order leaves out. @fig-j-steer shows what happens, column by column.

::: {.figure #fig-j-steer window=1 position=4 char=o}
Steering one column of the specimen towards one character. Each row is a depth; each small multiple is the model's distribution over the seven characters at the edited position after $x\leftarrow x+\alpha v_a$, with the bar of $a$ in full ink. The line through the tops of those bars is $p_a$ as a function of $\alpha$. The open ring after $\alpha=4$ is the limit for large $\alpha$, the readout of $v_a$ alone; the last multiple is the ablation, which sits on the same line at the point $\alpha^*$ marked on the axis. [Choose the column on the line, the character, and $\alpha$.]{.web}
:::

Three things can be read off it. The first is that $\alpha$ is not a strength. The step has length $\lvert\alpha\rvert\,\lVert v_a\rVert$, and the lens vectors are long at the early depths and short at the last: at depth $0$ their lengths run from {{value:wrNormMin0}} to {{value:wrNormMax0}} against columns of mean length {{value:wrColMean0}}, at depth $1$ from {{value:wrNormMinL}} to {{value:wrNormMaxL}} against {{value:wrColMeanL}}. A step of $\alpha=1$ is therefore a large edit early and a small one late, and the counts that follow measure the lengths of the lens vectors as much as the model: rescale $J_\ell$ and they move, while the lens does not. Count, over the $80$ columns and the six characters at each that are not already the model's choice, how often a step of $\alpha=1$ makes the steered character the model's choice: {{value:steerOneZero}} times at depth $0$, {{value:steerOneHalf}} at depth $\tfrac12$, and {{value:steerOneOne}} at depth $1$, where $J_1=I$ and the lens vectors are the rows of $W_{\text{out}}$ itself. At depth $1$ even $\alpha={{value:steerAlphaMax}}$ manages it {{value:steerFourOne}} times.

The second is that the curves level off, and some of them come back down. That is LayerNorm. Suppose the column is $x+\alpha v_a$ and $\alpha$ is large. Every map between depth $\ell$ and the readout either sits behind a LayerNorm, which sees the direction of its input and not its length, or is bounded; only the residual path carries $\alpha v_a$ to the end, undiminished. At the final LayerNorm the column is $\alpha v_a$ plus something of order one, and
$$\LN\bigl(\alpha v_a+O(1)\bigr)\;\longrightarrow\;\LN(v_a)\qquad(\alpha\to\infty),$$
with $\varepsilon$ dropped from the limit's LayerNorm. As $\alpha\to+\infty$ the output tends to $\softmax\bigl(W_{\text{out}}\LN(v_a)+b_{\text{out}}\bigr)$, the readout of the lens vector alone, whatever the column was and at whatever depth the edit was made, provided $v_a$ is not a multiple of $\mathbf 1$, which the centring would remove; as $\alpha\to-\infty$ the limit is the readout of $-v_a$. The attention scores stay bounded on the way, because queries and keys are made from normalised columns. The open rings in @fig-j-steer are this limit, computed from the readout's parameters. Where the lens vector, read by itself, does not say its own character, the curve rises, peaks and falls back towards the ring; in the column the figure opens with, the curve of ‘o’ at depth $0$ rises and must eventually fall towards its ring, beyond the range drawn. A large step along $v_a$ does not write "say $a$"; it writes $v_a$ over everything else, and the model reads $v_a$ as it reads any other column.

The third is the ablation. To *ablate* $a$ is to project the column onto the hyperplane on which the probe of $a$ reads zero:
$$x\;\longmapsto\;x-\inner{\hat v_a}{x}\,\hat v_a=(I-\hat v_a\hat v_a^\T)\,x,\qquad \hat v_a=v_a/\lVert v_a\rVert .$$
This is a point of the steering line. The probe reads $\inner{v_a}{x+\alpha v_a}=\inner{v_a}{x}+\alpha\lVert v_a\rVert^2$, which is zero at
$$\alpha^*=-\frac{\inner{v_a}{x}}{\lVert v_a\rVert^2},$$
so ablation is steering by exactly the amount that sets the probe to zero. It need not remove the character from the output. The probe is the averaged first-order picture; the model is neither averaged nor first order, and the last multiple of each row of @fig-j-steer shows what the model makes of the ablated column.

::: aside
Linear probes for what a network represents go back at least to [Alain and Bengio (2016)](https://arxiv.org/abs/1610.01644). Adding a fixed vector to the stream to change what a model writes is "activation addition" in [Turner et al. (2023)](https://arxiv.org/abs/2308.10248); projecting out one direction at every position is the "directional ablation" of [Arditi et al. (2024)](https://arxiv.org/abs/2406.11717). What is new in the paper is where the vectors come from: not from contrasting examples, but from the averaged derivative of the rest of the model.
:::

## Swapping two characters {#sec-swap}

The paper's sharpest intervention exchanges one concept for another and leaves the rest of the column alone. Choose two characters $s$ and $u$ and set their lens vectors side by side as the columns of a matrix
$$V=\begin{pmatrix}v_s & v_u\end{pmatrix}\in\R^{D\times2}.$$
(In this section $V$ is the paper's $D\times2$ matrix, not the size of the vocabulary.) The columns span a plane, $\Pi=\operatorname{span}\{v_s,v_u\}$. The *lens coordinates* of $x$ are the coefficients of the point of the plane nearest to $x$:
$$c=\arg\min_{c\in\R^2}\lVert x-Vc\rVert^2 .$$
The function is a quadratic in $c$, and its gradient vanishes at the minimum:
$$\nabla_c\lVert x-Vc\rVert^2=-2V^\T(x-Vc)=0,$$
so that
$$c=(V^\T V)^{-1}V^\T x=:V^+x .$$
Those two lines are the derivation of the *pseudoinverse* $V^+$. The $2\times2$ matrix $G=V^\T V$ holds the squared lengths of $v_s$ and $v_u$ on its diagonal and their inner product off it; it is invertible exactly when $v_s$ and $v_u$ are not parallel. (When they are, the code's `pinv` returns the shortest of the least-squares solutions.) The point $Vc=VG^{-1}V^\T x$ is the orthogonal projection of $x$ onto the plane.

The coordinates are not the probes. The probes are $V^\T x=Gc$, so reading and coordinates agree only when $G$ is the identity, that is when $v_s$ and $v_u$ are orthonormal. They never are here.

Now exchange the two coordinates, $\sigma(c_s,c_u)=(c_u,c_s)$, and put the column back together:
$$x'=x+V\bigl(\sigma(c)-c\bigr).$$
Write $x=Vc+(x-Vc)$, the point of the plane plus the part orthogonal to it. Then $x'=V\sigma(c)+(x-Vc)$: in the plane, the coordinates are exchanged; out of it, nothing changes, because the edit $V(\sigma(c)-c)$ lies in the plane. The plane is two of the $D$ dimensions, and whatever the column carries in the $D-2$ orthogonal to it is left exactly as it was; the part of everything that lies in the plane, position included, is rewritten. The construction needs $v_s$ and $v_u$ independent, so that $V^\T V$ is invertible; for a parallel pair the code's pseudoinverse picks the coordinates of least length, and the swap becomes a convention. That is the point of the construction. Two things it does not do. It does not leave the other characters' probes alone: $\inner{v_b}{x'}-\inner{v_b}{x}=\inner{v_b}{V(\sigma(c)-c)}$, which vanishes only for $v_b$ orthogonal to the plane. And it exchanges coordinates, not lengths: when $\lVert v_s\rVert\ne\lVert v_u\rVert$ the column moves by $\lvert c_s-c_u\rvert\,\lVert v_s-v_u\rVert$, which can be a long way. The paper also scales the edit, $x'=x+\alpha V(\sigma(c)-c)$: $\alpha=\tfrac12$ sets both coordinates to their mean, $\alpha=2$ overshoots the exchange.

An edit at position $t$ and depth $\ell$ can reach a later position $t'>t$ only through a map that mixes columns, and in a transformer those are the attention layers ([PicoGPT, *Attention*](https://tobiasosborne.github.io/PicoGPT.jl/attention/)). The specimen has one, between depth $0$ and depth $\tfrac12$. An edit at depth $0$ is read by it at every later position; an edit at depth $\tfrac12$ or $1$ has no attention ahead of it, and every later position is untouched, exactly. This is the triangle of @ch-jacobian seen from the other side: at depth $\tfrac12$ the blocks $K_{t't}$ with $t'>t$ are zero, and the future part of $J_{1/2}$ vanishes.

::: {.figure #fig-j-swap window=1 position=4 s=e u=t}
Swapping the lens coordinates of two characters in one column, at each depth, and running the model on. The slots are the positions of the window; at the edited position and every later one, the outlines are the model's distribution before the swap and the solid bars after it, with $s$ and $u$ in full ink. Each row gives the Gram matrix $G=V^\T V$ and the coordinates $c$ before and after the exchange. At depth $0$ the edit reaches later positions through attention; at depths $\tfrac12$ and $1$ nothing later moves. [Choose the column, $s$ and $u$.]{.web}
:::

The Shakespeare model has $D=128$, and the plane of a swap is two dimensions of them. The analysis tried every one of the $\binom{65}{2}=2080$ pairs on the prompt of @ch-jlens, swapping at depth {{value:gremioDepth}} in every column, and kept the pair whose swap changed the model's next character after the prompt and moved its distribution furthest. That pair is {{value:gremioPair}}. Unedited, the model expects {{value:gremioBefore}} after the prompt; with the swap at $\alpha=1$ it expects {{value:gremioAfter}}.

::: {.figure #fig-j-swap-gremio}
One swap in the Shakespeare model: the distribution of the character after the prompt, as the swap is scaled from $\alpha=0$ (the unedited model) to $\alpha=2$. Shown are the characters that reach the top four for some $\alpha$, the two swapped characters underlined; the last bar, marked …, is everything else. TV is the total variation distance from the unedited distribution.
:::

Read it for what it is. The pair was chosen for the size of its effect, among two thousand candidates, on one prompt; that the edit can move the prediction is shown, and that it moves it towards one of the two characters is shown at $\alpha=2$. It is not the paper's experiment, in which the model is asked to think of a sport and a swap of *Soccer* for *Rugby* makes it say *Rugby*. A model of characters has no sports to swap, and the closest it comes is a choice between letters.

## Is the J-space privileged? {#sec-privileged}

A model represents a concept by much more than its J-lens vectors. The paper asks whether the part of that representation that lies in the J-space is the part that drives what the model says. Its test, in the section "The J-space supports verbal report": take a *concept vector*, split it into its J-space component and the remainder, write each into the stream at the same length, and compare what they do.

Our concept vector for character $a$ at depth $\ell$ is the mean column before an $a$, minus the mean column:
$$m_a=\frac{1}{\lvert N_a\rvert}\sum_{n\in N_a}x_n-\frac1N\sum_{n=1}^{N}x_n,$$
where $N_a$ is the set of columns whose next character is $a$, among the $N=80$ columns of the specimen's ten windows. This $m_a$ is a next-character direction, the mean column before an $a$; the paper's concept vectors come from prompts about a concept, are not predictors of the next token, and have J-shares of a few per cent where ours, at the last depth, is nearly all. Split it as in @ch-jspace, $m_a=m_a^{J}+r_a$, with $m_a^J$ the nearest point of the J-space with $k=3$, found exactly by trying every subset. Then write each of $v_a$, $m_a^J$ and $r_a$, rescaled to the length $\lVert m_a\rVert$, into every column whose next character is not $a$, run the model on, and record the mean change of $\ln p_a$. @fig-j-privileged has the twenty-one triples.

::: {.figure #fig-j-privileged}
Writing three directions of equal length into the stream: the lens vector $v_a$, the J-component $m_a^J$ of the concept vector, and its remainder $r_a$. Each bar is the mean change of $\ln p_a$, in nats, over the columns whose next character is not $a$. Under each character: the share of $\lVert m_a\rVert^2$ that lies in the J-component, and $\cos(v_a,r_a)$. [Point at a character for its numbers.]{.web}
:::

The J-component moves the output about as much as the lens vector itself, and the remainder moves it less, in {{value:privilegedJWins}} of the {{value:privilegedCases}} cases. On its face this is the paper's result in miniature. Before taking it as evidence, look at the bottom line of the figure, because part of the result is arithmetic.

The J-component is an exact nonnegative least-squares fit, and such a fit leaves a residual that is orthogonal to every vector it uses. If a coefficient $c_i$ of the fit is positive, moving it a little in either direction keeps the fit feasible, so the derivative of the squared error in $c_i$ must vanish at the minimum:
$$\frac{\partial}{\partial c_i}\Bigl\lVert m_a-\sum_j c_jv_j\Bigr\rVert^2=-2\,\inner{v_i}{r_a}=0 .$$
So whenever $v_a$ is among the fit's vectors with a positive weight, $\inner{v_a}{r_a}=0$, and the remainder does not move the averaged score of $a$ at all, to first order: the score before the final LayerNorm, summed over the present and the later positions, in the averaged linear model. What @fig-j-privileged measures is the change of $\ln p_a$ at the edited position in the exact model, and between the theorem and the measurement sit the LayerNorm, the softmax, the present part of $J_\ell$, to which $r_a$ need not be orthogonal, and a write of length $\lVert m_a\rVert$, which is not small. It is the case in {{value:privilegedOrthogonal}} of the {{value:privilegedCases}} cases, where the figure prints a cosine of zero. The test therefore measures something narrower than it seems to: not a preference of the model for the J-space, but how well the averaged first-order picture, which predicts that the remainder does nothing to $a$, describes the model. That it describes it well is a finding, but it is the finding of @ch-jlens in a new costume.

The exceptions say the same thing from the other side. {{value:privilegedExceptions}} At depth $0$ the whole block is still ahead of the edit, and there the first-order picture is least reliable.

One more difference of shape. In the paper the J-component of a concept vector carries a median of 6–7 % of its squared length; in the specimen it carries far more (the percentages in the figure), because seven vectors in $\R^8$ with $k=3$ make cones that fill much of the space. The same orthogonality holds for any exact nonnegative fit, the paper's included, so a reader of the paper's experiment can keep it in mind. The paper adds a control of its own, clamping the J-lens coordinates so that the remainder's small effect cannot re-enter the J-space, and that control answers a different question.

::: aside
The conditions used here are the Karush–Kuhn–Tucker conditions of a problem with inequality constraints, from W. Karush's 1939 master's thesis and [Kuhn and Tucker (1951)](https://projecteuclid.org/euclid.bsmsp/1200500249). For nonnegative least squares they say: $\inner{v_i}{r}=0$ where $c_i>0$, and $\inner{v_i}{r}\le0$ where $c_i=0$. The active-set algorithm of the analysis, Lawson and Hanson's, is a search for the coefficients that satisfy them.
:::

## In the code {#sec-code-writing}

Every intervention of the chapter edits one column and continues the forward pass from the edited depth with the specimen's own maps (`continue_from`, built on PicoGPT's `attention`, `mix`, `layernorm` and the MLP). The swap is three lines:

::: {.code file="jspace/src/JLens.jl" fn="swap_coordinates"}
:::

`specimen_interventions` runs all of them: steering for every column, character and $\alpha\in\{-4,\dots,4\}$, the ablation, every swap of the $21$ pairs, and the concept vectors with their split.

## Exercises {.unnumbered}

::: exercise
Show that if $\gamma=\one$, $\beta=0$, $b_{\text{out}}=0$ and every row of $W_{\text{out}}$ sums to zero, then the J-lens ranks the characters exactly as their probes $\inner{v_a}{x}$ do. Which of these conditions does the specimen meet?
:::

::: exercise
Prove the saturation of steering for an edit at depth $1$ of the specimen: show that $\softmax(W_{\text{out}}\LN(x+\alpha v)+b_{\text{out}})$ tends to $\softmax(W_{\text{out}}\LN_0(v)+b_{\text{out}})$ as $\alpha\to\infty$, where $\LN_0$ is LayerNorm with $\varepsilon=0$. What happens if $v$ is a multiple of $\one$? What is the limit as $\alpha\to-\infty$?
:::

::: exercise
Show that the ablation is idempotent: ablating $a$ twice is ablating it once. Show that ablating $a$ and then $b$ gives the same column as ablating $b$ and then $a$ if and only if $v_a$ and $v_b$ are orthogonal or parallel.
:::

::: exercise
Let $x'$ be the swap of $x$ in the plane of $v_s,v_u$. Show that the swap of $x'$ is $x$ again, and that $\lVert x'-x\rVert=\lvert c_s-c_u\rvert\,\lVert v_s-v_u\rVert$. Show that the probes after the swap are $G\sigma(c)$, and that the swap exchanges the two probes, $\inner{v_s}{x'}=\inner{v_u}{x}$ and $\inner{v_u}{x'}=\inner{v_s}{x}$, if $\lVert v_s\rVert=\lVert v_u\rVert$.
:::

::: exercise
For the nonnegative least-squares fit $\min_{c\ge0}\lVert x-Ac\rVert^2$, show that the residual $r=x-Ac$ satisfies $\inner{a_i}{r}=0$ for every column with $c_i>0$ and $\inner{a_i}{r}\le0$ for every column with $c_i=0$. The J-component with $k=3$ is the best such fit over all subsets of three lens vectors. What do these conditions say about the lens vectors that are not in the chosen subset?
:::

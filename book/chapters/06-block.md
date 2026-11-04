# The block {#ch-block}

::: claim
A block makes two corrections to the stream of columns, $X\leftarrow X+f(X)$, each of the form of one step of Euler's method: in the first the tokens read from each other, in the second each token computes alone. Each correction reads a copy of the stream that LayerNorm has centred and scaled to just inside a sphere.
:::

::: names
| maths | physics | ML |
|---|---|---|
| $X\in\R^{D\times N}$, updated by $X\leftarrow X+f(X)$ | the state, evolved in depth | residual stream |
| $x_{k+1}=x_k+f_k(x_k)$ | an explicit Euler step | residual connection, skip connection |
| $\Pi=I-\one\one^\T/D$ | projection out of the uniform mode | mean subtraction, centring |
| $\hat x=\Pi x/\sigma$, $\lVert\hat x\rVert\lesssim\sqrt D$ | projection to (just inside) a sphere | layer normalisation, LayerNorm |
| $y=\gamma\odot\hat x+\beta$ | a diagonal affine map | learned gain and bias, "scale and shift" |
| $y=\sum_k r_k\,(W_2)_{:,k}+b_2$, $r=\relu(W_1\hat f+b_1)$ | a sum of switched sources | MLP, feed-forward layer |
| $\theta\in\R^p$ | a point in configuration space | the weights, the model |
:::

## The stream {#sec-stream}

After the embedding of @ch-vectors the model holds a matrix $X_0\in\R^{D\times N}$, one column per token. Everything that follows keeps this shape until the very last map. A block is a map $\R^{D\times N}\to\R^{D\times N}$, and it is built in a particular way: it never replaces $X$, it adds to it. With $\LN_1$, $\LN_2$ the two normalisations of @sec-layernorm, the block is
$$X_{\text{mid}}=X+\operatorname{Attn}\bigl(\LN_1(X)\bigr),\qquad X'=X_{\text{mid}}+\operatorname{MLP}\bigl(\LN_2(X_{\text{mid}})\bigr),$$
where $\operatorname{Attn}$ is the attention of @ch-attention together with its output map $W_o\,\cdot+b_o$, and $\operatorname{MLP}$ is the two-layer map of @sec-mlp. Each line reads the current state, computes a *branch*, and adds the branch back. The matrix being corrected is called the *residual stream*. It is the model's one working memory: the embedding writes the first state into it, every branch reads a normalised copy of it and writes an increment into it, and at the end the readout reads the last state.

::: aside
In the code the two residual sums are the lines `X₁ = X₀ + Y₁` and `X₂ = X₁ + Y₂` of `block` (@sec-block-code). Their pullbacks, `X̄₁ .+= X̄₂` and `X̄₀ .+= X̄₁`, are the road for the gradient described below.
:::

Each line has the shape of a familiar formula. For an ordinary differential equation $\dot x=f(s,x)$ on a grid $s_{k+1}=s_k+h$, the explicit Euler method is
$$x_{k+1}=x_k+h\,f(s_k,x_k).$$
A residual update $X_{k+1}=X_k+f_k(X_k)$ is exactly an update of this form, with step $h=1$ and a vector field $f_k$ that is different at every step, since every branch has parameters of its own. The specimen has $L=1$ block and therefore two steps; a model with $L$ blocks takes $2L$. Depth plays the role of time, and the stream is the state being evolved rather than a state discarded and rebuilt at every layer.

The analogy is a statement about the form of the update, and no more. Nothing in the architecture makes the steps small, or makes $f_k$ depend smoothly on $k$, so the model is not a discretisation of some underlying flow and nothing converges to one as layers are added. What the form does give is a default. A branch that outputs zero leaves the stream untouched, so a block can only *refine* the state it is handed; a freshly initialised model whose branches are small is close to the identity, and the embedding survives it intact. The other gift arrives in @ch-pulling-back: the derivative of $x\mapsto x+f(x)$ is $I+Df$, and the $I$ is a road along which the gradient travels back without passing through any branch.

::: aside
Residual connections come from image recognition: He, Zhang, Ren and Sun, *Deep residual learning* (2016). The name records the idea that a branch need only learn the *residual* $f(x)=\text{target}-x$, a correction to the identity, rather than the whole map. Reading the stack of updates as an Euler scheme goes back to E (2017) and to Haber and Ruthotto (2017); Chen, Rubanova, Bettencourt and Duvenaud (2018) took the limit seriously and made the depth continuous ("neural ODEs"). The transformer of this book does not.
:::

@fig-residual-stem draws the whole model in this picture. Each token is a vertical stem along which its column flows downwards. The two branches of the block leave the stem, compute, and merge back at a $+$. All the maps act on one column at a time, except attention, which reaches across to the earlier stems: it is still the only place where tokens meet.

::: {.figure #fig-residual-stem window=1 column=8}
The whole model for one window, one stem per token. The stream runs down each stem; the attention branch and the MLP branch leave it, compute, and merge back at a $+$. The width of the stem is the size of the column, $\lVert x\rVert/\sqrt D$, and the width of a branch is the size of the update it adds. Only the attention branch reaches across to other stems: an arc is the part of a later token's update that an earlier token hands it. On the left, the selected column is drawn with its vectors. Run the forward pass, or click a stem.
:::

## LayerNorm {#sec-layernorm}

The branches do not read the stream itself. Each reads a copy that has been normalised, column by column, by a map called *layer normalisation*, or LayerNorm. For one column $x\in\R^D$ define
$$\mu=\frac{\one^\T x}{D},\qquad u=x-\mu\one=\Pi x,\qquad \sigma=\sqrt{\frac{\lVert u\rVert^2}{D}+\varepsilon},\qquad \hat x=\frac u\sigma,\qquad y=\gamma\odot\hat x+\beta,$$
with $\Pi=I-\one\one^\T/D$, a fixed $\varepsilon=10^{-5}$, and two learned vectors $\gamma,\beta\in\R^D$. In index form, $\hat x_i=(x_i-\mu)/\sigma$ and $y_i=\gamma_i\hat x_i+\beta_i$. The mean $\mu$ and the variance $\sigma^2-\varepsilon$ are taken over the $D$ entries of one column, never across tokens, and the variance divides by $D$, not by $D-1$: it is the mean square of the entries of $u$, not an estimate of anything. Each column is normalised on its own, with the same $\gamma$ and $\beta$ for all of them.

Read geometrically, LayerNorm does three things.

*It centres.* $\Pi$ is symmetric and $\Pi^2=\Pi$, its kernel is the line spanned by $\one$ and its image is the hyperplane $\one^\perp$ of vectors whose entries sum to zero. So $u=\Pi x$ is the orthogonal projection of $x$ onto $\one^\perp$: the component of $x$ along $\one$, the *uniform mode* in which every entry moves together, is removed and nothing else is touched.

*It scales.* If $\varepsilon$ were zero, $\hat x=\sqrt D\,u/\lVert u\rVert$ would lie on the sphere of radius $\sqrt D$ in $\one^\perp$; equivalently, the root mean square of its entries would be exactly 1. With $\varepsilon>0$,
$$\lVert\hat x\rVert^2=\frac{\lVert u\rVert^2}{\sigma^2}=D\Bigl(1-\frac{\varepsilon}{\sigma^2}\Bigr)<D,$$
so the column lands *just inside* that sphere, by an amount that depends on the column: the shorter $u$ was, the deeper it lands. The point of $\varepsilon$ is the column with $u=0$, whose entries are all equal; there $\hat x=0$ instead of $0/0$.

*It applies a diagonal affine map.* $y=\gamma\odot\hat x+\beta$ stretches axis $i$ by $\gamma_i$ and shifts it by $\beta_i$. These $2D$ numbers are learned, and they take the column off the sphere again: what the branch receives lies on (just inside) an ellipsoid, the image of the sphere under $\diag(\gamma)$, shifted by $\beta$. The sphere is an intermediate stage, and the gain and offset give back, in a controlled way, a scale and a centre of the model's own choosing.

::: aside
There is an exact sphere behind $\varepsilon$, one dimension up. Append the number $\sqrt{D\varepsilon}$ to $u$, scale the vector $(u,\sqrt{D\varepsilon})\in\R^{D+1}$ to length $\sqrt D$, and drop the last entry: what remains is precisely $\hat x$. The missing coordinate holds the length by which $\hat x$ falls short.
:::

Centring and scaling do not change the direction of $u$. Up to the tiny effect of $\varepsilon$, what LayerNorm keeps of a column is the unit vector $u/\lVert u\rVert$ and nothing else; it forgets two numbers, the component along $\one$ and the length of what is left. @fig-layernorm-sphere draws exactly those two numbers. Each column $x$ becomes the point
$$(a,r)=\Bigl(\frac{\one^\T x}{\sqrt D},\ \lVert\Pi x\rVert\Bigr),\qquad a^2+r^2=\lVert x\rVert^2,$$
in a half-plane whose horizontal axis is the direction $\one/\sqrt D$ and whose vertical axis is distance from it. The circle of radius $\sqrt D$ is the sphere $\lVert x\rVert=\sqrt D$. Centring moves every point horizontally onto the vertical axis; scaling moves it along that axis to just below the top of the circle; and since this picture records only the two forgotten numbers, every column of the window arrives at the same place in it.

::: {.figure #fig-layernorm-sphere window=1 column=4}
LayerNorm, geometrically, for the second normalisation of the block, the one in front of the MLP. On the left every column of the window is a point $(\one^\T x/\sqrt D,\ \lVert\Pi x\rVert)$; centring moves it across onto the vertical axis, scaling moves it along that axis to just inside the circle $\lVert x\rVert=\sqrt D$, and the gain and offset move it off again. The inset magnifies how far inside each column lands. On the right, one column as a spoke glyph at each stage, with the dotted circle at root mean square 1. Run the stages, or click a column.
:::

Why normalise at all? The stream accumulates: every branch adds to it, and nothing in the forward pass ever shrinks it. A branch that read the raw stream would see inputs whose size depends on how much has been written before, and would have to work at every scale. A normalised copy has a fixed size, so each branch sees its input on the same footing, whatever its depth. The price is that a branch cannot see the mean or the length of a column; whatever the model wants a branch to know must be written in the direction. In @ch-pulling-back the same geometry reappears in the derivative: the pullback of LayerNorm projects the incoming adjoint onto $\one^\perp$ and onto the tangent space of the sphere, so no gradient ever asks to change what LayerNorm forgets.

::: aside
Ba, Kiros and Hinton introduced layer normalisation in 2016. The original transformer (Vaswani et al., 2017) normalised *after* each residual sum, $\LN(x+f(x))$, which normalises the stream itself. GPT-2 (Radford et al., 2019) moved the normalisation to the input of each branch, as here, and added one more before the readout; this "pre-norm" arrangement leaves the stream un-normalised, and is the one most models now use.
:::

## The MLP {#sec-mlp}

The second branch is the same map for every column, a composition of an affine map, an entrywise nonlinearity, and another affine map. For the column $\hat f=\LN_2(x_{\text{mid}})\in\R^D$,
$$u=W_1\hat f+b_1\in\R^F,\qquad r=\relu(u),\qquad y=W_2r+b_2\in\R^D,$$
with $W_1\in\R^{F\times D}$, $W_2\in\R^{D\times F}$, $b_1\in\R^F$, $b_2\in\R^D$ and the *rectifier* $\relu(u)_k=\max(u_k,0)$. The width $F$ of the middle is $4D$ by convention, 32 in the specimen. On all the columns at once this reads $Y=W_2\relu(W_1\hat F+b_1\one^\T)+b_2\one^\T$, and the stream becomes $X'=X_{\text{mid}}+Y$. ML calls the branch a *multilayer perceptron* or *feed-forward layer*, and its $F$ middle coordinates *hidden units*.

The map is clearest read through the columns of $W_2$. Write $w_k\in\R^D$ for row $k$ of $W_1$; then
$$y=\sum_{k=1}^{F}\max\bigl(w_k\cdot\hat f+b_{1k},\,0\bigr)\;(W_2)_{:,k}\;+\;b_2.$$
Unit $k$ owns a fixed vector, the column $(W_2)_{:,k}\in\R^D$, and a detector, the half-space $w_k\cdot\hat f+b_{1k}>0$. When the input lies in the half-space the unit *fires* and adds its vector, scaled by how far inside the input lies; otherwise it adds nothing at all. The MLP is a sum of $F$ switched sources. On each region of $\R^D$ where the set of firing units does not change, it is an affine map, so the whole branch is continuous and piecewise affine, with flat pieces glued along the $F$ hyperplanes $w_k\cdot\hat f+b_{1k}=0$. The rectifier is the only nonlinearity of this kind in the model; the others are the softmax and the normalisation.

::: {.figure #fig-mlp window=1 column=4}
The MLP of one column. The normalised column $\hat f$ is mapped to $F$ numbers $u_k$, drawn about a zero line; the rectifier keeps the positive ones and sets the others to zero. Below each firing unit is what it adds, $r_k$ times its column of $W_2$, on one scale with their sum, the update $y$ at the foot. Click a column, or point at a unit.
:::

Unlike attention, the MLP never looks at another column: a token computes here alone, with the same $W_1$, $W_2$ for every position. Between them, $W_1$ and $W_2$ hold more than half of the specimen's parameters. Attention decides *where* to read; the MLP is where most of the model's fixed, learned content sits, a table of $F$ vectors indexed by $F$ half-spaces.

::: aside
The rectifier is an ideal diode: a small change of $u_k$ passes through when the unit fires and is blocked when it does not. At $u_k=0$ exactly it has no derivative; the code uses the slope $0$ there, a convention rather than a theorem (@ch-pulling-back). The perceptron is Rosenblatt's (1958); the transformer paper calls this branch the "position-wise feed-forward network".
:::

## One composition {#sec-composition}

We can now write the whole model down. With $C$ the one-hot matrix of the input characters and $B$ the block,
$$\ell(\theta)=\operatorname{CE}\Bigl(\softmax\bigl(W_{\text{out}}\,\LN_f\bigl(B(EC+P)\bigr)+b_{\text{out}}\one^\T\bigr),\ y\Bigr).$$
Read from the inside out: embed, run the block, normalise once more, map each column to $V$ logits with the *readout* $Z=W_{\text{out}}X_f+b_{\text{out}}\one^\T$, turn each column of logits into the Gibbs distribution of @ch-energies, and charge the cross-entropy of @ch-surprise against the true next characters $y$. A model with $L$ blocks has $B_L\circ\cdots\circ B_1$ in place of $B$. Every arrow in @fig-composition is one of these maps.

::: {.figure #fig-composition}
The specimen as one composition of maps, from characters to a single number. Each arrow is a map, with the parameters it uses drawn as squares of area proportional to how many numbers they hold; each node is a space, labelled with the shape of its array for one window. Only attention acts across columns.
:::

All the parameters together are one vector $\theta\in\R^p$. Counting along the arrows,
$$\begin{aligned}p&=\underbrace{DV}_{E}+L\bigl(\underbrace{4D^2+4D}_{\text{attention}}+\underbrace{2DF+F+D}_{\text{MLP}}+\underbrace{4D}_{\LN_1,\,\LN_2}\bigr)+\underbrace{2D}_{\LN_f}+\underbrace{DV+V}_{\text{readout}}\\&=2DV+V+2D+L\,(4D^2+2DF+9D+F),\end{aligned}$$
which for the specimen is $p=1007$. Neither the context length $T$ nor the number of heads $H$ appears: positions are a fixed code and heads only partition rows that are already counted. In the code the named parameters $E,\gamma_1,\dots,b_{\text{out}}$ are *views* onto one vector, in a fixed order, so that an optimiser in @ch-learning sees nothing but a point in $\R^p$ and a vector of the same length to move it by.

::: aside
The order is binding: $E$; then for each block $\gamma_1$ $\beta_1$ $W_q$ $b_q$ $W_k$ $b_k$ $W_v$ $b_v$ $W_o$ $b_o$ $\gamma_2$ $\beta_2$ $W_1$ $b_1$ $W_2$ $b_2$; then $\gamma_f$ $\beta_f$ $W_{\text{out}}$ $b_{\text{out}}$; each matrix column by column. `carve(θ, c)` in `src/parameters.jl` cuts $\theta$ into these views, and `flatten` glues the gradients back together in the same order.
:::

Running this composition pushes values forward through every map, top to bottom. Training will need the opposite: the derivative of the one number at the bottom with respect to all $p$ numbers at the side. That is the subject of @ch-pulling-back.

## In the code {#sec-block-code}

The block is written from four functions. The first two are the affine map, applied to every column with the same weights, and the rectifier:

::: {.code file="src/block.jl" fn="affine"}
:::

::: {.code file="src/block.jl" fn="relu"}
:::

LayerNorm computes $\mu$, $\sigma$ and $\hat x$ for every column at once; its pullback is derived in @ch-pulling-back.

::: {.code file="src/block.jl" fn="layernorm"}
:::

The block itself is the two residual sums, and its pullback reads the same lines backwards:

::: {.code file="src/block.jl" fn="block"}
:::

The whole model is the embedding, $L$ blocks, a final LayerNorm and the readout:

::: {.code file="src/model.jl" fn="gpt"}
:::

Every function returns its value together with its pullback, `back`. The forward pass is the first half of each function; @ch-pulling-back is about the second.

## Exercises {.unnumbered}

::: exercise
Show that $\Pi=I-\one\one^\T/D$ is symmetric and idempotent, with kernel $\operatorname{span}(\one)$ and image $\one^\perp$. Then show $\lVert\hat x\rVert^2=D(1-\varepsilon/\sigma^2)$, and deduce that, for $\lVert u\rVert^2\gg D\varepsilon$, the column lands a distance $\sqrt D-\lVert\hat x\rVert\approx\varepsilon D^{3/2}/(2\lVert u\rVert^2)$ inside the sphere.
:::

::: exercise
Show that for $\varepsilon=0$, $\LN(cx+b\one)=\LN(x)$ for every $c>0$ and $b\in\R$. What does LayerNorm do to $-x$? With $\varepsilon>0$, which of these invariances survive exactly?
:::

::: exercise
Let $S\subseteq\{1,\dots,F\}$ be the set of units that fire at $\hat f$, and $P_S$ the diagonal matrix with ones at the positions in $S$. Show that near $\hat f$ (away from the hyperplanes) the MLP is the affine map $\hat f\mapsto W_2P_SW_1\hat f+W_2P_Sb_1+b_2$. What is its rank at most? Show that the whole branch is continuous.
:::

::: exercise
Try it in @fig-layernorm-sphere. Which column of the window lands deepest inside the sphere? Check your answer against the first exercise, using $\lVert\Pi x\rVert$ from the readout.
:::

::: exercise
Try it in @fig-mlp. Click through the eight columns of the window. Is there a unit that fires for every one of them, and one that fires for none? What does a unit that never fires contribute to the window, and what will the gradient of its weights be (@ch-pulling-back)?
:::

::: exercise
Derive the parameter count $p=2DV+V+2D+L(4D^2+2DF+9D+F)$ from @fig-composition, and evaluate it for the Shakespeare model of @ch-speaking, with $V=65$, $D=128$, $F=512$ and $L=4$. Which single kind of array dominates as $D$ grows?
:::

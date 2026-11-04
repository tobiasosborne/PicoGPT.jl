# Pulling back {#ch-pulling-back}

::: claim
The derivative of the loss with respect to every parameter costs about as much as one forward pass: start at the loss with the number one, and pull it back through the transpose of each map's derivative, in reverse order, adding wherever a value was used twice.
:::

::: names
| maths | physics | ML |
|---|---|---|
| $Df(x)$, pushing $dx$ forward to $dy$ | linear response | Jacobian, forward-mode derivative |
| $\bar x=Df(x)^\T\bar y$ | the adjoint map | backpropagation through $f$, vector–Jacobian product |
| $\bar x=\partial\ell/\partial x$, a covector | a generalised force, conjugate to $x$ | the gradient, "grad" |
| $(f_k\circ\cdots\circ f_1)^*=f_1^*\circ\cdots\circ f_k^*$ | the adjoint method | backward pass, reverse-mode differentiation |
| $\Delta^*(\bar u,\bar v)=\bar u+\bar v$ | responses superpose | gradient accumulation |
| $\bigl(\ell(\theta+he_i)-\ell(\theta-he_i)\bigr)/2h$ | a symmetric perturbation experiment | numerical gradient, gradient check |
:::

## One number, many parameters {#sec-one-number}

Fix the training windows. Then the model and its loss are one function of the parameters,
$$\ell:\R^p\to\R,\qquad \theta\longmapsto\text{the mean surprise of the targets},$$
with $p=1007$ for the specimen. Learning, in @ch-learning, moves $\theta$ downhill, and for that it needs the gradient $\nabla\ell(\theta)\in\R^p$: for every parameter, how fast the loss changes when that parameter alone is nudged.

The obvious way to get it is to nudge. For each coordinate $i$,
$$\frac{\partial\ell}{\partial\theta_i}\approx\frac{\ell(\theta+h e_i)-\ell(\theta-he_i)}{2h},$$
which costs two evaluations of $\ell$, that is two forward passes, per parameter: $2p=2014$ for the specimen and about 1.6 million for the Shakespeare model of @ch-speaking, every time the parameters move. It is also only approximate, and we shall see in @sec-gradcheck that its accuracy cannot be pushed very far.

This chapter derives the other way. One forward pass computes the loss and records its intermediate values. One backward pass then computes all $p$ derivatives exactly (up to rounding), at a cost of a small constant number of forward passes, whatever $p$ is. The backward pass is not a new idea; it is the chain rule, read from the end. What makes it cheap is the order in which the chain rule's factors are multiplied, and what makes it simple is that each factor is a transpose.

It helps to be careful about what kind of object the gradient is. The derivative of $\ell$ at $\theta$ is a linear map $d\ell(\theta):\R^p\to\R$, which eats a displacement $h$ of the parameters and returns the first-order change of the loss. A linear map to the numbers is a *covector*. On $\R^p$ with its usual inner product every covector is $h\mapsto\langle g,h\rangle$ for exactly one vector $g$, and that vector is called the gradient. So $\nabla\ell$ is a covector wearing the clothes of a vector. It answers the question "how much does the loss change if I move this way?", not "where is the parameter?". This is the distinction the book's two inks draw: values, which are pushed forward, in value ink; and covectors, which are pulled back from the loss, in covector ink.

## The pullback {#sec-pullback}

Let $f:\R^n\to\R^m$ be differentiable, $y=f(x)$. Its derivative at $x$ is the linear map $Df(x):\R^n\to\R^m$ with
$$f(x+dx)=f(x)+Df(x)\,dx+o(\lVert dx\rVert).$$
It pushes a small change of the input forward to the change of the output, $dy=Df(x)\,dx$. This is the forward direction, the direction in which the model computes.

Now suppose that $y$ is used, further on, to compute a loss, and that we know how the loss responds to $y$: a covector $\bar y\in\R^m$ with $d\ell=\inner{\bar y}{dy}$. Then
$$d\ell=\inner{\bar y}{Df(x)\,dx}=\inner{Df(x)^\T\bar y}{dx},$$
and the response of the loss to $x$ is therefore
$$\boxed{\;\bar x=Df(x)^\T\,\bar y\;},\qquad \bar x_j=\sum_{i=1}^m\bar y_i\,\frac{\partial y_i}{\partial x_j}.$$
The map $\bar y\mapsto\bar x$ is the *pullback* of $f$ at $x$. It runs against the computation, from the output space back to the input space, and it is linear. The bar is the book's notation for it: $\bar x=\partial\ell/\partial x$ has the same shape as $x$, and is read "x-bar", or the adjoint of $x$. In matrix form the pullback is the transpose of the Jacobian; in practice the Jacobian is never formed, and the pullback is a function that computes $Df(x)^\T\bar y$ directly, as cheaply as $f$ itself.

Compose. The model is a chain of maps, $\ell=f_k\circ\cdots\circ f_1$, the last of which returns a single number. With $J_i$ the derivative of $f_i$ at its own input, the chain rule says $D\ell=J_kJ_{k-1}\cdots J_1$, and transposing reverses the order:
$$\nabla\ell=(J_k\cdots J_1)^\T\cdot1=J_1^\T\Bigl(J_2^\T\bigl(\cdots(J_k^\T\cdot1)\cdots\bigr)\Bigr).$$
Read the brackets from the inside out. Start at the loss with $\bar\ell=1$; apply the pullback of the last map, then of the one before it, and so on back to the parameters. Every intermediate result is a covector of the size of the value it belongs to, and no step multiplies two matrices. The backward pass is this bracket, evaluated.

The same product can be bracketed the other way, $J_k(\cdots(J_2(J_1v)))$: push one direction $v$ forward through the maps. That is *forward-mode* differentiation; it yields one directional derivative $D\ell\,v$ per pass, so the whole gradient takes $p$ passes, one per coordinate, no better than finite differences in count. Reverse mode yields the whole gradient in one pass because the chain ends in one number. For a map with many outputs and few inputs the comparison would go the other way.

::: aside
Reverse-mode differentiation was described by Seppo Linnainmaa in his 1970 master's thesis, as a way of computing the rounding error of a long calculation. Paul Werbos proposed it for neural networks in 1974; Rumelhart, Hinton and Williams made it famous as *backpropagation* in 1986. Physicists know it as the adjoint method: the costate of Pontryagin's optimal control, and the adjoint equations of weather forecasting, which pull the sensitivity of one forecast number back through a whole simulation.
:::

How much does it cost? Each map of the model is a matrix product, an elementwise function, a sum, a softmax or a lookup, and the pullback of each costs at most a small constant times the map itself; we derive them all below. Adding up, a forward and a backward pass together cost a constant number of forward passes, independently of $p$. In the web edition, @fig-gradcheck measures the constant in your browser. The price is memory: the pullback of a map needs the values that went into it, so the forward pass must keep its intermediates until the backward pass has used them.

In the code, every operation returns its value *together with* its pullback, a function `back` that has kept what it needs. The loss itself is two such operations, and its pullback is the composition of theirs, in reverse order:

::: {.code file="src/model.jl" fn="loss"}
:::

@fig-forward-backward shows the whole of it on the specimen's stems: the values flow down, then one covector per token flows back up the same paths.

::: {.figure #fig-forward-backward window=2 column=8}
Forward, then back. The forward pass flows down the stems in value ink; then the loss sends one covector up each stem, through the same maps in reverse order, in covector ink. The size of each adjoint is drawn where it lands. At every $+$ the adjoint goes both ways, and at every fork the adjoints coming back add. Scrub the clock, or click a stem. Then switch to the trained model: only the predictions whose context is ambiguous still send anything back.
:::

## Fan-out: adjoints add {#sec-fan-out}

A chain is the simplest program. The model is not quite a chain: some values are used more than once. The rule for that case looks like the one rule of the backward pass that is not a transpose. It is one too.

Using a value twice is applying the duplication map $\Delta(x)=(x,x)$. Its derivative duplicates perturbations, $D\Delta\,h=(h,h)$, and
$$\inner{(\bar u,\bar v)}{(h,h)}=\inner{\bar u+\bar v}{h},\qquad\text{so}\qquad \Delta^*(\bar u,\bar v)=\bar u+\bar v.$$
*The adjoint of a value used several times is the sum of the adjoints of its uses.* A responsive quantity has several routes to the loss, and their responses superpose. Three places in the model use this.

**The residual stream.** A residual update $y=x+f(x)$ uses $x$ twice, once on the identity path and once in the branch. Its pullback is
$$\bar x=\bar y+Df(x)^\T\,\bar y.$$
The first term is the adjoint travelling straight back along the stream; the second is the same adjoint pulled through the branch. The backward pass of a residual network has a residual stream of its own, and it is the transpose of the forward one: the linearised update $dX_{k+1}=(I+Df_k)\,dX_k$ of the Euler steps of @sec-stream has the adjoint $\bar X_k=\bar X_{k+1}+Df_k^\T\bar X_{k+1}$. The identity gives every branch a direct line to the loss. It does not, by itself, keep the product of many such factors from growing or shrinking.

**One input, three maps.** In attention the normalised column $\hat x$ feeds the query, key and value maps. Its adjoint is the sum of the three pullbacks,
$$\bar{\hat x}=W_q^\T\bar q+W_k^\T\bar k+W_v^\T\bar v.$$

**Parameters used at every token.** A bias is added to every column, $Y=WX+b\one^\T$, so its adjoint is a sum over the columns, $\bar b=\bar Y\one$: a sum, not a mean (the single factor $1/N$ of the mean lives in the loss). The embedding is the extreme case. Column $a$ of $E$ is read once for every occurrence of the character $a$, so
$$\bar E_{:,a}=\sum_{n\,:\,x_n=a}\bar X_{0,:,n},$$
the adjoints of all the positions where $a$ occurs, added into one column. This is called a scatter-add. Overwriting instead of adding is a classic bug: it throws away all but one observation of each character.

::: aside
In the code the two residual adjoints are the lines `X̄₁ .+= X̄₂` and `X̄₀ .+= X̄₁` in the pullback of `block`, and the three-way sum is `back_Â(Â̄q + Â̄k + Â̄v)` (@sec-block-code). The residual lines are marked with a comment.
:::

## The pullbacks, one by one {#sec-pullbacks}

Every map in the model is one of six kinds. Here is the pullback of each, derived by the recipe of @sec-pullback: write the differential $dy$, pair it with $\bar y$, and move everything onto $dx$. Matrices carry the Frobenius inner product $\inner{U}{V}=\operatorname{tr}(U^\T V)=\sum_{ij}U_{ij}V_{ij}$, and the cyclic property of the trace does the moving.

### Affine maps

$Y=WX+b\one^\T$ with $W\in\R^{r\times s}$, $X\in\R^{s\times N}$, $b\in\R^r$: the readout, the query, key, value and output maps, and both layers of the MLP. The differential is $dY=dW\,X+W\,dX+db\,\one^\T$, and
$$\inner{\bar Y}{dY}=\inner{\bar YX^\T}{dW}+\inner{W^\T\bar Y}{dX}+\inner{\bar Y\one}{db}.$$
Read off the three adjoints, in matrix form and in index form:
$$\bar W=\bar YX^\T,\quad \bar W_{ij}=\sum_n\bar Y_{in}X_{jn};\qquad \bar X=W^\T\bar Y,\quad \bar X_{jn}=\sum_iW_{ij}\bar Y_{in};\qquad \bar b=\bar Y\one,\quad\bar b_i=\sum_n\bar Y_{in}.$$
The input adjoint is the output adjoint pulled back through the transpose. The weight adjoint is a sum over the columns of outer products, $\bar W=\sum_n\bar y_n\,x_n^\T$: for every token, the response at the output times the signal at the input. Each token contributes a matrix of rank one, and the gradient of $W$ is their sum. @fig-affine-pullback shows it for the readout, where $\bar z$ comes straight from the loss.

::: {.figure #fig-affine-pullback window=1 column=8}
The pullback of the readout $z=W_{\text{out}}x+b_{\text{out}}$ for one column, before training. The adjoint $\bar z=(p-e_y)/N$ arrives from the loss; $\bar x=W_{\text{out}}^\T\bar z$ carries it back into the stream, and $\bar z\,x^\T$ is this column's rank-one share of the gradient of the weights. The full gradient $\bar W_{\text{out}}$ is the sum of the shares of all the columns. Click a column, move the window, point at a square.
:::

### The rectifier

$r=\relu(u)$, entry by entry. Where $u_i>0$ the map is the identity and where $u_i<0$ it is constant, so $Dr(u)=\diag(\one_{u>0})$, a diagonal of ones and zeros, and
$$\bar u=\bar r\odot\one_{u>0}.$$
The pullback is a mask: a unit that was off in the forward pass passes no adjoint back. At $u_i=0$ there is no derivative, the left and right slopes being $0$ and $1$; the code returns $0$ there. That is a convention, not a theorem, and finite differences must stay away from the kink to check it.

### Softmax, and the seed of everything

For $p=\softmax(z)$, the quotient rule gives $\partial p_i/\partial z_j=p_i\delta_{ij}-p_ip_j$, so
$$Dp(z)=\diag(p)-pp^\T,\qquad \bar z=p\odot\bigl(\bar p-\inner{p}{\bar p}\one\bigr).$$
This Jacobian is symmetric, and for any $v$, $v^\T Dp(z)\,v=\sum_ip_iv_i^2-\bigl(\sum_ip_iv_i\bigr)^2=\operatorname{Var}_{i\sim p}(v_i)\ge0$: it is the covariance matrix of the Gibbs distribution of @ch-energies. It annihilates $\one$, which is shift invariance once more: adding the same number to every logit moves nothing, so the pullback never contains such a component, and every $\bar z$ it produces has entries summing to zero.

The model never pulls back through the softmax of its output, because the loss is written directly in terms of the logits (@ch-surprise):
$$\ell=\frac1N\sum_{n=1}^N\Bigl[\lse(z_n)-Z_{y_n,n}\Bigr],\qquad \boxed{\;\bar Z_{:,n}=\frac{p_n-e_{y_n}}{N}\;},$$
since the derivative of $\lse$ is the softmax and that of $Z_{y,n}$ picks out the target. This is the seed of the backward pass: the prediction minus the truth, one column per token, with the loss's single factor $1/N$.

::: {.code file="src/softmax.jl" fn="crossentropy"}
:::

### LayerNorm

For one column (@sec-layernorm), $u=\Pi x$, $\sigma=\sqrt{\lVert u\rVert^2/D+\varepsilon}$, $\hat x=u/\sigma$, $y=\gamma\odot\hat x+\beta$. The affine part is easy: $\bar\gamma=\sum_n\bar y_n\odot\hat x_n$, $\bar\beta=\sum_n\bar y_n$, and the adjoint of $\hat x$ is $g=\gamma\odot\bar y$. For the rest, $du=\Pi\,dx$ and $d\sigma=u^\T dx/(D\sigma)$, which gives
$$d\hat x=\frac1\sigma\Bigl(\Pi-\frac{\hat x\hat x^\T}{D}\Bigr)dx.$$
This Jacobian is symmetric, so it is its own transpose, and
$$\boxed{\;\bar x=\frac1\sigma\Bigl(\Pi-\frac{\hat x\hat x^\T}{D}\Bigr)g=\frac1\sigma\Bigl(g-\operatorname{mean}(g)\,\one-\hat x\,\operatorname{mean}(g\odot\hat x)\Bigr)\;}$$
for every $\varepsilon\ge0$. The geometry of the forward map reappears in its pullback. $\Pi$ removes the component of $g$ along $\one$, which LayerNorm cannot see, since centring forgets it. The second term removes the component along $\hat x$, the radial direction, which normalisation forgets: rescaling a column does not change where it lands on the sphere. What survives is the part of $g$ tangent to the sphere and orthogonal to $\one$, divided by $\sigma$. With $\varepsilon>0$ the radial response is not exactly zero but small, of order $\varepsilon/\sigma^2$, because the column lands only just inside the sphere; the formula is exact either way.

### Attention

Per head and window (@ch-attention): $S=Q^\T K/\sqrt d$, $A=\softmax_{\text{rows}}(S+M)$, $O=VA^\T$. Pull back in reverse order. The product $O=VA^\T$ is bilinear, and the affine rule, applied to each factor, gives
$$\bar V=\bar OA,\qquad \bar A=\bar O^\T V,\qquad \bar A_{ij}=\bar o_i\cdot v_j.$$
Notice that $\bar A$ is a full $T\times T$ matrix. The product $VA^\T$ does not know that $A$ is lower-triangular, and $\bar A_{ij}$ for $j>i$ is how the loss *would* respond if reader $i$ could read the future source $j$. Next comes the softmax of each row, restricted to the allowed sources $j\le i$; by the softmax rule,
$$\bar S_{ij}=\begin{cases}A_{ij}\bigl(\bar A_{ij}-\sum_{k\le i}A_{ik}\bar A_{ik}\bigr),&j\le i,\\[2pt]0,&j>i.\end{cases}$$
The factor $A_{ij}$, which is zero above the diagonal, discards the future: no gradient comes from it. The mask is not differentiated; it restricts the function to the allowed coordinates. Finally $S=Q^\T K/\sqrt d$ is bilinear in $Q$ and $K$:
$$\bar Q=\frac{K\bar S^\T}{\sqrt d},\quad \bar q_i=\frac1{\sqrt d}\sum_j\bar S_{ij}\,k_j;\qquad \bar K=\frac{Q\bar S}{\sqrt d},\quad \bar k_j=\frac1{\sqrt d}\sum_i\bar S_{ij}\,q_i.$$
Three checks of orientation come for free: the first row of $\bar S$ is zero, since $A_{11}=1$ cannot change; everything above the diagonal is zero; and every row of $\bar S$ sums to zero. The last one is worth a section of its own (@sec-bk). The code is the `back` of `attention` and of `mix` in @sec-code.

::: {.figure #fig-softmax-pullback window=1 head=1}
No gradient comes from the future. The weights $A$ of one head before training, the adjoint $\bar A$, which the product $O=VA^\T$ defines on the whole square, and the adjoint $\bar S$ of the scores. Run the pullback: subtract from each row of $\bar A$ its mean under $A$, then multiply by $A$, and everything above the diagonal is lost. Every row of $\bar S$ sums to zero. Point at a row.
:::

That completes the list: embedding, affine maps, the rectifier, LayerNorm, attention, the loss, joined by sums at every fan-out. The backward pass of the whole model is these pullbacks, applied in the reverse of the order of @sec-composition.

## The gradient check {#sec-gradcheck}

Every formula above was derived by hand, and a derivation by hand can be wrong: a missing transpose, a sum where a mean belongs, a forgotten use of a shared value. The program will run anyway and train worse. So we compare with an experiment, the central difference of the opening section.

How good is the experiment? Taylor's theorem gives
$$\frac{\ell(\theta+he_i)-\ell(\theta-he_i)}{2h}=\frac{\partial\ell}{\partial\theta_i}+\frac{h^2}6\,\frac{\partial^3\ell}{\partial\theta_i^3}+O(h^4):$$
the symmetric difference cancels the even terms, and the error falls like $h^2$. But the two losses are computed in floating point, each with a relative error of order the unit roundoff $u=2^{-53}\approx10^{-16}$. Their difference, which is small, inherits an absolute error of order $u\,\ell$, and dividing by $2h$ turns it into an error of order $u\ell/h$. The total,
$$O(h^2)+O(u\ell/h),$$
is smallest when the two terms balance, at $h$ of order $u^{1/3}$, about $10^{-5}$ for quantities of order one. A smaller step is not a more accurate experiment: it subtracts two numbers that agree in more of their digits, and keeps only the digits that are noise.

@fig-gradcheck runs the experiment on the specimen before training. On the left, one parameter from each of four arrays, against the step: the error follows the $h^2$ line down, turns where it meets the rounding line, and climbs back up it. On the right, every one of the parameters at once.

::: {.figure #fig-gradcheck h=-6}
Checking the derivation against an experiment, before training. Left: the error of a central difference against the pulled-back gradient for four parameters, as the step $h$ shrinks: it falls like $h^2$ until rounding takes over, and then it rises like $u\ell/h$. Right: the error for every parameter at one step, grouped by array, with the size of each array's largest gradient entry marked by a bar in covector ink. Click a circle on the left to check every parameter at another step.
:::

The check also comes in a sharper form, which tests one pullback at a time. For any map $f$ and any covector $\bar y$, the gradient of the scalar function $x\mapsto\inner{\bar y}{f(x)}$ is $Df(x)^\T\bar y$, which is exactly what the pullback claims to return. So draw a random $\bar y$, run `back`, and compare with central differences of that scalar function. This is the *adjoint test*. The test suite applies it to every operation of the model, and the gradient check to the model as a whole.

What does a passing check prove? That on these inputs, at these points, the formulas and the code agree to the precision the experiment allows: wrong transposes, missing sums and misplaced factors show up as errors of order one, many decades above the cloud. It does not prove the formulas in general, which is what the derivations are for; and at a kink of the rectifier it proves nothing, because there is no derivative there to compare with.

## The parameter that never learns {#sec-bk}

An exercise of @ch-attention noticed that the key bias $b_k$ shifts every score in a row by the same amount. Now we can say exactly what that does to learning. Write the keys as $K=\tilde K+b_k\one^\T$. Then
$$S=\frac{Q^\T\tilde K}{\sqrt d}+\frac{(Q^\T b_k)\,\one^\T}{\sqrt d},$$
and the second term is constant along every row. The softmax of a row does not change when a constant is added to it, so $A$, the outputs and the loss do not depend on $b_k$ at all. The backward pass agrees. By the affine rule, $\bar b_k=\bar K\one=Q\bar S\one/\sqrt d$, and $\bar S\one=0$ because every row of $\bar S$ sums to zero: $\sum_jA_{ij}\bigl(\bar A_{ij}-c_i\bigr)=c_i-c_i=0$ with $c_i=\sum_kA_{ik}\bar A_{ik}$. So
$$\bar b_k=0,$$
identically, for every input and every value of the other parameters. Of the 1007 numbers of the specimen, these eight can never learn. They are kept on purpose: they belong to the standard architecture, which this book follows, and they make a good puzzle. Many implementations drop them.

In floating point, "identically zero" becomes "zero up to rounding". The computed rows of $\bar S$ sum to numbers like $10^{-19}$ rather than to zero (point at a row of @fig-softmax-pullback), and the computed $\bar b_k$ is of that size, at the very bottom of the scale of @fig-gradcheck, far below every other array's gradient. For the difference quotient the loss does not change at all, except in its last bit, so the check of $b_k$ measures nothing but rounding.

::: aside
Rounding-level gradients still move parameters, if the optimiser divides by the size of the gradient. Adam does (@ch-learning): at the floor of the loss its small constant $\epsilon$ is all that stands between a gradient of order $10^{-17}$ and a step of order the learning rate. In the specimen's training run the key bias drifts away from zero by about $10^{-11}$, and the eight paths of $b_k$ in the cover's trace of every parameter are the ones that never visibly move.
:::

The output bias has a weaker version of the same symmetry. Adding $c\one$ to $b_{\text{out}}$ shifts every logit of a column equally and leaves every probability alone, so $\one^\T\bar b_{\text{out}}=\sum_n\one^\T\bar Z_{:,n}=0$: the gradient never changes the mean of $b_{\text{out}}$, only the differences between its entries. Unlike $b_k$, it still learns those.

## In the code {#sec-pull-code}

The gradient check is a dozen lines; it differences one coordinate at a time:

::: {.code file="src/gradcheck.jl" fn="gradcheck"}
:::

The adjoint test builds the scalar function $x\mapsto\inner{\bar y}{f(x)}$ for a random $\bar y$ and hands it to `gradcheck`:

::: {.code file="src/gradcheck.jl" fn="checkpullback"}
:::

## Exercises {.unnumbered}

::: exercise
Derive the pullbacks of the elementwise product $y=a\odot b$ and of the transpose $Y=X^\T$. Then check the attention formulas $\bar V=\bar OA$ and $\bar A=\bar O^\T V$ by writing $O=VA^\T$ in index form.
:::

::: exercise
A chain of $k$ maps between spaces of dimension $n_0,n_1,\dots,n_k$ has Jacobians $J_i\in\R^{n_i\times n_{i-1}}$. Count the multiplications needed to form the full Jacobian $J_k\cdots J_1$ bracketed from the right, and bracketed from the left. Which is cheaper when $n_k=1$, and which when $n_0=1$? What are $n_0$ and $n_k$ for the specimen's loss?
:::

::: exercise
For the LayerNorm pullback, show that $\one^\T\bar x=0$, and that $\hat x^\T\bar x=0$ when $\varepsilon=0$. Compute $\hat x^\T\bar x$ for $\varepsilon>0$. Which perturbations of the input column cannot change the output, and why can their adjoints not appear?
:::

::: exercise
Show that the softmax Jacobian $\diag(p)-pp^\T$ is positive semidefinite and that, for finite logits, its null space is spanned by $\one$. Deduce that the entries of $\bar b_{\text{out}}$ always sum to zero. Then show that the value bias $b_v$ reaches the output of attention only as the constant $W_ob_v$ added to every column (the rows of $A$ sum to one), so that it duplicates part of $b_o$. Is its gradient zero?
:::

::: exercise
In @fig-gradcheck, click the step $h=10^{-2}$, then $h=10^{-10}$. Which arrays have the largest errors in each case, and why? At every step most dots of $b_k$ sit at the bottom of the scale, and a few sit exactly on the rounding line. Explain both: why is the difference quotient of $b_k$ usually exactly zero, and why, when it is not, is it of size $u\ell/h$?
:::

::: exercise
In @fig-softmax-pullback, point at the first row, then at the last. Why is the first row of $\bar S$ empty? Which row of $\bar A$ loses the most to the mask, and why does losing it cost the model nothing?
:::

# The averaged Jacobian {#ch-jacobian}

::: claim
The rest of the model has a derivative: one $D\times D$ block for every source position and every target position at or after it. Summing the blocks over targets and averaging over sources and texts leaves one matrix $J_\ell$ per depth, which keeps what the model is disposed to do with a direction and forgets what one text does with it; and it costs only $D$ pullbacks per text.
:::

::: names
| maths | physics | ML |
|---|---|---|
| $K_{t't}=\partial X_L[:,t']/\partial X_\ell[:,t]\in\R^{D\times D}$ | the response at $t'$ to a kick at $t$, a propagator | Jacobian |
| $K_{t't}=0$ for $t'<t$ | a retarded response: causality | causal |
| $J_\ell=\frac1{BT}\sum_b\sum_t\sum_{t'\ge t}K^{(b)}_{t't}$ | the static response, averaged over kicks | averaged Jacobian |
| row $i$ of $K_{t't}$, the pullback of $e_i$ | the adjoint of one output coordinate | vector–Jacobian product |
| $J^{\text{self}}_\ell$, $J^{\text{future}}_\ell$ | the response now, and later | present-only and future-only variants |
:::

## The rest of the model as a map {#sec-rest}

Fix one window of $T$ characters. The rest of the model from depth $\ell$ on is a map of the whole window,
$$F_\ell:\R^{D\times T}\to\R^{D\times T},\qquad X_\ell\longmapsto X_L,$$
built from the blocks after depth $\ell$. It is differentiable wherever no rectifier sits exactly at its kink, and its derivative at $X_\ell$ is a linear map from $D\times T$ matrices to $D\times T$ matrices: a $DT\times DT$ matrix. Cut it into $T\times T$ blocks of size $D\times D$, one for each pair of a *target* column $t'$ and a *source* column $t$:
$$dX_L[:,t']=\sum_{t=1}^TK_{t't}\,dX_\ell[:,t],\qquad (K_{t't})_{iq}=\frac{\partial X_L[i,t']}{\partial X_\ell[q,t]}.$$
The block $K_{t't}$ says how a small change of column $t$ at depth $\ell$ moves column $t'$ at the last depth. In matrix form, with the columns stacked into one long vector, $\operatorname{vec}(dX_L)=\mathcal K\operatorname{vec}(dX_\ell)$, and $\mathcal K$ is the block matrix $(K_{t't})$ with rows of blocks indexed by targets and columns of blocks by sources.

Attention lets a column read only columns at or before it ([PicoGPT, attention](https://tobiasosborne.github.io/PicoGPT.jl/attention/)), and every other map acts on one column at a time. A change of column $t$ can therefore move column $t'$ only if $t'\ge t$:
$$K_{t't}=0\qquad\text{for }t'<t,$$
and $\mathcal K$ is block lower-triangular. Of its $T^2$ blocks, $T(T+1)/2$ can be nonzero: 36 for the specimen's windows of eight characters. @fig-j-blocks draws all 36 for the first window, laid out as $\mathcal K$ is written, with nothing above the diagonal, as in the attention triangle.

::: {.figure #fig-j-blocks depth=0}
The derivative of the rest of the model for the window *to be or*, block by block. Row $t'$, column $t$ holds $K_{t't}$, the $8\times8$ matrix of derivatives of the final column $t'$ with respect to column $t$ at the chosen depth, as a Hinton diagram: the area of a square is the size of an entry, filled if it is positive and open if it is negative, all on one scale. Nothing is drawn above the diagonal: a later column cannot move an earlier one. On the right, this window's own average $J^{(1)}_\ell$ and the average $J_\ell$ over all ten windows, on the same scale. [Choose the depth on the axis, and point at a block.]{.web}
:::

The three depths of the specimen show three kinds of derivative.

At depth $1$ nothing is left: $F_1$ is the identity, $K_{tt}=I$ and every other block vanishes.

At depth $\tfrac12$ only the MLP update remains, and it acts on each column alone. The off-diagonal blocks vanish here too, but the diagonal ones are $K_{tt}=I+D\bigl(\operatorname{MLP}\circ\LN\bigr)(x_t)$, and they differ from column to column: each column has its own pattern of active rectifiers and its own length for LayerNorm to divide by.

At depth $0$ attention remains, and the triangle fills in. The block $K_{t't}$ with $t'>t$ is what attention hands from column $t$ to the later column $t'$, through both what $t'$ reads and how much it attends to it, and then carries through the MLP of column $t'$.

::: aside
For a physicist, $K_{t't}$ is a retarded Green's function on a lattice of eight times: the response at time $t'$ to a kick at time $t$, zero before the kick. The sum over $t'\ge t$ below is the total response to one kick, the static susceptibility.
:::

## Averaging {#sec-averaging}

The derivative at one window is a large object, $T(T+1)/2$ blocks, and it changes with the window. The paper replaces it by one $D\times D$ matrix per depth. In our convention,
$$J_\ell=\frac1{BT}\sum_{b=1}^{B}\sum_{t=1}^{T}\sum_{t'=t}^{T}K^{(b)}_{t't},\qquad (J_\ell)_{iq}=\frac1{BT}\sum_{b,t}\ \sum_{t'\ge t}\frac{\partial X^{(b)}_L[i,t']}{\partial X^{(b)}_\ell[q,t]},$$
over $B$ windows $b$. Read from the inside out. A nudge $dx$ to column $t$ at depth $\ell$ moves the final columns $t,t+1,\dots,T$ by $K_{t't}\,dx$; the sum over $t'$ adds these up, the total first-order effect of the nudge on the present output and on every later one. The mean over $t$ and $b$ then forgets which position was nudged and in which text. $J_\ell\,dx$ is what a nudge $dx$ at depth $\ell$ does to the final stream, now and later, on average over where and when it happens.

The paper writes the same average as an expectation over pairs, $\E_{t,\,t'\ge t,\,\text{prompt}}\bigl[\partial h_{\text{final},t'}/\partial h_{\ell,t}\bigr]$. For windows of fixed length there are $T(T+1)/2$ pairs to every $T$ sources, so the paper's matrix is ours times $2/(T+1)$: a positive scalar. We sum over targets instead of averaging because then, at the last depth, $K_{t't}=\delta_{t't}I$ and
$$J_L=I$$
exactly. The scalar is not quite harmless, because LayerNorm's $\varepsilon$ sets a fixed scale of its own, and @ch-jlens shows what that does to the lens when $J_\ell$ is shrunk by $2/(T+1)$.

The two averages weight positions alike: in both, source $t$ contributes its $T-t+1$ blocks, so the first column of a window contributes $T$ blocks to $J_\ell$ and the last contributes one. Early positions weigh more in the disposition. That is a choice of the paper's, not an identity, and it interacts with keeping the first positions in the average (@sec-jacobian-shakespeare).

Why average at all? The paper's answer is that the derivative at one text conflates two things: the model's general disposition to do something with a direction, and the particular use it is putting that direction to in this text. In our terms, $K^{(b)}_{t't}$ depends on the window through everything the forward pass computed: which rectifiers are on, how long each column is, and above all which later columns attend to column $t$, and how much. Averaging over many texts keeps what survives all of them. @fig-j-average shows how much a single window's average differs from the grand one. At depth $\tfrac12$, where no attention remains and only the first two of these causes act, the ten windows' own averages lie between {{value:jacWindowDevMin}}% and {{value:jacWindowDevMax}}% of the size of $J_{1/2}$ away from it, in Frobenius norm, although they are ten overlapping windows of one line.

::: {.figure #fig-j-average}
Above: one window at a time. The average $J^{(b)}_{1/2}=\frac1T\sum_t\sum_{t'\ge t}K^{(b)}_{t't}$ of each of the specimen's ten windows at depth $\tfrac12$, with its distance from their mean $J_{1/2}$ as a percentage of $\lVert J_{1/2}\rVert$, Frobenius norms throughout. Below: present and future. At each depth, $J_\ell=J^{\text{self}}_\ell+J^{\text{future}}_\ell$, the averaged effect of a column on its own output and on the outputs after it. After depth $\tfrac12$ no attention remains, and the future part is empty. One area scale for every matrix. [Point at a window or a depth.]{.web}
:::

::: aside
In the paper: *Methods*, "The Jacobian Lens", defines $J_\ell$ and gives the reason for averaging; the appendix "Methodological Details and Ablations" varies everything: the target layer, gradients through the attention pattern, the target positions, median against mean, and dropping early positions and outlying prompts. We use the final depth (the paper's default lens stops one block earlier), full gradients and the plain mean, and drop nothing.
:::

## Present and future {#sec-present-future}

The sum over targets splits into the target $t'=t$ and the targets after it, and so does $J_\ell$:
$$J_\ell=J^{\text{self}}_\ell+J^{\text{future}}_\ell,\qquad J^{\text{self}}_\ell=\frac1{BT}\sum_{b,t}K^{(b)}_{tt},\qquad J^{\text{future}}_\ell=\frac1{BT}\sum_{b,t}\sum_{t'>t}K^{(b)}_{t't}.$$
The paper tries both parts as lenses of their own. The *self-only* lens asks what a column does to its own output, and is closest in spirit to the logit lens. The *future-only* lens asks what a column makes available to later positions, which can only travel through attention: the part of the state that is broadcast forward in the text. In the specimen, the future part vanishes at depths $\tfrac12$ and $1$, since nothing after depth $\tfrac12$ mixes columns (@fig-j-average, below).

## Computing it by pullbacks {#sec-by-pullbacks}

How is $J_\ell$ computed without forming $\mathcal K$? Take one row of one block. Row $i$ of $K_{t't}$ is
$$e_i^\T K_{t't}=\Bigl(\frac{\partial X_L[i,t']}{\partial X_\ell[:,t]}\Bigr)^{\!\T},$$
the gradient of one number, entry $i$ of the final column $t'$, with respect to column $t$ at depth $\ell$. That is a pullback, in the sense of [PicoGPT's chapter on pulling back](https://tobiasosborne.github.io/PicoGPT.jl/pulling-back/). Put the covector $e_i$ into column $t'$ of an otherwise zero $D\times T$ matrix at the last depth, and pull it back through the pullbacks of the half blocks, down to depth $\ell$:
$$\bar X_\ell[:,t]=\sum_{t''}K_{t''t}^\T\,\bar X_L[:,t''],\qquad\text{so}\qquad \bar X_L=e_ie_{t'}^\T\quad\Longrightarrow\quad\bar X_\ell[:,t]=K_{t't}^\T e_i.$$
The first identity is the transpose of $\mathcal K$, written block by block.
Column $t$ of the result is row $i$ of $K_{t't}$, for every source $t$ at once. One block costs $D$ pullbacks, one per row; the whole triangle of a window costs $DT$.

The pullback is linear in the covector, and that is the trick. Put $e_i$ into *every* column at once, $\bar X_L=e_i\one^\T$. Then
$$\bar X_\ell[:,t]=\sum_{t'=1}^TK_{t't}^\T e_i=\sum_{t'\ge t}K_{t't}^\T e_i,$$
causality supplying the restriction $t'\ge t$ for free. Column $t$ is row $i$ of $\sum_{t'\ge t}K_{t't}$, which is exactly what $J_\ell$ sums; the mean over the columns and the windows is row $i$ of $J_\ell$. So $J_\ell$ costs $D$ pullbacks per window, whatever $T$ is, and the same pullbacks deliver it at every depth on their way down. @fig-j-pullback runs both versions on the first window.

::: {.figure #fig-j-pullback row=1 target=8}
Computing the averaged Jacobian by pullbacks, on the window *to be or*. The forward pass runs down the stems; the covector $e_i$ is placed at the last depth and pulled back up, through the MLP to depth $\tfrac12$ and through attention to depth $0$. What arrives in each column is drawn in covector ink, one spoke per entry. From one target column it brings row $i$ of the blocks $K_{t't}$, and nothing to the columns after $t'$. From every column at once it brings, in one pass, row $i$ of $\sum_{t'\ge t}K_{t't}$; their mean over the sources is row $i$ of the window's average, and the same row of $J_0$ over all ten windows is drawn beside it. [Click a glyph of the bottom row, or the line under it; click a row of $J_0$; scrub the pullback.]{.web} Each row has its own spoke scale, printed at its left; lengths compare within a row only.
:::

Two remarks. First, the computation produces $J_\ell$ one *row* at a time, and each row is a pulled-back covector: the rows of $J_\ell$ are gradients. That is why the lens vectors of @ch-jlens, the rows of $W_{\text{out}}J_\ell$, are drawn in covector ink. Second, the trick sums over targets, so it cannot separate them: $J^{\text{self}}_\ell$ needs the diagonal blocks alone, and costs a pullback per target and per row. That is why the Shakespeare export below omits it.

## In the Shakespeare model {#sec-jacobian-shakespeare}

For the Shakespeare model each $J_\ell$ is a $128\times128$ matrix, at each of nine depths. The analysis computes them from {{value:lensShWindows}} windows of {{value:jacShLength}} characters of held-out text, every one of their {{value:jacShSources}} positions a source, {{value:jacShPairs}} pairs of positions in all, with 128 pullbacks per window. It keeps every window and every position, including the first positions of each window, which the paper considers dropping. It does not compute the self and future parts, which would cost 256 times as much.

## In the code {#sec-code-jacobian}

The pullback from the last depth to depth $\ell$ is the pullbacks of the half blocks, applied in reverse order:

::: {.code file="jspace/src/JLens.jl" fn="pull_to_depth"}
:::

One block, one row per pullback:

::: {.code file="jspace/src/JLens.jl" fn="jacobian_block"}
:::

and the average, with $e_i$ in every column, recorded at every depth on the way down:

::: {.code file="jspace/src/JLens.jl" fn="averaged_jacobians"}
:::

## Exercises {.unnumbered}

::: exercise
Show that with the sum over targets, $J_L=I$, $J^{\text{self}}_L=I$ and $J^{\text{future}}_L=0$. Show that the paper's average over pairs is $\frac2{T+1}J_\ell$ for windows of fixed length $T$. What becomes of the relation between the two when windows have different lengths?
:::

::: exercise
Suppose a model had no attention, so that every map after depth $\ell$ acts on one column at a time. Show that $K_{t't}=0$ for $t'\ne t$, that $J^{\text{future}}_\ell=0$, and that $J_\ell$ is the average over columns of the derivative of the column map. Which depths of the specimen are of this kind, and what does @fig-j-blocks show there?
:::

::: exercise
Count pullbacks for one window: for one block, for the whole triangle, for $J_\ell$, and for $J^{\text{self}}_\ell$. Do it for the specimen and for the Shakespeare model. How many numbers would the full derivative $\mathcal K$ of one Shakespeare window hold?
:::

::: exercise
In @fig-j-pullback (on the web; in print, from the two states drawn), seed $e_i$ in the last column and then in the first. Which source columns receive something in each case, and why? Then seed every column, and explain why the first column at depth $0$ receives the most terms in its sum.
:::

::: exercise
The mean over sources and the sum over targets are both linear, so $J_\ell\,dx$ is an average of first-order effects. Give an example, with a one-dimensional "model", of a map whose derivative averages to zero over its inputs although no single derivative is small. What would the lens built from such an average read?
:::

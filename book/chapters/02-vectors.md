# Symbols become vectors {#ch-vectors}

::: claim
Character $a$ becomes the column $Ee_a$ of a learned matrix $E\in\R^{D\times V}$, position $t$ becomes a fixed vector of sines and cosines on which a shift acts as a rotation, and their sum is the first state of the model: one column of $\R^D$ per token.
:::

::: names
| maths | physics | ML |
|---|---|---|
| a standard basis vector $e_a\in\R^V$ | a pure state, an occupation-number basis vector | one-hot vector |
| $E\in\R^{D\times V}$, column $a$ is $Ee_a$ | a coupling of each symbol to $D$ modes | embedding matrix, token embedding |
| $x\mapsto E[:,x]=EC$, with $C$ one-hot | evaluating a linear map on basis states | embedding lookup |
| the dimension $D$ of the columns | the number of modes | model dimension, width |
| $P_t=\bigl(\sin\bigl(\omega_r(t-1)\bigr),\cos\bigl(\omega_r(t-1)\bigr)\bigr)_{r}$, fixed | phases of $D/2$ oscillators | sinusoidal positional encoding |
| $X_0=E[:,x]+P\in\R^{D\times T}$ | the initial condition | input embeddings |
:::

## One-hot vectors {#sec-one-hot}

The integers $1,\dots,V$ that name the symbols are labels, not quantities. Nothing about the model should depend on the fact that `o` is numbered 5 and `r` is numbered 6, or on any arithmetic of the labels. So we send symbol $a$ to the standard basis vector
$$e_a\in\R^V,\qquad (e_a)_b=\delta_{ab}.$$
Every symbol is a unit vector and any two are orthogonal: $e_a\cdot e_b=\delta_{ab}$, so $\lVert e_a-e_b\rVert=\sqrt2$ for every pair $a\ne b$. No symbol starts closer to one than to another. Relabelling the alphabet permutes the basis and changes nothing else. In the language of the field, $e_a$ is a *one-hot* vector: one coordinate is hot, the others cold.

A one-hot vector is a terrible representation to compute with, and a perfect one to compute *from*. A linear map $E:\R^V\to\R^D$ is determined by what it does to the basis, and on $e_a$ it simply returns its column $a$.

## The embedding matrix {#sec-embedding}

The model's first parameters are a matrix $E\in\R^{D\times V}$, whose columns are indexed by the alphabet. Symbol $a$ is sent to
$$Ee_a=E_{:,a}\in\R^D,\qquad (Ee_a)_i=\sum_bE_{ib}\,\delta_{ba}=E_{ia}.$$
That is the whole of the *embedding*: column $a$ of $E$ is what the model means by the character $a$. For a sequence $x_1,\dots,x_T$, collect the one-hot vectors as the columns of $C=(e_{x_1},\dots,e_{x_T})\in\{0,1\}^{V\times T}$; then
$$EC=\bigl(E_{:,x_1},\dots,E_{:,x_T}\bigr)=E[:,x]\in\R^{D\times T}.$$
In matrix form the lookup is a product with a one-hot matrix; in code it is indexing, and $C$ is never formed. The specimen has $D=8$ and $V=7$, so $E$ holds {{value:embeddingParams}} of its parameters. Training adjusts them like all the others.

::: {.figure #fig-one-hot char=5}
A character becomes a vector. The alphabet is a row of seven slots, and the one-hot vector $e_a$ fills one of them; the embedding returns column $a$ of $E$, drawn as a spoke glyph with one spoke for each of its eight components. [Click another character.]{.web}
:::

The vectors in this book are drawn as spoke glyphs: component $k$ is a spoke at angle $2\pi(k-1)/D$ from twelve o'clock, clockwise, with length proportional to $\lvert x_k\rvert$ and a filled tip for a positive component, an open one for a negative. The key in the margin is drawn from the figure's own vector. All the glyphs of one figure share one scale, so lengths can be compared across a figure but not between figures.

::: aside
Why should a character be a point in $\R^8$, when $\R^7$ would hold seven orthogonal vectors exactly? Because every later map is linear in these columns, or close to it, and the model needs room: in $D$ dimensions it can place the characters where its later maps find them useful, and $D$ is chosen once for the whole model, not for the alphabet.
:::

What does training do to $E$? @fig-embedding-matrix shows the matrix before and after. At the start its entries are drawn independently from a standard normal distribution: seven random directions in $\R^8$, which are already far from one another. Training moves them, but not far. The specimen learns its text mostly in the maps that come later, which see the columns of $E$ only through the linear maps they apply to them.

::: {.figure #fig-embedding-matrix}
The embedding matrix $E\in\R^{8\times7}$ during training. Each column belongs to a character, drawn as squares whose area is $\lvert E_{ia}\rvert$, filled where $E_{ia}>0$, and again as a spoke glyph. The faint glyphs are where the columns started. [Scrub through the 200 training steps.]{.web}
:::

## Positions as Fourier features {#sec-positions}

The column $Ee_{x_t}$ does not know where it stands. The maps of @ch-block act on each column separately, and attention, the only map in which columns interact, weights them by scores between columns; without more information it would treat the past as an unordered set. @ch-attention makes this precise in an exercise: without positions, a character cannot tell the order of what came before it, and `bot` read from its `t` looks exactly like `obt`. So the model adds to column $t$ a fixed vector $P_t$ that encodes $t$.

The code is the one of the original transformer paper. Take $D/2$ frequencies in geometric progression and, for each, a sine and a cosine of the position:
$$P_{2r+1,\,t}=\sin\bigl(\omega_r(t-1)\bigr),\qquad P_{2r+2,\,t}=\cos\bigl(\omega_r(t-1)\bigr),$$
$$\omega_r=10000^{-2r/D},\qquad r=0,\dots,D/2-1,\quad t=1,\dots,T.$$
Position 1 is phase zero. Each pair of rows is a clock hand turning at angular velocity $\omega_r$ per step; the first turns by one radian per step, the last by about $10^{-4\cdot(D-2)/D}$. For the specimen, $D=8$ gives four clocks, with $\omega_r=1,\,10^{-1},\,10^{-2},\,10^{-3}$. Every column has the same length, $\lVert P_t\rVert^2=\sum_r(\sin^2+\cos^2)=D/2$: the positions lie on a product of $D/2$ circles, a torus inside the sphere of radius $\sqrt{D/2}$.

::: {.figure #fig-positions}
The positional code $P\in\R^{8\times8}$. Each row is a sampled sinusoid: the hairline is $\sin(\omega_r\tau)$ or $\cos(\omega_r\tau)$ for continuous $\tau$, and the stems are its values at the eight positions, the entries of $P$. On the right are the same numbers as a matrix, with area $\lvert P_{it}\rvert$, filled where positive, and column $t$, the vector added at position $t$, as a spoke glyph. [Point at a position.]{.web}
:::

The fast clock tells neighbours apart; the slow ones barely move over eight positions. They are there for longer contexts: with $T=256$, as in the Shakespeare model of @ch-speaking, the slowest of its clocks still turns by less than a tenth of a radian over the whole window, and in between the clocks cover every scale of distance.

## A shift is a rotation {#sec-rotation}

Why sines and cosines, rather than writing $t$ itself into a coordinate? Because a shift of position then acts linearly, and by the same map at every position. For each frequency, the angle-addition formulas give
$$\begin{pmatrix}\sin\omega_r(t+k)\\ \cos\omega_r(t+k)\end{pmatrix}=\begin{pmatrix}\cos\omega_rk&\sin\omega_rk\\ -\sin\omega_rk&\cos\omega_rk\end{pmatrix}\begin{pmatrix}\sin\omega_rt\\ \cos\omega_rt\end{pmatrix},$$
and stacking the pairs,
$$P_{t+k}=R(k)\,P_t,\qquad R(k)=\bigoplus_{r=0}^{D/2-1}\begin{pmatrix}\cos\omega_rk&\sin\omega_rk\\ -\sin\omega_rk&\cos\omega_rk\end{pmatrix}.$$
Each block is a rotation, by $-\omega_rk$ because the coordinates are ordered sine before cosine, so $R(k)$ is orthogonal, $R(0)=I$ and $R(k+\ell)=R(k)R(\ell)$: the shifts act on $\R^D$ as a group of rotations. The map $R(k)$ does not depend on $t$. A linear map applied to the code can therefore compute "$k$ places earlier" in the same way wherever it stands.

A second consequence follows from $\sin a\sin b+\cos a\cos b=\cos(a-b)$:
$$P_s\cdot P_t=\sum_{r=0}^{D/2-1}\cos\bigl(\omega_r(t-s)\bigr).$$
The inner product of two positions depends only on their distance. The Gram matrix $P^\T P$ is constant along its diagonals: it is a Toeplitz matrix, with $D/2$ on the diagonal. It does not simply fall off with distance. The fast clock has period $2\pi$, a little more than six steps, so on that clock positions six apart are nearly in phase again, and only the slower clocks keep them apart.

::: {.figure #fig-rotation shift=3}
A shift is a rotation. Each circle is one frequency $\omega_r$: the dots are the eight positions, at angle $\omega_r(t-1)$, and the hands are $P_t$ and $P_{t+k}$. Shifting by $k$ turns every hand by $\omega_rk$, whatever $t$ is. Right: the inner products $P_s\cdot P_t$, constant along each diagonal. [Drag a hand round its circle to change $t$, or the shift $k$ along its scale.]{.web}
:::

None of this makes the trained model translation invariant. The model never sees $P$ alone; it sees $E[:,x]+P$, through learned maps that can use absolute position as freely as relative position. The rotation property guarantees only that relative position is *available* as a linear function of the code.

::: aside
Rotary position embeddings ([Su et al., 2021](https://arxiv.org/abs/2104.09864)), now used by most large language models whose design is published, take the rotation literally: instead of adding $P_t$, they rotate the query and key vectors of @ch-attention by $R(t)$ before the inner product, so that a score depends on positions only through $t-s$.
:::

## The first state {#sec-x0}

The input to the first block is the sum of the two codes,
$$X_0=E[:,x]+P\in\R^{D\times T},\qquad (X_0)_{it}=E_{i,x_t}+P_{it}.$$
In matrix form, $X_0=EC+P$; for a batch of windows, the same $P$ is added to each, column by column. Addition rather than concatenation keeps the width at $D$: the character and the position share the same coordinates, superposed, and the learned maps that follow must separate them. The same character at two positions is now two different vectors, and the same position holds a different vector for each character.

::: {.figure #fig-x0 window=1}
What enters the first block. For each position of the window, top to bottom: the column $E_{:,x_t}$ of the character, the positional code $P_t$, and their sum $X_0[:,t]$, all on one scale. Repeated characters, joined by a hairline above, share their top glyph and differ below. [Drag the window along the line.]{.web}
:::

From here on, $X\in\R^{D\times N}$, one column per token, is called the *residual stream*, for reasons that @ch-block explains. Every block reads it and adds to it; nothing else ever writes to it.

## The pullback, in one line {#sec-embed-pullback}

How does the loss reach $E$? Hold the characters $x$ fixed: then $dX_0=dE\,C$, and for any covector $\bar X_0$,
$$\langle\bar X_0,dX_0\rangle=\operatorname{tr}\bigl(\bar X_0^\T dE\,C\bigr)=\langle\bar X_0C^\T,dE\rangle,\qquad\text{so}\qquad \bar E=\bar X_0C^\T,\quad \bar E_{:,a}=\sum_{n:\,x_n=a}\bar X_{0,:,n}.$$
Column $a$ of $\bar E$ collects the adjoints of every token that is the character $a$. The adjoint of a lookup is a *scatter-add*. The positional code is fixed, so it receives no adjoint, and the characters themselves are integers, so they have none. @ch-pulling-back derives every other pullback in the same way.

## In the code {#sec-code-vectors}

The positional code is a comprehension over rows and positions:

::: {.code file="src/embedding.jl" fn="positions"}
:::

and the embedding returns its value and its pullback, the scatter-add above:

::: {.code file="src/embedding.jl" fn="embed"}
:::

## Exercises {.unnumbered}

::: exercise
Show that $\lVert e_a-e_b\rVert=\sqrt2$ for all $a\ne b$, and that any permutation of the alphabet acts on $\R^V$ as an orthogonal map. If the alphabet were instead encoded as the single number $a\in\R$, which pairs of characters would a linear map find closest?
:::

::: exercise
Prove $P_{t+k}=R(k)P_t$ and $R(k)^\T R(k)=I$. Show that $P_s\cdot P_t=\sum_r\cos(\omega_r(t-s))$, and compute $\lVert P_{t+1}-P_t\rVert^2$ for the specimen's four frequencies. Which clock contributes most?
:::

::: exercise
Try it in @fig-positions. Over the eight positions of the specimen, the slowest clock turns by $7\omega_3$. How large is that angle, and what do its two rows of $P$ look like? What would you lose by deleting them?
:::

::: exercise
Try it in @fig-x0. Choose a window in which a character appears twice and compare the two columns of $X_0$. Then use @fig-rotation to find the two distinct positions of the specimen whose codes have the largest inner product, and explain the answer with the period of the fast clock. Why can $P_s\cdot P_t$ never be smaller than $-D/2$?
:::

::: exercise
The embedding has $DV$ parameters. A *learned* positional code, as in GPT-2, would replace $P$ by a parameter matrix of the same shape. How many parameters would that add to the specimen, and what property of this chapter would it give up?
:::

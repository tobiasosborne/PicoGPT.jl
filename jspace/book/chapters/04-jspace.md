# The J-space {#ch-jspace}

::: claim
Fix a sparsity $k$. The J-space is the set of nonnegative combinations of $k$ lens vectors: a union of cones, not a subspace. Every column has a nearest point in it, its J-component, and the remainder is orthogonal to that point. How much of the stream lies in the J-space, against random directions, is a measurement, and on our two models it can be made exactly.
:::

::: names
| maths | physics | ML |
|---|---|---|
| $C_S=\bigl\{\sum_{a\in S}c_av_a:\ c_a\ge0\bigr\}$, a polyhedral cone | the mixtures of a few pure spectra | a sparse nonnegative combination |
| $\mathcal J_k=\bigcup_{\lvert S\rvert=k}C_S$ | spectra fitted with at most $k$ lines | J-space |
| $x_J$, the nearest point of $\mathcal J_k$ to $x$ | the best nonnegative fit | J-space component |
| $c_a\ge0$ with $x_J=\sum_ac_av_a$ | the fitted abundances | (local) J-space coordinates |
| $x-x_J$, orthogonal to $x_J$ | the residual of the fit | non-J-space component, residual |
| $(v_a)_{a=1}^V$: here independent and not spanning ($V<D$); in the paper spanning and dependent | a set of generators, like coherent states | frame |
| $V$ random unit vectors | a null model | random control |
| the $k$ at which the lens vectors stop beating random ones | the number of lines worth fitting | occupancy |
:::

## From a list of vectors to a set {#sec-cones}

At each depth @ch-jlens gave us $V$ lens vectors $v_1,\dots,v_V$, covectors on the stream. To speak of columns that are *made of* lens vectors we must turn them into columns, and the J-space does it with the Euclidean inner product of the stream's coordinates: in coordinates nothing happens, the numbers of a row become the numbers of a column. It is a choice, which @sec-steer examines; it is the paper's, and it is not arbitrary, since LayerNorm measures the length of a column with the same inner product. The lens of @ch-jlens did not need it. Everything in this chapter does.

Given a set $S$ of characters, the *cone* of their lens vectors is the set of their nonnegative combinations,
$$C_S=\Bigl\{\sum_{a\in S}c_av_a\ :\ c_a\ge0\Bigr\}\subset\R^D,$$
or in matrix form $C_S=\{A_Sc:\ c\ge0\}$, with $A_S\in\R^{D\times\lvert S\rvert}$ the matrix whose columns are the $v_a$, $a\in S$. A cone is closed under addition and under multiplication by nonnegative numbers. It contains $v_a$ but not, in general, $-v_a$, and that is the difference from a subspace: a subspace would contain both. Fix a sparsity $k$. The *J-space* is the union of all the cones of $k$ lens vectors,
$$\mathcal J_k=\bigcup_{\lvert S\rvert=k}C_S .$$
There are $\binom Vk$ cones in the union, and since a coefficient may be zero, $\mathcal J_1\subset\mathcal J_2\subset\cdots\subset\mathcal J_V$. A union of cones is not a subspace and not even convex: the segment between points of two different cones generally leaves both.

Why nonnegative, and why sparse? The paper's answer is empirical and interpretive. A column that carries a character's lens vector carries a positive amount of it; $-v_a$ is not a concept the lens can name. And in its models only a few lens vectors are strongly present in a column at a time, so it allows $k$ of them, typically up to 25. The paper calls a spanning set of vectors that need not be independent a *frame*, and the lens vectors with their sparsity limit a *sparse subframe*; the J-space is what that subframe generates. That is all the vocabulary we need from it.

::: aside
In the paper, the J-space is defined in the Methods section "The J-Space" and formalised in the appendix "Formalization of the J-space". The appendix writes the union with $\operatorname{span}$ and the distance with the orthogonal projection $\Pi_S$ onto the span of $S$, while the text says cones. With nonnegativity the nearest point of a cone is a nonnegative least-squares fit, not a projection onto a span; the two agree exactly when the projection onto the span happens to have nonnegative coefficients. Our analysis computes the cones.
:::

## The nearest point {#sec-nearest}

The distance from a column $x$ to the J-space is
$$d_k(x)=\min_{y\in\mathcal J_k}\lVert x-y\rVert=\min_{\lvert S\rvert=k}\ \min_{c\ge0}\ \lVert x-A_Sc\rVert .$$
The inner problem is *nonnegative least squares*: minimise a convex quadratic over the orthant $c\ge0$. A cone is closed and convex, so for each $S$ the nearest point $A_Sc$ exists and is unique. The outer minimum runs over finitely many cones. A winner is the J-component $x_J$ of $x$, its coefficients $c_a$ are the J-coordinates, and $x-x_J$ is the remainder; ties between cones are possible, and a winning subset may use some of its vectors with coefficient $0$, so the coordinates belong to a chosen nearest point (the code keeps the first subset in lexicographic order).

The minimiser of the inner problem is characterised by the Karush–Kuhn–Tucker conditions, which the analysis's code quotes in its docstring: with $r=x-A_Sc$,
$$c\ge0,\qquad A_S^\T r\le0,\qquad c_a\,(A_S^\T r)_a=0\ \text{ for every }a .$$
A coefficient is either zero, or its vector is orthogonal to the residual. Multiply the last condition by $c_a$ and sum:
$$\inner{x_J}{x-x_J}=\sum_ac_a\,\inner{v_a}{r}=0,$$
and therefore, by Pythagoras,
$$\lVert x\rVert^2=\lVert x_J\rVert^2+\lVert x-x_J\rVert^2 .$$
The remainder is orthogonal to the J-component, as it would be for a projection onto a subspace, although the map $x\mapsto x_J$ is not linear. That is what makes "the share of $\lVert x\rVert^2$ in the J-space", $1-d_k(x)^2/\lVert x\rVert^2$, a number between $0$ and $1$ with a meaning.

::: aside
For a single closed convex cone $C$ this is a theorem of Jean-Jacques Moreau (*Comptes rendus de l'Académie des sciences*, 1962): every vector splits uniquely as $x=\Pi_C(x)+\Pi_{C^\circ}(x)$, into its nearest points in $C$ and in the polar cone $C^\circ=\{y:\inner{y}{z}\le0\ \forall z\in C\}$, with the two parts orthogonal. The J-space is a union of cones, so it gets the orthogonality cone by cone, and not the uniqueness.
:::

With $k=2$ the whole problem can be drawn. Two lens vectors $v_s,v_u$ and the column $x$ span a subspace of dimension three, and every point of the problem lies in it: the plane of $v_s$ and $v_u$, the cone in that plane, $x$, its foot $p$ in the plane, and the nearest point. An exact picture of three dimensions loses nothing of the eight. @fig-j-cone draws it for a column of the specimen, and introduces the glyph this book uses for a cone.

::: {.figure #fig-j-cone window=1 column=4 depth=0.5}
The cone of two lens vectors and the nearest point of it, exactly. Left: the plane spanned by $v_s$ and $v_u$ at depth $\tfrac12$ (the disc, which a subspace would fill), seen from above; their cone, hatched; $-v_s$ and $-v_u$, dashed; a column $x$ of the specimen above the plane, its foot $p$, its nearest point $x_J$ on the cone, and the remainder $x-x_J$, dashed. The numbers on the edges are the J-coordinates $c_s,c_u$ of $x_J=c_sv_s+c_uv_u$. Beyond the dotted rays every point of the plane is nearest to the apex. Right: the cones of all $21$ pairs, each in its own plane, with the foot of the same $x$ (open) and its nearest point (filled); under each, the share of $\lVert x\rVert^2$ left over. The J-space with $k=2$ is the union of the $21$ cones, and its distance from $x$ is the smallest of the $21$ numbers, in bold. [Choose the column on the line, a pair among the small cones, and drag the foot $p$ across the plane.]{.web}
:::

In the plane, the nearest point of the cone to the foot $p$ is found by inspection. If $p$ lies in the cone, it is its own nearest point. If it lies beyond one edge, in the strip between that edge and the dotted ray perpendicular to it, the nearest point is the foot of the perpendicular on that edge. If it lies between the two dotted rays, in the polar cone, the nearest point is the apex: the column leans away from both vectors, and the cone contributes nothing. The figure opens on the fourth character of the first window, at depth $\tfrac12$. Its best pair is {{value:jsConePair}}, its foot lies inside their cone, and the remainder, {{value:jsConeMiss}} of $\lVert x\rVert^2$, is all perpendicular to the plane. Many of the other cones on the right leave all of $x$ over: their nearest point is the apex.

One more thing can be read off the small cones. At depth $\tfrac12$ the vectors of {{value:jsOppositePair}} have cosine {{value:jsOppositeCos}}: their cone is nearly a half-plane, and nonnegative combinations of two nearly opposite vectors reach almost any direction in it by cancelling large coefficients. Nonnegativity constrains such a pair very little. @fig-j-decomposition shows where that leads.

## Every subset, exactly {#sec-exact}

In the specimen $V=7$, so for each $k$ there are at most $\binom73=35$ cones, and over all $k$ there are $2^7-1=127$ of them. The analysis tries every one: for each subset, nonnegative least squares by the active-set method of Lawson and Hanson, which moves coefficients between a free set, solved by ordinary least squares, and a set held at zero, until the conditions above hold.

::: {.code file="jspace/src/JLens.jl" fn="nnls" doc=true}
:::

and keeps the best cone for each $k$:

::: {.code file="jspace/src/JLens.jl" fn="exact_decomposition" doc=true}
:::

@fig-j-decomposition shows the result for the column of @fig-j-cone at all three depths and every $k$.

::: {.figure #fig-j-decomposition window=1 column=4}
The J-coordinates of one column of the specimen as $k$ grows, at its three depths. Row $k$ is the nearest point of the J-space with $k$ lens vectors, found by trying every subset. A dot in the column of a character means that its lens vector is in the winning subset; its area is the length $\lVert c_av_a\rVert$ of that term of $x_J$, on one scale for the whole figure, and an open ring is a vector of the subset whose coefficient is exactly $0$. At the right of each row, the share of $\lVert x\rVert^2$ the J-space misses (bar) and the same share for random sets of seven unit vectors (tick, the mean of $20$ sets). [Choose the column on the line; point at a row for its numbers.]{.web}
:::

Three things are worth seeing. At depth $1$ a single lens vector holds most of the column, and it is the readout row of the character the model is about to predict; over all columns, the best single vector at depth $1$ is that row in {{value:jsSpecTopOne}} of {{value:jlSpecColumns}}. At depths $0$ and $\tfrac12$ the subsets are not nested. The best pair need not contain the best single vector, and over all columns, depths and $k$, the best $(k+1)$-subset fails to contain the best $k$-subset in {{value:jsNotNested}} of {{value:jsNestCases}} cases; a method that only ever adds vectors, like the one of @sec-greedy, can therefore miss the best cone. The count takes the subsets as the export records them, zero coefficients included. And for $k\ge4$ at depth $0$ the dots grow large: the fit pays nearly opposite vectors large coefficients to cancel each other, and what remains is a small difference of large terms. The coefficients of such a fit say little about which characters the column "contains", although the fit is the best there is.

## Against chance {#sec-chance}

How much of the stream lies in the J-space? Any $V$ vectors capture something, so the number means something only against a control: the paper's is a set of the same size made of random directions, and the analysis uses $20$ sets of seven independent random unit vectors in $\R^8$. @fig-j-variance compares the two over all {{value:jlSpecColumns}} columns.

::: {.figure #fig-j-variance}
How much of the specimen's stream the J-space captures, against random sets of the same size. In each panel, a depth; against $k$, the share $1-d_k(x)^2/\lVert x\rVert^2$ averaged over all columns, for the lens vectors (dots; the columns themselves as faint lines) and for random sets of seven unit vectors (open dots, the mean of $20$ sets). At the right, the share in the full span of the seven vectors, for the lens vectors and for random sets, and $7/8$, what a random seven-dimensional subspace of $\R^8$ holds on average. [Point at a $k$.]{.web}
:::

At depth $1$ the J-space with all seven vectors misses {{value:jsSpecMissLastJ}} of $\lVert x\rVert^2$ on average, where random sets miss {{value:jsSpecMissLastRandom}}: the final stream lies close to nonnegative combinations of the readout's rows, which is what the readout's rows are for. At depths $0$ and $\tfrac12$ the lens vectors are barely better than chance: with all seven they miss {{value:jsSpecMissZeroJ}} and {{value:jsSpecMissHalfJ}}, the random sets {{value:jsSpecMissZeroRandom}} and {{value:jsSpecMissHalfRandom}}, a lead of a point or a few with no spread to set it against; for $k\le3$ they do worse than the random sets.

Here the shape of the problem differs from the paper's, and the figure shows how. The paper's models have $V\gg D$: their lens vectors span the whole stream many times over, and only sparsity and nonnegativity limit the J-space. Ours have $V<D$: seven vectors in $\R^8$ span at most seven dimensions, and even with $k=V$ every column keeps the part orthogonal to their span. At depth $0$ the span holds {{value:jsSpecSpanZero}} of $\lVert x\rVert^2$ on average, close to the $7/8$ of a random subspace (an exercise), and the cone of all seven vectors only {{value:jsSpecCapturedZero}}. The difference is the price of nonnegativity; the rest is the price of $V<D$. At depth $1$ the span holds {{value:jsSpecSpanLast}}.

## Sixty-five characters in 128 dimensions {#sec-greedy}

The Shakespeare model has $V=65$ characters in $D=128$ dimensions, and the cones are too many to try. The analysis, like the paper, builds the fit greedily. Nonnegative orthogonal matching pursuit starts from $x_J=0$, adds the lens vector (normalised to unit length, which leaves its cone unchanged) whose inner product with the current remainder is largest and positive, refits all the chosen coefficients by nonnegative least squares, and repeats:

::: {.code file="jspace/src/JLens.jl" fn="nnomp" doc=true}
:::

Each step can only reduce the remainder, so the curve of the captured share against $k$ rises. The *occupancy* of a depth is the first $k$ at which one more lens vector captures less than one more random direction: the point where the lens vectors stop beating chance. Defined so, an occupancy of $1$ means that they never beat it: the first lens vector already captures less than the first random direction. @fig-j-occupancy runs it over {{value:jlShPositions}} held-out positions at each of the nine depths.

::: aside
Matching pursuit is due to [Mallat and Zhang (1993)](https://doi.org/10.1109/78.258082), for decomposing signals over redundant dictionaries of time–frequency atoms; the orthogonal variant refits all chosen coefficients at each step. The paper uses gradient pursuit, which replaces the refit by a few gradient steps and is cheaper on large vocabularies.
:::

::: {.figure #fig-j-occupancy}
The J-space of the Shakespeare model, depth by depth. In each panel, against $k=0,\dots,65$: the share of $\lVert x\rVert^2$ captured by greedy nonnegative fits with the lens vectors (dark line) and with random sets of $65$ unit vectors (hairline, the mean of five sets), averaged over held-out positions. The vertical mark is the occupancy, labelled with its $k$; $25$, the paper's typical sparsity, is ticked. At the right of each panel, the share in the full span of the $65$ vectors, for the lens vectors (thick) and for random sets (thin). [Point at a depth for its numbers.]{.web}
:::

The answer is plain. Through depth $2$ the lens vectors capture less of the stream than random directions do, at every $k$, and the occupancy is $1$: they do not beat chance even once. At depths $2\tfrac12$ and $3$ they lead for the first few vectors and then fall behind. Only at depths $3\tfrac12$ and $4$ do they stay ahead, and the largest occupancy before the last depth is {{value:jsShOccEarly}}, at depth {{value:jsShOccEarlyDepth}}. At the last depth it is {{value:jsShOccLast}}: with $25$ vectors the J-space captures {{value:jsShTwentyFiveJ}} of $\lVert x\rVert^2$, random sets {{value:jsShTwentyFiveRandom}}. But at the last depth the lens vectors are the rows of the readout itself, so this says that the final stream is built to be read, which we knew. The spans tell the same story. A random $65$-dimensional subspace of $\R^{128}$ holds $65/128$ of a column on average. The span of the lens vectors holds less than that at every depth before $3\tfrac12$, as little as {{value:jsShSpanMin}} at depth {{value:jsShSpanMinDepth}}, about as much at $3\tfrac12$, and clearly more only at the last depth: {{value:jsShSpanLast}} against {{value:jsShSpanRandomLast}}.

The paper finds occupancy near zero through the first third of its layers, then a plateau of about $25$ across a band of middle layers, and a J-space that never explains more than a tenth of the variance beyond chance. A four-block model of characters shows no band: its lens vectors become a good set of generators for the stream only where they are the readout. The paper's occupancy can be near zero because it is counted per position; ours is bounded below by $1$ by construction, so its early values of $1$ report the same thing. @ch-workspace reads this and the other signatures against the paper's.

## Exercises {.unnumbered}

::: exercise
In the plane, let $v_s$ and $v_u$ meet at an angle $\theta<\pi$. Show that the polar cone of their cone, the set of points whose nearest point is the apex, is a cone of opening angle $\pi-\theta$, bounded by the rays perpendicular to $v_s$ and $v_u$. What happens as $\theta\to\pi$, and what does that say about the pair of @sec-nearest with cosine {{value:jsOppositeCos}}?
:::

::: exercise
Let $U$ be a uniformly random $m$-dimensional subspace of $\R^D$ and $\Pi_U$ the orthogonal projection onto it. Show that $\E\,\lVert\Pi_Ux\rVert^2=\frac mD\lVert x\rVert^2$ for every fixed $x$. (Hint: $\E\,\Pi_U$ commutes with every rotation, and its trace is $m$.) Compare with the random spans of @fig-j-variance and @fig-j-occupancy.
:::

::: exercise
Give three vectors in $\R^2$ and a point $x$ such that the best cone of two of the vectors does not contain the best single vector. Why does this make greedy pursuit an approximation, and why is it exact when $k=1$?
:::

::: exercise
How many cones make up the J-space of the Shakespeare model at $k=25$? Estimate how long it would take to fit each of them by nonnegative least squares at one microsecond each.
:::

::: exercise
Suppose the $V$ lens vectors are linearly independent and $V<D$. Show that $d_V(x)\ge\lVert x-\Pi x\rVert$, with $\Pi$ the orthogonal projection onto their span, and find the condition on the coefficients of $\Pi x$ under which equality holds.
:::

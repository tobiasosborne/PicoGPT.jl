# Is it a workspace? {#ch-workspace}

::: claim
The paper's three structural signatures of a workspace can all be computed on the Shakespeare model: in four blocks it shows no band of depths, no capacity beyond chance and no MLPs that favour J-lens directions, but attention heads that preserve them far above a control; each answer is worth exactly as much as its control.
:::

::: names
| maths | physics | ML |
|---|---|---|
| directions that many later maps read and write | a shared register | global workspace |
| $\inner{K}{L}_F/\lVert K\rVert_F\lVert L\rVert_F$ for two centred Gram matrices | the overlap of two normalised states | centred kernel alignment (CKA) |
| the depths unlike both ends | a phase between two regimes | workspace band |
| the depths before the band | the phase before the transition | sensory layers |
| the depths after the band | the phase after it | motor layers |
| $\E[(z-\bar z)^4]/\sigma^4-3$ over the readout's logits | zero for a Gaussian, large for a spike; blind to the overall scale | excess kurtosis of the readout |
| $\lVert\mathrm{MLP}(\hat v)\rVert$ over its median for random $\hat u$ | a response coefficient | MLP gain |
| $W_o^{(h)}W_v^{(h)}$, of rank at most $d$ | a channel between positions | OV circuit |
| the mean reciprocal rank of $\cos(Mv_i,v_i)$ among the $\cos(Mv_i,v_j)$, less the same for random directions | the fidelity of a channel | label preservation |
:::

## Three signatures, and how to read them {#sec-signatures}

The paper's case that the J-space is a *global workspace* has two halves. One is functional: what the model can report, hold in mind, reason with and route. The other, in its section "Structure Supports Function", is structural: whether the J-space, looked at as an object in the model, has the shape that global workspace theory gives a workspace. It names three signatures. The J-space carries workspace-like content only in an intermediate *band* of depths, between an early regime where it is empty and a late one where it is the imminent output. It has a *limited capacity*: tens of concepts at a time, a small share of the stream. And the model's weights *broadcast* it: MLPs amplify J-lens directions, and some attention heads relay them between positions.

The functional half needs a model that can be asked things. The structural half needs only weights and activations, and every number in it can be computed on the Shakespeare model, with its {{value:wsBlocks}} blocks and {{value:wsDepths}} depths; the specimen, with one block, can answer only some of the questions. This chapter computes them and reports what comes out. It is not a contest. The paper studies models with tens of layers and vocabularies of words; ours has four blocks and an alphabet. What it can teach is how to read such a statistic: before any number, ask what the same statistic would give with no workspace at all (the control), what else it measures besides the thing it is named after (the confound), and how large the difference is compared with the spread of the control (the scale).

Where the paper leaves a choice, or where we could not follow it, we chose, and the choices are these. $J_\ell$ sums over targets where the paper's displayed formula averages over pairs (@ch-jacobian; the paper's pseudocode sums), and it is taken to the final depth where the paper's default lens stops one block earlier. The lens keeps the readout's bias, which the paper's formula omits. No position and no prompt is dropped from the averages. The sparse fits use orthogonal matching pursuit with exact nonnegative refits where the paper uses gradient pursuit, and our occupancy is read off position-averaged curves where the paper's is counted per position and reported as a median. Row (a) of @fig-j-band is top-1 agreement where the paper asks whether any of the top $k$ lens tokens matches, and the paper's fourth signal, the autocorrelation of the top lens token along the text, is not computed. The MLP gain includes the branch's LayerNorm and both of its biases, which the paper does not specify. Each is a plausible reading of the paper; none is the only one, and a verdict that depended on any of them would not be worth much.

## A band of depths {#sec-band}

The paper's simplest evidence for a band compares the lens vectors of different depths directly. At depth $\ell$ there are $V$ lens vectors, the rows of $W_{\text{out}}J_\ell$. Their geometry among themselves is their Gram matrix $K_\ell=(W_{\text{out}}J_\ell)(W_{\text{out}}J_\ell)^\T\in\R^{V\times V}$, the inner products of every pair, computed after centring each coordinate over the $V$ characters. *Linear centred kernel alignment* compares two depths by the cosine of the angle between their Gram matrices, seen as vectors of $V^2$ numbers:
$$\operatorname{CKA}(\ell,\ell')=\frac{\inner{K_\ell}{K_{\ell'}}_F}{\lVert K_\ell\rVert_F\,\lVert K_{\ell'}\rVert_F}\in[0,1],\qquad \inner{K}{L}_F=\sum_{a,b}K_{ab}L_{ab}.$$
It is one when the two sets of vectors are the same up to a rotation of the stream, a common scale and a translation, and it does not care which basis either depth uses. That is the point: the stream's coordinates are not shared between depths (@ch-lenses), and CKA compares only what does not depend on them. It is also why the scale of $J_\ell$, a convention (@ch-jacobian), cannot touch it.

In the paper's models the matrix of CKAs between depths has a clear block structure: an early block of about the first third of the layers, a long middle block, and a short late one. The middle block is the workspace band. @fig-j-cka is the same matrix for our two models.

::: {.figure #fig-j-cka}
Linear CKA between the J-lens vectors of every pair of depths, for both models. The CKAs all lie close to one, so the squares below the diagonal draw the dissimilarity: the area of each is proportional to $1-\operatorname{CKA}$, on one scale for both models. Above the diagonal, the CKA itself. A band would show as small squares inside blocks and large squares between them.
:::

The Shakespeare model has no middle block. Every pair of depths before the last has a CKA of at least {{value:wsCkaInnerMin}}, falling off slowly with the distance between them, and the last depth stands apart, with CKAs between {{value:wsCkaLastMin}} and {{value:wsCkaLastMax}} to all the others. That is one block and the output: the geometry of the lens vectors drifts slowly through the first four blocks' worth of depths and changes at the readout. The specimen's three depths cannot show a band at all.

The paper adds statistics of the lens itself, depth by depth, and finds the same band in each. The J-lens's agreement with the model's own next token is near zero early, rises at the start of the band and jumps at the end. The excess kurtosis of the lens's readout, a measure of how peaked it is, is near zero early, rises at the start of the band and falls in the last layers. And the number of dimensions needed to hold the lens vectors is small early and rises sharply at the band's start. @fig-j-band computes all of these, with chapter 4's occupancy and span share, on the Shakespeare model.

::: {.figure #fig-j-band}
The paper's per-depth signatures of a workspace band, on the Shakespeare model. In every row the J-lens is a short bar in covector ink and its control, where there is one, an open ring: (a) top-1 agreement with the model's next character, against the logit lens; (b) the excess kurtosis of the readout's logits, averaged over positions, against the logit lens; (c) how many singular directions of $W_{\text{out}}J_\ell$ hold 90 % of the sum of its squared singular values; (d) the occupancy of chapter 4; (e) the share of $\lVert x\rVert^2$ in the span of the lens vectors, against random subspaces of the same dimension. [Point at a depth for its numbers.]{.web}
:::

Every row drifts, with small wobbles, towards its value at the last depth, and none has a middle stretch that differs from both ends. The agreement rises late and jumps at the readout; at every intermediate depth it stays below the logit lens's, which is also better at predicting the actual next character, by between {{value:wsAccGapMin}} and {{value:wsAccGapMax}} percentage points (@fig-j-lens-depth). The excess kurtosis is positive {{value:wsKurtosisPositive}}, where the J-lens is the readout; everywhere else it is negative, the logits spread flatter than a Gaussian's (kurtosis does not see the overall scale of the logits: it is a statistic of shape, not of confidence). The effective dimension climbs one or two directions per half block, {{value:wsEffDims}} of {{value:wsRank}}, with the largest step at the readout. In four blocks, every signature of the lens says the same thing: the J-space converges on the output, smoothly, and nothing in it pauses on the way.

That is a statement about four blocks of a model of characters, and about nothing larger. The paper's band begins about a third of the way into models with many more layers. Whether a model of four blocks is too shallow to have one, whether a model of characters has nothing to put in one, or whether the lens fails to see it, these numbers cannot say; and the paper itself notes that its band is found by statistics of the lens, so that an absence in the early layers could be the lens's as much as the model's.

## A limited capacity {#sec-capacity}

The paper measures capacity by the occupancy of @ch-jspace: the number $k$ of lens vectors at which a greedy nonnegative fit stops improving faster with lens vectors than with random directions. In its band the median occupancy plateaus at about 25, and the J-space component explains no more than 10 % of the variance beyond what the same number of random directions explains. A workspace is a small part of the stream, and full.

For the Shakespeare model, row (d) of @fig-j-band gives the occupancy at each depth: {{value:wsOccupancy}}. Through depth $2$ the first lens vector already captures less than the first random direction, so the occupancy is $1$, its smallest possible value; from $2\tfrac12$ to $3\tfrac12$ a lens vector beats a random direction for a step or a few of the fit and then stops; only at the last depth, where the lens vectors are the rows of $W_{\text{out}}$, do they keep beating random directions for long. Read with its control, a small occupancy is not a small workspace. It says that, from the first vector or after a few, the lens vectors describe the stream no better than random directions do, and a capacity is a limit on something that is there.

Row (e) says the same thing without the fit. The span of the $65$ lens vectors is a subspace of $\R^{128}$, and a random subspace of dimension $k$ holds, on average, the fraction $k/D$ of the squared length of any fixed vector (exercise): here $65/128\approx{{value:wsSpanExpected}}$, and the random spans of the analysis hold between {{value:wsSpanRandomMin}} and {{value:wsSpanRandomMax}}. The span of the lens vectors holds less than that at {{value:wsSpanBelow}}: the stream lies preferentially *outside* the J-span until the last block. At the last depth it holds {{value:wsSpanLast}}, which is not a property of a workspace but of a readout: the stream at depth $L$ is what $W_{\text{out}}$ reads. (For the specimen, $7/8\approx{{value:wsSpecSpanExpected}}$ and its random spans hold {{value:wsSpecSpanRandom}} at its three depths; seven directions in $\R^8$ leave little room for any statistic of this kind.)

Here the shape of our models matters most. In the paper there are far more lens vectors than dimensions, the J-space is a union of thin cones in a space they span many times over, and capacity is a question of sparsity. With $V<D$ the lens vectors cannot span the stream at all, and every question about the J-space is first a question about a $65$-dimensional subspace and its complement.

## Broadcast {#sec-broadcast}

### Across depth: the MLPs {#sec-mlp-gain}

The paper asks whether the MLPs amplify J-lens directions more than other directions. Write the MLP branch of block $b$, with its own LayerNorm and biases and without the identity path, as $f_b(x)=W_2\relu(W_1\LN(x)+b_1)+b_2$. For a unit vector $\hat v$, the *MLP gain* of block $b$ is the length of the branch's output, divided by the median of the same over random unit vectors $\hat u$:
$$g_b(\hat v)=\frac{\lVert f_b(\hat v)\rVert}{\operatorname{median}_{\hat u}\lVert f_b(\hat u)\rVert}.$$ The populations are the lens vectors at the MLP's input depth $b-\tfrac12$, the rows of $W_{\text{out}}$ (the lens vectors of the logit lens), the columns of the previous block's $W_2$ (the directions single neurons write), and random directions. In the paper's models the J-lens vectors' gain rises through the band to about ten times the random one, while the neurons' directions stay near one.

::: {.figure #fig-j-gain}
The MLP gain of four populations of unit directions in every block of both models, normalised so that random directions have median one; for each population, the median and the quartiles. The J-lens vectors are taken at each MLP's input depth.
:::

@fig-j-gain has the answer for our models. In the specimen's one block the lens vectors are amplified more than random directions and the rows of $W_{\text{out}}$ less: medians {{value:wsSpecMlpJ}} and {{value:wsSpecMlpLogit}}. In the Shakespeare model the medians of the lens vectors, block by block, are {{value:wsMlpJ}}; those of the rows of $W_{\text{out}}$ are {{value:wsMlpLogit}}, and those of the neurons' directions {{value:wsMlpNeurons}} in blocks 2 to 4. No block favours the lens vectors by much, two amplify them less than random directions, and in the last three the readout's own rows do better.

The statistic has a confound worth naming even where it gives a clear answer. Every population is fed to the MLP alone, a unit vector in place of a column of the stream, and the paper calls its gain a first-pass approximation for that reason: in the model, an MLP never sees a lens vector by itself but a column carrying many things at once, and its rectifier is not linear in the sum.

One caveat applies here as much as to the heads below. The lens vectors at depth $b-\tfrac12$ are rows of $W_{\text{out}}J_{b-1/2}$, and $J_{b-1/2}$ is the averaged derivative through this very MLP; a direction made by pulling a gradient back through a map tends to lie along the directions that map amplifies. The specimen's gain above one is open to this objection, and the remedy is the one proposed for the heads: average $J$ with the block's MLP path removed.

### Across positions: the attention heads {#sec-ov}

Content moves between positions only through attention, and a head moves it through its *OV circuit*. In the notation of [PicoGPT's *Attention*](https://tobiasosborne.github.io/PicoGPT.jl/attention/), with $W_v^{(h)}\in\R^{d\times D}$ the rows of $W_v$ that belong to head $h$ and $W_o^{(h)}\in\R^{D\times d}$ the columns of $W_o$ that do, the head reads the values $W_v^{(h)}\hat x_j$ of the sources and the output map writes $W_o^{(h)}o_i$ into the stream, so apart from the attention weights, LayerNorm and the biases, what a source contributes to a reader passes through the $D\times D$ matrix
$$M_h=W_o^{(h)}W_v^{(h)},\qquad \operatorname{rank}M_h\le d .$$
The paper scores a head on a population of unit directions in two ways. Its *gain* is the mean of $\lVert M_hv\rVert$ over the population, divided by the mean over random directions. Its *label preservation* asks whether the head copies the population faithfully rather than scrambling it: for each $v_i$, rank $\cos(M_hv_i,v_i)$ among all the $\cos(M_hv_i,v_j)$, take the mean reciprocal rank over $i$, and subtract the same score for random directions, so that a head that copies everything indiscriminately scores zero (exercise). The control that matters is the *rotated* J-lens: the same vectors under one fixed random rotation of $\R^D$. It keeps every length and every angle between them, and changes only how they sit relative to the weights, which is exactly the property under test.

::: {.figure #fig-j-heads}
Every attention head of both models, scored on three populations: the J-lens vectors at the head's input depth (bars, labelled block.head), the same vectors rotated (rings, joined to their head's bar) and the rows of $W_{\text{out}}$ (squares). Across: the gain, with random directions at one. Up: label preservation, with random directions at zero. [Point at a head for its numbers.]{.web}
:::

This is the one signature the Shakespeare model shows clearly. In all {{value:wsHeads}} heads the J-lens vectors are preserved better than their rotation, with label preservations between {{value:wsHeadJMin}} and {{value:wsHeadJMax}} against {{value:wsHeadRotMin}} to {{value:wsHeadRotMax}} for the rotated set and at most {{value:wsHeadLogitMax}} for the rows of $W_{\text{out}}$; the heads of the first block preserve them most. The J-lens vectors also have the larger gain in {{value:wsHeadGainWins}} of the {{value:wsHeads}} heads. The specimen's two heads show nothing of the kind: their label preservations for the J-lens vectors, {{value:wsSpecHeadJ}}, are matched by the rotated set's, {{value:wsSpecHeadRot}}.

Now read it as the paper's readers should. The control is good: rotating the set removes nothing but the orientation, so whatever separates the bars from the rings is about orientation relative to these weights. But the J-lens vectors are not independent of these weights. The lens vectors at the input of block $b$ are rows of $W_{\text{out}}J_{b-1}$, and $J_{b-1}$ is the averaged derivative of everything after depth $b-1$, block $b$'s attention included: @ch-jacobian computed it by pulling back through these very heads. A set of directions defined through a map can be aligned with that map by construction, and the rotated control, which destroys every alignment at once, cannot tell an alignment inherited from the definition from one that training arranged for broadcast. A control that could would compute the J-lens with the head's path removed, and test the head on that. Nor is preservation the same as use. The paper's functional test removes the heads and measures what is lost from the J-space downstream (recall of the top 25 lens tokens falls to 0.67, against 0.86 for random heads); we have not done that.

## What the small model says {#sec-verdict}

Gathered up: no band of depths; no capacity that beats the control; MLPs that do not favour J-lens directions; attention heads that do preserve them, with a reason to suspect part of that is built into the definition. None of this contradicts the paper, whose claims are about production models, far larger and with vocabularies of words; the paper itself lists, among its open questions, whether smaller models have a workspace at all. What it shows is that the signatures are not automatic. A transformer trained on next-character prediction, analysed with the paper's definitions as we have implemented them (@sec-signatures says where the implementation chooses), does not produce them by default, so where they appear in large models they say something about those models.

The statistics also share a property that is easy to forget: all of them are computed from the lens. A band in a statistic of the J-lens is a band in the J-lens, and only through it in the model. The averaged Jacobian of @ch-jacobian is an approximation of the rest of the model by one linear map, and for the Shakespeare model it is, at every intermediate depth, a slightly worse map to the output than the identity, by agreement with the model, by accuracy and by divergence from the model's distribution. A workspace found through it can be no sharper than it is.

::: aside
Global workspace theory is due to Bernard Baars (*A Cognitive Theory of Consciousness*, 1988); Stanislas Dehaene and colleagues made it a neural theory, with "ignition" and broadcast across cortex ([Dehaene, Kerszberg and Changeux, 1998](https://doi.org/10.1073/pnas.95.24.14529)). The paper uses the theory as a list of functional and structural properties to test, and takes no position on consciousness; neither does this book. Linear CKA was introduced for comparing neural representations by [Kornblith et al. (2019)](https://arxiv.org/abs/1905.00414).
:::

## In the code {#sec-code-workspace}

CKA is two Gram matrices and one inner product:

::: {.code file="jspace/src/JLens.jl" fn="linear_cka"}
:::

The MLP gains and the OV statistics of every head are computed from the weights alone, with the populations normalised to unit length:

::: {.code file="jspace/src/JLens.jl" fn="structure_statistics"}
:::

## Exercises {.unnumbered}

::: exercise
Show that linear CKA is unchanged if the vectors of one depth are all rotated by the same orthogonal matrix, or all multiplied by the same positive number. Show by an example that it changes under a general invertible linear map. Why did the change of scale of $J$ between the two exports of the analysis leave the CKAs unchanged?
:::

::: exercise
Let $\Pi$ be the orthogonal projection onto a random $k$-dimensional subspace of $\R^D$, uniformly distributed. Show that $\mathbb E\lVert\Pi x\rVert^2=(k/D)\lVert x\rVert^2$ for every fixed $x$. (Hint: average over the rotations that carry one subspace to another, and use $\operatorname{tr}\Pi=k$.) Compare $65/128$ and $7/8$ with the random spans of the two exports.
:::

::: exercise
Show that a head with $M_h=cI$, $c>0$, gives every population, random ones included, the largest possible mean reciprocal rank, so that its label preservation is zero. Which heads score a large positive label preservation on a population $\{v_i\}$?
:::

::: exercise
Show that $V$ logits of which one is $h>0$ and the others zero have excess kurtosis $\dfrac{(V-1)^3+1}{V(V-1)}-3$, close to $V$ for large $V$, and that $V$ logits spaced evenly on an interval have excess kurtosis tending to $-\tfrac65$. Where between these do the Shakespeare model's readouts lie?
:::

::: exercise
The rotated control keeps the Gram matrix of the J-lens vectors. Describe a control that would test whether the label preservation of a head is inherited from the definition of $J$: what would you compute, and what result would make you believe the head was arranged to broadcast?
:::

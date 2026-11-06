# Prologue {#ch-prologue}

::: claim
The paper reads what a language model is poised to say, at every depth of its computation, by replacing the rest of the model with one matrix: the model's derivative, averaged over many contexts. This book computes that matrix, and everything the paper builds on it, on two models small enough to see whole.
:::

::: names
| maths | physics | ML |
|---|---|---|
| a column $x=X_\ell[:,t]\in\R^D$ of the stream at depth $\ell$ | the state | activation, residual stream at layer $\ell$ |
| the depth $\ell\in\{0,\tfrac12,1,\dots,L\}$ | the time of the Euler steps | layer |
| the readout $W_{\text{out}}\in\R^{V\times D}$ | a measurement | unembedding, $W_U$ |
| $J_\ell\in\R^{D\times D}$, a derivative averaged over contexts | an averaged linear response | the lens matrix |
| $v_a=J_\ell^\T W_{\text{out}}^\T e_a$, an averaged gradient | a covector, a static susceptibility | J-lens vector |
:::

## What the paper claims {#sec-claim}

In July 2026 a group at Anthropic published a long paper with a short thesis: *Verbalizable Representations Form a Global Workspace in Language Models*, by Wes Gurnee, Nicholas Sofroniew, Jack Lindsey and colleagues. A language model computes, at every position of its input and at every depth of its computation, a vector. The paper looks inside these vectors for the part that the model could put into words, and finds it with a construction simple enough to fit on one line. For every token $a$ of the vocabulary and every depth $\ell$ it computes one vector $v_a$, the *J-lens vector*. Read against the state $x$ at depth $\ell$, the number $\inner{v_a}{x}$ measures how strongly $x$ pushes the model towards saying $a$, now or at any later position: to first order, and on average over many contexts. A state with a large reading on $v_a$ is *poised to say* $a$, whether or not it ever does.

The paper then calls the set of states that are sparse nonnegative combinations of these vectors the *J-space*, and argues that it behaves like what neuroscience calls a *global workspace*: a small, shared stage on which a few contents at a time are held, reported, steered, reasoned with and broadcast to the rest of the system. The evidence is of two kinds. Behavioural: the model is asked what it is thinking about, told to hold a concept in mind, or interrupted mid-task, and the J-space is read and edited while it answers. Structural: in which band of depths the J-space carries content, how much of the state it occupies, and whether the model's weights treat its directions differently from others.

::: aside
In the paper: the construction is in *Methods*, "The Jacobian Lens" and "The J-Space"; the structural tests are in "Structure Supports Function". The paper is at [transformer-circuits.pub/2026/workspace](https://transformer-circuits.pub/2026/workspace/index.html). We cite it by section name throughout.
:::

## The one idea {#sec-idea}

Everything in the paper rests on one mathematical idea, and this book is about that idea: *replace the rest of the model by its averaged derivative*.

Take the stream at some depth $\ell$, the matrix $X_\ell\in\R^{D\times N}$ of [PicoGPT's chapter on the block](https://tobiasosborne.github.io/PicoGPT.jl/block/), one column per token. The rest of the model is a map $F_\ell:X_\ell\mapsto X_L$ from the stream at depth $\ell$ to the stream at the last depth, followed by the readout
$$z=W_{\text{out}}\LN(x)+b_{\text{out}},\qquad x=X_L[:,t],$$
which turns a final column into the scores of the $V$ characters. The map $F_\ell$ is not linear and it mixes columns, through attention. Its derivative at one input is a large matrix, and it changes from input to input. Average it over positions and over many texts, in a way @ch-jacobian makes precise, and what is left is one $D\times D$ matrix $J_\ell$. Then read the column $x$ at depth $\ell$ as if the rest of the model were $J_\ell$:
$$\operatorname{lens}_\ell(x)=\softmax\bigl(W_{\text{out}}\LN(J_\ell x)+b_{\text{out}}\bigr).$$
That is the *Jacobian lens*. The rows of $W_{\text{out}}J_\ell$ are the $V$ lens vectors $v_a^\T$. Each is, by construction, an averaged gradient: of the score of one character, taken before the final normalisation, with respect to the stream at depth $\ell$. It is a covector, and it is drawn in this book's covector ink.

@fig-j-idea shows the idea on the smallest model we own, for one column of one window. The column runs down its stem through the three depths of that model, and at the foot the model reads it out. From each depth a bypass leaves the stem, passes through one matrix, and reads out at once.

::: {.figure #fig-j-idea window=1 column=6}
The idea, on one column of the specimen. Down the stem, the real model: the column at depth $0$ (after the embedding), at depth $\tfrac12$ (after attention) and at depth $1$ (after the MLP), and at the foot the model's own distribution over the next character. From each depth a bypass replaces everything below it by one matrix $J_\ell$, the averaged derivative, and reads out at once. At depth $1$ nothing is left to replace, and the bypass gives the model's own readout. [Click a character to choose the column, and point at a depth.]{.web}
:::

Why should an *averaged* derivative be the right object, and not the derivative at the text in hand? The paper's answer, which @ch-jacobian takes apart, is that the derivative at one text mixes two things: what the model is disposed to do with a direction in general, and what it happens to be doing with it in this text. The average keeps the disposition. The title of this book is the thesis in two words: the lens reads what the state is *poised to say*.

## Two models {#sec-models}

The paper works on Anthropic's production models. We work on the two models of [PicoGPT](https://tobiasosborne.github.io/PicoGPT.jl/), which we own completely: every parameter, every training step, every pullback written by hand.

*The specimen* has {{value:specimenParams}} parameters. It reads windows of $T=8$ characters of the line *to be or not to be*, with a vocabulary of $V=7$ characters and a stream of width $D=8$, through one block of two heads. Its stream has three depths, $0$, $\tfrac12$ and $1$, and every derivative in this book can be drawn for it as $8\times8$ squares. The averages in it are exact over its training batch: all ten windows, all positions and all pairs of positions.

*The Shakespeare model* has {{value:shakespeareParams}} parameters: $V=65$ characters, $D=128$, four blocks of four heads, contexts of $T=256$ characters, trained on Tiny Shakespeare. Its stream has nine depths. It is big enough for its lens readouts to carry structure, and small enough that its averaged derivatives can be computed by the hand-written pullbacks at every position of held-out text.

Depth is counted in half blocks: $X_0$ is the stream after the embedding, $X_{\ell-1/2}$ the stream after the attention update of block $\ell$, and $X_\ell$ the stream after its MLP update. In PicoGPT's notation for the specimen, $X_{1/2}$ is $X_{\text{mid}}$ and $X_1$ is $X_{\text{out}}$. The last depth is $L$, which is $1$ for the specimen and $4$ for Shakespeare.

One difference of shape matters throughout. The paper's models have vocabularies of word pieces far larger than the width of their streams, $V\gg D$, so the $V$ lens vectors at a depth are far more numerous than the dimensions they live in, and any state can be written as a combination of them in infinitely many ways. Ours have vocabularies of characters, and $V<D$: seven vectors in $\R^8$, sixty-five in $\R^{128}$. They span at most a $V$-dimensional subspace, and a state is in general not a combination of them at all. Several statements of the paper therefore change shape here, and where one does, we say so rather than force it.

## What we cannot do {#sec-cannot}

Most of the paper's evidence is behavioural. It asks a model that can talk what it is thinking about, tells it to hold a concept in mind, interrupts it with questions, audits it for hidden goals, and trains it to reflect. Our models do not talk: they continue text one character at a time. Those experiments are out of scope, and the epilogue says in a paragraph each what they are and why we cannot run them. What we can do is compute every mathematical object the paper defines, on models where nothing is hidden, and run its structural tests honestly. If a four-block character model shows no workspace, that is a result, and we report it as one.

The paper uses the language of a theory of consciousness, and is careful to say that it means only the functional notion: what information is available for report, reasoning and control, not whether anything is experienced. It takes no position on the philosophy, and neither do we. In this book "poised", "workspace" and "verbalizable" mean exactly what their definitions say, and nothing more.

::: aside
The global workspace theory is Bernard Baars's, from *A Cognitive Theory of Consciousness* (Cambridge University Press, 1988). Its neuronal version is due to Stanislas Dehaene, Michel Kerszberg and Jean-Pierre Changeux, [*A neuronal model of a global workspace in effortful cognitive tasks*](https://doi.org/10.1073/pnas.95.24.14529) (PNAS, 1998). The paper borrows the theory's functional signatures, not its claims about brains.
:::

## How to read this book {#sec-how}

The book assumes PicoGPT, or at least three of its chapters: [the block](https://tobiasosborne.github.io/PicoGPT.jl/block/), for the residual stream and LayerNorm; [attention](https://tobiasosborne.github.io/PicoGPT.jl/attention/), for the only map that mixes columns; and [pulling back](https://tobiasosborne.github.io/PicoGPT.jl/pulling-back/), for the pullback as the transpose of the derivative. It keeps PicoGPT's conventions: the column is the token, forward flows down the page, every object has its three names, and two inks. Value ink marks what is computed going forward: the stream, the parameters, the matrices $J_\ell$ as maps of the stream. Covector ink, with bars and hatching, marks what is pulled back: gradients, and therefore the lens vectors.

@ch-lenses applies the readout to the stream at every depth, which is the older *logit lens*, and shows why it fails early. @ch-jacobian builds the averaged derivative $J_\ell$ and computes it by pullbacks. @ch-jlens defines the Jacobian lens and its vectors and reads both models with it. @ch-jspace defines the J-space, a union of cones, and measures how much of the stream lies in it. @ch-writing edits the stream along lens vectors and watches the output change. @ch-workspace runs the paper's structural tests of a workspace on the Shakespeare model, and the epilogue collects what we could not do.

## Exercises {.unnumbered}

::: exercise
Count. For each model, how many lens vectors are there at one depth, in which space do they live, and how many numbers are there in all the matrices $J_\ell$ together, over every depth? Compare with the number of parameters of each model.
:::

::: exercise
Suppose the rest of the model acted on each column by the same linear map, $F_\ell(X)=AX$. Show that its derivative is $A$ at every input, that every average of it is $A$, and that the Jacobian lens with $J_\ell=A$ reads exactly what the model says. Which parts of a transformer block make $F_\ell$ nonlinear, and which one makes it act on more than one column at a time?
:::

::: exercise
LayerNorm with $\varepsilon=0$ does not see the length of a column: $\LN(cx)=\LN(x)$ for every $c>0$. Deduce that replacing $J_\ell$ by $cJ_\ell$, $c>0$, leaves the Jacobian lens unchanged when $\varepsilon=0$. Which quantities built from the lens vectors $v_a$ would still change? (@ch-jacobian uses this freedom to fix the scale of $J_\ell$, and @ch-jlens shows what $\varepsilon>0$ does to it.)
:::

::: exercise
Let $v_1,\dots,v_V\in\R^D$. Show that if $V>D$ they are linearly dependent, so that a vector in their span has infinitely many expansions $x=\sum_ac_av_a$. Show that if $V\le D$ and they are independent, a vector in their span has exactly one, and that a vector outside it has none. Which case is the paper's, and which is ours?
:::

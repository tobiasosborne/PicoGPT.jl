# The task {#ch-task}

::: claim
A text is a finite sequence over a finite alphabet. By the chain rule of probability, a distribution on texts is the same thing as a rule that sends every context to a distribution over the next symbol, and a language model is such a rule.
:::

::: names
| maths | physics | ML |
|---|---|---|
| a finite set $\mathcal A$ with $V=\lvert\mathcal A\rvert$ elements | the single-site state space | vocabulary |
| $a\in\{1,\dots,V\}$, the index of a symbol | a label of a state | token, token id |
| $x=(x_1,\dots,x_m)\in\mathcal A^m$ | a configuration, a history | sequence, text |
| the prefix $x_{\le t}=(x_1,\dots,x_t)$ | the past | context, prompt |
| $q(\,\cdot\mid x_{\le t})\in\Delta^{V-1}$ | a conditional law, a transition kernel | next-token prediction |
| $P(x)=\prod_t q(x_{t+1}\mid x_{\le t})$ | a chain of conditional experiments | autoregressive model |
| the largest $t$ the map accepts, $T$ | the length of the memory | context length, context window |
| $-\log q(x_{t+1}\mid x_{\le t})$ | surprisal | per-token loss |
| the pairs $(x_{\le t},x_{t+1})$ we fit | the sample | training set |
:::

## Text as a finite sequence {#sec-sequence}

Collect the distinct characters of a text and sort them. That is the *alphabet* $\mathcal A$, a finite set with $V$ elements, and from here on its elements are known only by their position in the sorted list, $a\in\{1,\dots,V\}$. A text of length $m$ is then a point $x\in\{1,\dots,V\}^m$: a word of $m$ letters over $V$ symbols, nothing more. The model never sees a character, only the integer that stands for it.

The book's specimen is trained on one line of $m=18$ characters, `to be or not to be`. Its alphabet has $V=7$ symbols, the space first because it sorts first: ␣ b e n o r t. As a word over $\{1,\dots,7\}$ the line begins $(7,5,1,2,3,1,5,6,\dots)$. The numbering is arbitrary and means nothing: the model will have to learn that `r` may follow `o`, and the fact that they are numbered 5 and 6 does not help it. @ch-vectors makes sure that the numbers cannot even suggest it.

::: aside
The space is a symbol like any other. Figures draw it as ␣ so that it can be seen.
:::

## The chain rule {#sec-chain-rule}

Let $P$ be any probability distribution on $\mathcal A^m$. Applying the definition of conditional probability $m-1$ times gives
$$P(x_1,\dots,x_m)=P(x_1)\prod_{t=1}^{m-1}P(x_{t+1}\mid x_1,\dots,x_t).$$
This is the *chain rule of probability*. It assumes nothing: no independence, no Markov property, no stationarity. It only says that one may draw a text one symbol at a time, each from its distribution given everything before it.

The converse is just as true, and it is the one we use. Take *any* family of distributions $q(\,\cdot\mid c)$ on $\mathcal A$, one for every context $c$. The product $P(x)=q(x_1)\prod_tq(x_{t+1}\mid x_{\le t})$ is non-negative, and it sums to one: summing over $x_m$ removes the last factor, then summing over $x_{m-1}$ removes the next, and so on down to $x_1$. So a distribution on texts and a rule "context $\mapsto$ distribution of the next symbol" are the same object, written in two ways. A model of the second kind is called *autoregressive*: it writes by feeding its own output back in as the next context.

## A language model is a map {#sec-map}

Write $\Delta^{V-1}=\{p\in\R^V: p_a\ge 0,\ \sum_ap_a=1\}$ for the probability simplex: the distributions on the alphabet. A *language model* with context length $T$ is a map
$$q_\theta:\ \bigcup_{t=1}^{T}\mathcal A^t\longrightarrow\Delta^{V-1},\qquad x_{\le t}\longmapsto q_\theta(\,\cdot\mid x_{\le t}),$$
from contexts of up to $T$ symbols to distributions over the next one. The subscript $\theta\in\R^p$ is the list of numbers that fixes which map we mean: the *parameters*. Everything in the rest of the book is about how $q_\theta$ is computed from $\theta$ (@ch-vectors to @ch-block), and how $\theta$ is chosen (@ch-surprise to @ch-learning).

Cutting the context at $T$ symbols is a modelling decision, not a consequence of the chain rule; for the specimen, $T=8$. A context of $t$ symbols is a point of a finite set with $V^t$ elements, so in principle $q_\theta$ is a table. The table is big. A transformer is a way of writing it down with far fewer numbers than it has entries, in a form that can be adjusted a little at a time.

@fig-contexts-to-distributions shows the trained specimen's map on the eight prefixes of one window. Each column is a context, growing by one character, and below it the distribution the model assigns to the character that comes next.

::: {.figure #fig-contexts-to-distributions window=2}
The specimen as a map from contexts to distributions. Column $t$ is the context $x_{\le t}$, the first $t$ characters of the window; below it are the seven probabilities $q_\theta(a\mid x_{\le t})$, one bar per character $a$, in the order of the alphabet. The character that actually comes next is marked. Drag the window along the line.
:::

Most columns are a single bar of height one: after `o be o`, the specimen is sure that `r` comes next, and it is right. The first column is different. Given only `o`, the specimen splits its mass almost evenly between ␣ and `r`. That is not a mistake, and we come back to it in @sec-floor.

## The probability of a text {#sec-probability}

Given a model, the chain rule assigns a probability to every text. For a window of $T$ characters followed by one more, the model gives the last $T$ of them, each conditioned on what precedes it:
$$q_\theta(x_2,\dots,x_{T+1}\mid x_1)=\prod_{t=1}^{T}q_\theta(x_{t+1}\mid x_{\le t}).$$
The first character is given, never predicted: there is nothing yet to predict it from. Products of many numbers below one are awkward, so we take logarithms. The *surprise* of one prediction is
$$s_t=-\log q_\theta(x_{t+1}\mid x_{\le t})\ \in[0,\infty],$$
measured in *nats* because the logarithm is natural. A certain and correct prediction costs nothing; a prediction that gave the truth probability $1/V$, as uniform guessing does, costs $\log V$; a prediction that ruled the truth out costs infinitely much. The surprises of a text add up, and the probability of the text is the exponential of minus their sum.

::: {.figure #fig-chain-rule window=2}
The chain rule as a staircase. Above each character $x_{t+1}$ of the window is the probability the model gave it in its context, $q_\theta(x_{t+1}\mid x_{\le t})$; the staircase below accumulates the surprises $-\log q_\theta$, and its height at the end is minus the log-probability of the whole text. A hairline marks what uniform guessing would cost. Click a character to see the whole distribution it was drawn from; drag the window.
:::

The mean surprise per prediction, averaged over all the predictions we have, is the number that training will make small. @ch-surprise derives it properly, as a cross-entropy, and shows that making it small is maximum likelihood.

::: aside
The unit matters little: in bits, using $\log_2$, every surprise is multiplied by $1/\ln 2$. Nats keep the derivatives clean.
:::

## The training set {#sec-windows}

The model accepts at most $T$ symbols, so we cut the corpus into *windows*: every run of $T$ consecutive characters that has one more character after it. For a window starting at $s$, the *input* is $x_s,\dots,x_{s+T-1}$ and the *target* is the same run shifted by one, $x_{s+1},\dots,x_{s+T}$. Position $t$ of the window is asked for the character that follows its prefix. A corpus of $m$ characters has $m-T$ windows; the specimen's line has ten.

::: {.figure #fig-training-windows}
The whole training set. The corpus is at the top; below it, each row is one window, its inputs in ink and, offset by one place, its targets in value ink. Every target is one prediction, and the ten windows of eight make the $N=T\cdot B$ columns that the model processes at once.
:::

One window yields $T$ predictions, not one. A single pass of the model computes all $T$ distributions of a window together, each conditioned only on its own prefix; @ch-attention shows how the model keeps position $t$ from reading positions after $t$, so that the targets are not visible to the predictions they test. Stacking the $B$ windows side by side gives the $N=T\cdot B$ columns of the arrays in every later chapter. For the specimen, $B=10$ and $N=80$.

The windows overlap, so the same stretch of text is predicted several times, from contexts of different lengths. That is deliberate. It also means that the objective is an average over these 80 conditional predictions, not the probability of the line counted once.

## What no model can learn {#sec-floor}

Could a model make every surprise zero? Only if every context determines its next character. Look at the first position of each window, where the context is a single character. Two windows begin with `o`: in one the next character is ␣ (from `to be`), in the other it is `r` (from `or`). Three begin with ␣, followed by `b`, `o` and `n`. A model sees the same context in each case, so it must give the same distribution, and no distribution puts probability one on two different characters.

The best a model can do with a context $c$ that occurs $n_c$ times, followed $n_{c,a}$ times by $a$, is to predict the observed frequencies $q(a\mid c)=n_{c,a}/n_c$. (That this is optimal is Gibbs' inequality, proved in @ch-surprise.) The mean surprise is then the entropy of the text's own next-character statistics,
$$H_{\min}=\frac1N\sum_c\sum_a n_{c,a}\log\frac{n_c}{n_{c,a}}.$$
Every other context of the specimen's windows is followed by one character only, and contributes nothing. The two windows that start with `o` each cost at least $\log 2$, the three that start with ␣ each cost at least $\log 3$, and so
$$H_{\min}=\frac{2\log 2+3\log 3}{80}\ \text{nats}.$$
It is a property of the text and of the way we cut it, not of any model. A perfect specimen would sit exactly on this floor, and @ch-learning shows how close training gets.

::: aside
The floor depends on the cut as well as on the text: windows of another length see other contexts, and a longer line has other ambiguities. @ex-floor-longer asks for one.
:::

::: {.figure #fig-entropy-floor}
The floor under the loss. The 80 predictions of the training set, laid out as in @fig-training-windows; each is drawn at the smallest surprise any model can achieve on it. Only five are not free, all at the first position of a window. Point at one to see which characters share its context.
:::


## In the code {#sec-code}

The alphabet and the change of names between characters and integers are three short functions:

::: {.code file="src/text.jl" fn="alphabet"}
:::

::: {.code file="src/text.jl" fn="encode"}
:::

and the windows, with their targets shifted by one, are one line:

::: {.code file="src/train.jl" fn="windows"}
:::

Column $b$ of `x` is window $b$, so a batch of $B$ windows is a $T\times B$ array of integers, read down its columns, as every array in this book is.

## Exercises {.unnumbered}

::: exercise
A table that lists $q(\,\cdot\mid c)$ for every context $c$ of length $1$ to $T$ has $V-1$ free numbers per context. Show that it has $V^{T+1}-V$ free numbers in all, and evaluate this for the specimen. How does it compare with the specimen's parameter count in @ch-prologue?
:::

::: exercise
A *bigram* model is a language model whose prediction depends on the last symbol only, $q(\,\cdot\mid x_{\le t})=q(\,\cdot\mid x_t)$. How many free numbers does it have? Find the bigram model with the smallest mean surprise on the specimen's 80 predictions, and show that its mean surprise is larger than $H_{\min}$. Which contexts does a bigram model confuse that the specimen does not?
:::

::: {.exercise #ex-floor-longer}
Show that $H_{\min}=0$ exactly when every context in the training set is followed by one character only. Then compute $H_{\min}$ for the line `to be or not to be or` with $T=8$. Which contexts are now ambiguous?
:::

::: exercise
Try it in @fig-chain-rule. Find a window in which the trained model's staircase is not flat. At which position does it rise, and by how much, compared with the floor of @fig-entropy-floor?
:::

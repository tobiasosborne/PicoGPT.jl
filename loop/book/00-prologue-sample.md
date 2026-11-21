# Prologue {#ch-prologue}

::: claim
A looped transformer is one block, with the same parameters at every pass, applied again and again to its own output; on a language of `be` and `not`, with a window of two, information travels one word per pass, and the loss falls to a floor $\ell^*$ set by how far that information has travelled.
:::

::: names
| maths | physics | ML |
|---|---|---|
| a map iterated on its own output, $X_{k+1}=G(X_k)$ | a discrete-time dynamical system | looped, weight-tied, recurrent-depth transformer |
| the pass index $k$ | time | loop, recurrence, iteration |
| the support $\{t-k,\dots,t\}$ after $k$ passes | a light cone, a Lieb--Robinson bound | receptive field |
| a mask with band width $2$ | nearest-neighbour coupling | sliding-window attention |
| $s_t=a_ts_{t-1}$, with $a_t=\pm1$ | the transfer matrix of an Ising chain | state tracking |
| $X^*=G(X^*)$ | a steady state, an equilibrium | fixed point |
| a loss at every pass | --- | deep supervision |
:::

## What this is {#sec-what}

This book, *Not Not To Be*, is about one map used many times. The reader has already met the map: it is the block of the first book, in which attention mixes the columns of the residual stream and a small network acts on each column, with normalisations and residual additions. Nothing in the block is new. What is new is what we do with it.

A stack applies different blocks in succession, each with its own parameters. A loop applies the same block in succession. Write $G$ for the block as a map on the stream and $k$ for the pass. The model is the iteration
$$X_{k+1}=G(X_k),$$
with the same $\theta$ inside $G$ at every pass. An extra pass costs time; it costs no parameters. The question is therefore not only what a composition computes when evaluated once, but what an iteration computes as $k$ grows, and whether it settles. At a point with $X^*=G(X^*)$ further passes do nothing: a fixed point. Running more passes than were trained is extrapolating in time. Thinking longer, here, is not a metaphor. It is a larger $K$, the number of passes.

## The specimen {#sec-specimen}

The protagonist reads a language of two words, `be` and `not`, and says after every word whether the sentence so far affirms or denies. Thus `not not be` affirms, and `not not not be` denies. The meaning of a sentence so far is finite and explicit: the parity, even or odd, of the number of its `not`s. Even affirms, odd denies. In signs, $s_t=a_ts_{t-1}$, where `be` keeps the sign and `not` reverses it. Every word matters to every later answer: change an early `be` to a `not` and all later answers reverse. The first word must be carried to the end.

One extra rule decides how it is carried. Each word may attend only to itself and its predecessor: a window of two. Attention is the only place where columns mix, so in one pass the news of a word can reach the next word and no further. There is no shortcut from the start of a sentence to its end; the parity of the early `not`s must be handed along, word by word, pass by pass.

As a transformer, the specimen keeps every part of the known block, with learned attention over the window, both LayerNorms, a ReLU MLP, an embedding and a readout. Only the sizes are small, $V=2$, $D=4$, $H=1$, $F=16$, $L=1$, and
$$270=2VD+V+2D+4D^2+2DF+9D+F.$$
The same $270$ parameters serve at every pass. Training takes a loss at every pass, on lengths $4$--$12$, with $K$ from $T-1$ to $2T$ passes for a sentence of length $T$.

::: aside
The plain block, looped unchanged, does not learn this reliably. Besides the loop, four small changes, each a line or two and each answering a measured failure, are needed. They are introduced at the failures that forced them; the failures are part of the result.
:::

## The light cone and the floor {#sec-cone}

After $k$ passes, the column at word $t$ can depend only on words $\{t-k,\dots,t\}$, cut off at the start of the sentence. The first word has reached word $t$ only when $k\ge t-1$. The boundary $t=k+1$ is a light cone: inside it the news has arrived, outside it has not. The name is borrowed from physics; the bound is not. Information here has a maximum speed, one word per pass.

::: {.figure #fig-cone}
The cone filling in. Words run left to right, passes top to bottom, one cell for each word at each pass. Its square in value ink has area $2\,|p-\tfrac12|$, for the probability $p$ given to the correct answer, affirm or deny: full at certainty, absent when undecided at probability $\tfrac12$, filled when the model's answer is right and open when it is wrong. The empty region is outside the cone, squares appear inside it, and the boundary $t=k+1$, labelled on the plate, advances one word per pass.
:::

Outside the cone the earliest words are unseen at word $t$, and their `not`s can still reverse the answer either way. The best possible prediction is probability $\tfrac12$ each way, costing $\ln 2$ nats. Inside, a prediction can be exactly right. At pass $k$ the undecided words number $\max(T-k-1,0)$; counting them over $T$ words and $K$ passes gives the least loss under the window rule,
$$\ell^*=\frac{\ln 2}{TK}\sum_{k=1}^{K}\max(T-k-1,0),\qquad \ell^*=\frac{21\ln 2}{64}=0.227438919\ \text{nats at }T=K=8.$$

This is an anchor of the same kind as the entropy floor of the first book. There, learning a line almost perfectly was not falling short: the remainder was what the context genuinely left undecided, and the floor measured it exactly. Here the undecided part has a shape, and the shape moves. A loss at $\ell^*$ is a model right wherever information has arrived and honestly undecided wherever it has not. No training can improve on information that is not there yet.

Over the mixture of lengths and pass counts used in training, the same counting gives $0.161350952$ nats. Four of five seeds reach that floor to within $1\%$, the best $+0.0033\%$ above it; the remaining seed fails, and will be shown failing. With the same window but one pass only, the floor on the same mixture is $0.497544536$ nats at any width. Width cannot substitute for news that has had no time to travel.

## What you will see {#sec-what-you-will-see}

The four seeds that reach the floor are $100\%$ right at lengths $32$ and $64$, with $K=T$ and $K=2T$ passes, although training saw no sentence beyond length $12$ and no more than $24$ passes. The parameters are still the same $270$. What was learned is a local rule, good for longer sentences and for more passes at once. At length $64$, pass $128$, attention puts weight $0.967$--$1.000$, mean $0.989$, on the predecessor, and another pass moves the stream by a centred RMS step of $9\cdot10^{-16}$: the fixed point, reached and then left alone by further passes. The cone will also be seen backwards, when the loss is pulled back through the loop in covector ink and a covector travels back one word per pass.

The picture to keep is @fig-cone. On the first passes only the shortest beginnings are settled, and the rest is empty paper. Then the boundary moves, one word at a time, and the filled region grows behind it, affirmations and denials appearing in turn, until the whole sentence is understood and further passes change nothing. That filling cone, pass by pass, is the subject of this book.

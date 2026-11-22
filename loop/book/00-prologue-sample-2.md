# Prologue {#ch-prologue}

::: claim
When a calculation needs more dependent steps than a transformer has layers, it can either write intermediate answers into the text and read them back, or iterate one block on its own state; on a two-word language with a window of two, each pass can move news at most one word, turning serial steps into a visible distance and setting an exact loss floor.
:::

::: names
| maths | physics | ML |
|---|---|---|
| iterating one map, $X_{k+1}=G(X_k)$ | a discrete-time dynamical system | looped, weight-tied, recurrent-depth transformer |
| the pass index $k$ | time | loop, recurrence, iteration |
| emitting $\hat y_j$ into the next input | externalising an intermediate state | scratchpad, chain of thought |
| a mask of band width $2$ | nearest-neighbour coupling | sliding-window attention |
| possible dependence on $\{\max(1,t-k),\dots,t\}$ after $k$ passes | a light cone, a Lieb--Robinson bound | receptive field |
| $s_t=a_t s_{t-1}$ with $a_t=\pm 1$ | transfer matrix of a zero-field Ising chain, on signs | state tracking |
| $X^*=G(X^*)$ | steady state, equilibrium | fixed point |
:::

## Think inside, or write it down? {#sec-question}

A transformer of $L$ layers applies $L$ maps in succession before it speaks. Whenever a problem needs more dependent steps than $L$, the model has two ways out.

One way is to *write it down*: emit an intermediate token, feed it back into the input, and run the $L$ layers again. Each written token buys another round of computation; machine learning calls this external state a *scratchpad* or *chain of thought*. The other way is to *think inside*: write $G$ for a block on the residual stream and apply it again to its own output,
$$X_{k+1} = G(X_k), \qquad k = 0, 1, \dots, K-1,$$
with the same parameter vector $\theta$ at every pass. An extra pass spends time, not parameters, and leaves the text unchanged in length. Running more passes than were trained is extrapolating in time.

Real models test this trade-off. Inkling 2 Mini ($26.4$ billion parameters, $3.4$ billion active per token before looping) splits $24$ layers into four blocks of six: the outer two run once per token and the middle $12$ run three times with shared weights, giving $48$ layer applications from $24$ stored layers (about $5.4$ billion effective active parameters). In its variable-loop research variant, trained in later midtraining on $1$--$5$ passes ($1:2:6:2:1$), silent chained lookups (N-Hop Lookup) kept improving through $10$ passes, while measured closed-book recall barely moved (SimpleQA Verified: $6.3\%$, $7.3\%$, $6.7\%$ at $1$, $3$, $5$ passes). On GPQA Diamond questions solved at every pass count, raising passes from $1$ to $5$ cut mean written reasoning from $3{,}703$ tokens to $1{,}661$ ($1{,}731$ at $3$ passes).

At $26.4$ billion parameters one measures that trade-off without seeing every weight or proving an exact floor. This book, *Not Not To Be*, puts the same question to one block small enough to see whole.

## The specimen and its language {#sec-specimen}

The specimen reads two words, `be` and `not`, and reports after every word whether the sentence so far affirms or denies: `not not be` affirms, `not not not be` denies. Write $x_t = 0$ for `be` and $x_t = 1$ for `not`. The meaning up to word $t$ is the parity of its `not`s,
$$y_t = x_1 \oplus x_2 \oplus \cdots \oplus x_t,$$
with $0$ affirming and $1$ denying. In signs, $s_t = a_t s_{t-1}$ with $a_t = (-1)^{x_t}$: `be` keeps the sign and `not` flips it. Flipping the first word flips every later answer, so news of the start must reach the end.

The specimen keeps every piece of Book 1's block—learned attention, both LayerNorms, a ReLU MLP, residual additions, embedding, and readout—at $V = 2$, $D = 4$, $H = 1$, $F = 16$, $L = 1$:
$$p = 2VD + V + 2D + 4D^2 + 2DF + 9D + F = 270,$$
shared across every pass.

::: aside
Looping Book 1's stock block unchanged does not learn this reliably. Four small changes—a two-word window with a fixed $+4$ predecessor bias, fresh input anchored in the first residual, a parameter-free LayerNorm on the stream each pass, and a fixed first-word marker in place of sinusoidal positions, with only the first word seeding the stream—form the working recipe. Each answers a measured failure we show later; the screen supports the combined recipe, not each change alone.
:::

## Turning steps into distance {#sec-window}

With full causal attention, one pass reads every word of a short sentence at once. Although a fixed-size, one-layer transformer cannot compute parity at all lengths (an asymptotic obstruction proved by Kozachinskiy, Steifer, and Wałęga), on short sentences training finds shortcuts and nothing forces the loop.

One rule removes the shortcut: **each word attends only to itself and its predecessor**, a window of width $w = 2$. Because only attention mixes columns—embeddings, LayerNorms, MLPs, and readouts act pointwise—$L$ causal layers of width $w$ iterated $K$ times reach at most $KL(w-1)$ columns left. Here $L = 1$ and $w = 2$, so pass $k$ at word $t$ depends only on $\{t-k,\dots,t\}$, clipped at word $1$. Negation is a chain of maps $f_t(y) = y \oplus x_t$; the window puts each next map one edge away, converting dependent serial steps into a visible horizontal distance.

Word $1$ reaches word $t$ only when $k \ge t - 1$. The line $t = k + 1$ is a light cone. Outside it ($t > k + 1$), flipping $x_1$ leaves everything visible at $(t,k)$ identical while reversing $y_t$, so on fair independent words the best prediction is $(1/2, 1/2)$, costing $\ln 2$ nats. Inside it, the available words determine the answer. Counting the $\max(T-k-1, 0)$ undecided words per pass gives the exact floor
$$\ell^* = \frac{\ln 2}{TK} \sum_{k=1}^{K} \max(T - k - 1, 0), \qquad \ell^* = \frac{21\ln 2}{64} = 0.227438919\ \text{nats at } T = K = 8.$$
Like Book 1's entropy floor, $\ell^*$ is the population loss approached as the model assigns probability tending to one to the right answer wherever news has arrived, and one half wherever it has not.

::: {.figure #fig-cone}
The light cone filling in on a $64$-word sentence over $128$ passes. Position $t = 1,\dots,64$ runs left to right, pass $k = 1,\dots,128$ top to bottom. Cell $(t,k)$ holds a value-ink square of area $2\,|p - \tfrac12|$ for the predicted probability $p$ of the true parity: absent at $\tfrac12$, full at certainty, filled when right and open when wrong. At the population optimum the paper outside the labelled line $t = k + 1$ is empty. A trained model can leave ink there.
:::

## What you will see {#sec-what-you-will-see}

With a loss at every pass (deep supervision), lengths uniform on $4$--$12$ and $K$ uniform on $\{T-1,\dots,2T\}$ at each length, the mixture floor is $0.161350952$ nats, against $0.497544536$ nats for one pass with the same window at any width (accuracy ceiling $\tfrac12 + 1/T$: $53.1\%$ at $T=32$, $51.6\%$ at $T=64$).

In the five-seed selection screen ($1{,}250$ AdamW updates, batch $8$, $6.5$--$9.9$ seconds of Julia updates), four seeds reach the floor within $1\%$—the best at $0.161356316$ nats ($+0.0033\%$), the fifth failing at $+20\%$—and all four are $100\%$ right on the test words at lengths $32$ and $64$ with $K=T$ and $K=2T$ (up to $5\tfrac13\times$ the maximum trained length and depth), using the same $270$ parameters. On $30$ fresh unseen seeds with the recipe frozen, $25/30$ ($83\%$, $95\%$ Clopper--Pearson interval $65$--$94\%$) meet the $1\%$ criterion and decode every tested rollout exactly ($27/30$ meet the loss threshold); without the predecessor bias, $20/30$ ($67\%$) meet the loss criterion and $27/30$ extrapolate exactly. In the selected seed, at $T=64$, pass $128$, attention puts weight $0.967$--$1.000$ (mean $0.989$) on the predecessor, and the centred RMS step drops from $0.0053$ into pass $64$ to $9\cdot 10^{-16}$ into pass $128$.

::: aside
Before the $270$-parameter block, we meet a two-parameter cell $s_t \leftarrow \tanh(g(x_t)\,s_{t-1})$ with fixed predecessor routing: $5/5$ seeds reach the floor within $0.07$--$0.18\%$ in $1$--$2$ seconds and provably settle by pass $T$. For the transformer, we measure the step between passes.
:::

The chapters ahead work out both sides of the opening question: setting the loop beside writing intermediate answers down (including where writing is cheaper and the error law for independent flips on a single path), dissecting the $270$-parameter block, and pulling covectors backwards through @fig-cone in the second ink.

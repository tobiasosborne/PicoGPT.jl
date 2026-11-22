# Writing it down, or looping inside {#ch-written-vs-looped}

::: claim
To find the final parity of $T\ge 2$ independent fair words with a window-two block, writing $T-2$ intermediate answers and making $T-2$ extra passes take the same $T-1$ serial calls. Writing evaluates fewer columns for $T>2$, while looping uses zero scratch tokens. Under independent flips of a relayed bit, writing and a predecessor-only loop share the same Ising-chain error law.
:::

::: names
| maths | physics | ML |
|---|---|---|
| emitting $\hat y_j$ into the next input | externalising an intermediate state | scratchpad, chain of thought |
| gold $y_{j-1}$ vs predicted $\hat y_{j-1}$ | open-loop vs closed-loop driving | teacher forcing vs free decoding |
| binary symmetric channel, rate $\epsilon_i$ | zero-field open Ising chain, $\tanh(\beta J_i)$ | compounding error |
| dependency-path length | causal depth | serial steps, dependent calls |
| columns processed across calls | --- | evaluated token columns |
:::

## One task, three protocols {#sec-three-ways}

Keep the language of @ch-prologue—$x_t = 0$ for `be`, $x_t = 1$ for `not`, parity $y_t = x_1 \oplus \cdots \oplus x_t$ ($0$ affirms, $1$ denies)—on $T \ge 2$ independent fair words, scoring **only the final parity** $y_T$. The block reads $y_T$ from the logits at the last word using the same two token IDs, with no extra query token; the final answer costs one emitted token in every schedule. Prefix supervision during training is gradient signal, not a scratchpad: a label becomes a scratchpad only when its predicted token re-enters a later input.

The same window-two block runs this task three ways:

1. **One pass, nothing written.** Run once over $(x_1,\dots,x_T)$ and read column $T$. Pointwise layers cannot mix columns, so window-two attention exposes at most the last two words and boundary information. For $T > 2$, flipping $x_1$ pairs sentences with identical observations and opposite labels, so the population cross-entropy is minimized at forecast $(1/2, 1/2)$ (equal logits), where it is $\ln 2$.
2. **One pass per decision, written feedback.** Start with $(x_1, x_2)$ and emit $\hat y_2$; for $j = 3,\dots,T$, feed the fresh pair $(\hat y_{j-1}, x_j)$ (ordinary first-column start) through one pass and emit $\hat y_j$. On `not not be not`: `not not` $\to$ `be`, `be be` $\to$ `be`, `be not` $\to$ `not` (first two outputs are scratch). Each call sees the prior parity and next word, giving cross-entropy infimum $0$ under growing margins. Teacher forcing feeds gold $y_{j-1}$; free decoding feeds argmax $\hat y_{j-1}$. A block that gets all four two-word inputs right decodes exactly by induction.
3. **$K$ tied passes, nothing written.** Iterate $K \ge 1$ times over $(x_1,\dots,x_T)$ and read column $T$ on pass $K$. Each pass shifts news at most one column right, so $x_1$ can reach column $K+1$ on pass $K$. For $T > K+1$, pairing words by flipping $x_1$ gives a minimum population cross-entropy of $\ln 2$; at $K \ge T - 1$, the infimum is $0$.

| Schedule | Final observation | Population end-only CE infimum | Calls | Columns | Scratch |
|---|---|---:|---:|---:|---:|
| One pass, blank | At most the last $2$ words and boundary | $[T > 2]\ln 2$ | $1$ | $T$ | $0$ |
| Written feedback | Prior parity ($x_1$ at $T=2$) and $x_T$ | $0$ | $T - 1$ | $2(T - 1)$ | $T - 2$ |
| $K\ge 1$ tied passes | At most the last $K+1$ words and boundary | $[T > K+1]\ln 2$ | $K$ | $KT$ | $0$ |

At $T = 1$, one call and one output suffice; at $T = 2$, the minimum-depth schedules coincide. Scoring every prefix instead restores $\max(T-k-1,0)\ln(2)/T$ per pass, whose mixture averages gave the $0.497545$ and $0.161351$ nats of @ch-prologue—a different objective.

::: aside
This written protocol is streaming: the caller supplies each $x_j$ between answers. Appending a scratchpad *after* an already supplied prompt under a two-token mask gives the decoder different information.
:::

The losses in all three rows can be approached by the $270$-parameter block ($D=4, H=1, F=16, V=2$, first-word start, anchored first residual, fixed stream LayerNorm, $+4$ predecessor bias). A hand-set witness, checked exhaustively on lengths $1$--$8$, achieves smallest signed margin $7.03532$ inside the cone and largest absolute gap $0.099995$ outside at $M=100$ ($70.70679$ and $0.00999995$ at $M=10{,}000$), decoding every written answer rightly. For *trained* weights on these end-only schedules, {{TODO: measured end-only CE and free-decoding accuracy of trained 270-parameter seeds}}.

## The exchange rate, and when writing wins {#sec-exchange}

Compare the two exact schedules ($K = T - 1$ for the loop). The first call folds two bits and each later call consumes one more along the dependency path, so both make $T - 1$ dependent calls. To reach zero infimum, the block emits $T - 2$ intermediate answers or runs $T - 2$ extra whole-sentence passes: **in this protocol, one written answer trades one-for-one for one extra pass**.

At $T = 8$, both make $7$ calls: writing evaluates $14$ columns and emits $6$ scratch tokens; the dense loop evaluates $56$ columns and emits no scratch tokens. Each emits the final answer.

Say honestly when writing wins. Under ordinary dense forwards, writing evaluates $2(T - 1)$ columns against $T(T - 1)$—linear versus quadratic—so **writing is strictly cheaper in evaluated columns for every $T > 2$**. And when every prefix answer is already wanted as output, writing adds no extra scratch-token charge.

Two qualifications pull back. Column counts reflect dense execution and exclude embedding, decoding, padding, and I/O; a scheduled register could trace only a predecessor-only diagonal to $y_T$ in linear work. Also, a dense pass runs $T$ columns in parallel while token emission synchronizes between steps, so column counts alone do not predict latency. Looping adds only the final answer to the transcript.

::: {.figure #fig-three-ways}
The three protocols on `not not be not` ($T=4$) and their scaling for $T = 2,\dots,12$. Left: each call as a row of value-ink columns, arrows tracing $x_1$ to $y_4$—stopped after one step in one pass, relayed across $3$ fresh pairs via $2$ written tokens, or stepped across $3$ tied passes over $4$ columns. Right: evaluated columns ($T$, $2(T-1)$, $T(T-1)$) and scratch tokens ($0$, $T-2$, $0$), labelling $T=8$ ($14$ vs $56$; $6$ vs $0$).
:::

The $1:1$ rate is specific to this protocol, not a universal law. In Inkling 2 Mini's variable-loop variant, extra passes shortened written chains on GPQA Diamond ($3{,}703 \to 1{,}731 \to 1{,}661$ tokens at $1, 3, 5$ passes on questions solved at every pass count)—an empirical trade-off on one subset.

## When an intermediate step is wrong {#sec-errors}

Written feedback and an ideal predecessor-only loop both relay one bit. Number noisy transitions $i = 1,\dots,t$ from an exact seed, write $y_i$ here for the correct state after transition $i$, and let $e_i \in \{0,1\}$ mark an error relative to the *correct update of the received state*. Then
$$\hat y_i \oplus y_i = (\hat y_{i-1} \oplus y_{i-1}) \oplus e_i.$$
Two flips cancel. For independent Bernoulli errors with rates $\epsilon_i$, the error sign $(-1)^{\hat y_t \oplus y_t} = \prod_{i=1}^{t}(-1)^{e_i}$ has expectation $\prod_{i=1}^{t}(1 - 2\epsilon_i) = 1 - 2\Pr(\hat y_t \ne y_t)$, giving
$$\Pr(\hat y_t \ne y_t) = \tfrac12\left(1 - \prod_{i=1}^{t}(1 - 2\epsilon_i)\right),$$
or $\tfrac12\bigl(1 - (1 - 2\epsilon)^t\bigr)$ at equal rate $\epsilon$ (verified by exact rational enumeration through $t = 8$ at $\epsilon \in \{0, 0.1, 0.5, 0.8, 1\}$). Written feedback has $t = T - 1$ noisy decisions (including the final answer) if $x_1$ is exact, or $t = T$ if emitting $y_1$ is also noisy.

Strip off each edge's copy-or-flip sign: what remains is a binary symmetric channel (probabilities $1 - \epsilon_i, \epsilon_i$; sign eigenvalue $1 - 2\epsilon_i$). For fixed words and a fixed starting spin, the physical edge correlation in the zero-field open Ising chain is $\tanh(\beta J_i)=a_i(1-2\epsilon_i)$, where $a_i$ is the intended copy-or-flip sign on transition $i$. That sign is preferred below a $50\%$ flip rate, neither sign at $50\%$, and the opposite sign above it. Rates zero and one are deterministic limits. The endpoint correlation is the product of the edge correlations. For fixed $0<\epsilon<1$, the magnitude $|1-2\epsilon|^t$ tends to zero, driving accuracy to $\tfrac12$. At $\epsilon=\tfrac12$ one noisy step suffices. Under the same noise model, an ideal predecessor-only loop follows one backward diagonal of $T - 1$ updates from $x_1$ for its first arriving answer (later passes can use a newly computed boundary sign), giving the identical formula with $t = T - 1$.

::: aside
A teacher-forced error rate establishes neither independence nor the error law on mispredicted prefixes. The trained two-gain cell updates $\tanh(g(x)s)$ deterministically (exact sign under audited gain conditions), and a soft-attention block mixes paths and shares errors across passes; applying channel rates requires measurement.
:::

Writing and dense looping have the same serial depth, trading transcript tokens against dense columns; unlike a discrete token, an internal message can be gated smoothly from zero ($Y_k = S + g(X_{k-1} - S)$) so a closed loop equals one pass identically.

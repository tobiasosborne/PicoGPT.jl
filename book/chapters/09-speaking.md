# Speaking {#ch-speaking}

::: claim
To write, a language model runs on its own output: it draws the next character from its Gibbs distribution, appends it, and repeats; temperature trades fidelity for variety, and a model trained on Shakespeare shows exactly what next-character statistics can buy, and what they cannot.
:::

::: names
| maths | physics | ML |
|---|---|---|
| $a=\min\{a:\ p_1+\dots+p_a>u\}$, $u\sim U[0,1)$ | drawing a state from the Gibbs distribution | sampling |
| $\arg\max_a z_a$ | ground state, $\tau\to0$ | greedy decoding |
| $\softmax(z/\tau)$, $\tau\ne1$ | Gibbs distribution at another temperature | temperature sampling |
| $x_{1:t}\mapsto x_{t+1}$, repeated | a Markov chain on contexts | autoregressive generation |
| the last $T$ characters | a finite memory | context window |
| $x_{1:k}$, fixed | initial condition | prompt |
| $e^{\mathcal L}$ | effective number of states | perplexity |
:::

## One character at a time {#sec-one-at-a-time}

The chain rule of probability (@ch-task) factors the probability of a text into next-character probabilities,
$$P(x_1,\dots,x_m)=P(x_1)\prod_{t=1}^{m-1}P(x_{t+1}\mid x_1,\dots,x_t).$$
Read from left to right, it is also a recipe for drawing a text. Fix the first characters, the *prompt*. Then draw $x_{t+1}$ from $q(\,\cdot\mid x_{1:t})$, append it, and repeat. Any family of normalised conditional distributions defines a normalised distribution on texts in this way, and drawing one factor at a time draws a text from it. The model supplies the conditionals: run it on the text so far, take the logits $z$ at the last position, and turn them into $p=\softmax(z/\tau)$.

The model sees at most $T$ characters. When the text grows longer, only the last $T$ are fed in, numbered $1,\dots,T$ again. That is a restriction on the model, a finite memory, not a consequence of the chain rule. The process is then a Markov chain whose state is the last $T$ characters.

One draw needs one uniform random number $u\in[0,1)$. Stack the probabilities $p_1,\dots,p_V$ end to end along $[0,1]$, in vocabulary order. The draw is the character whose segment contains $u$, that is, the first $a$ with $p_1+\dots+p_a>u$. Since $u$ lands in a segment with probability equal to its length, character $a$ is drawn with probability $p_a$. This is inversion of the cumulative distribution function, and it is all the code does:

::: {.code file="src/sample.jl" fn="draw"}
:::

::: {.code file="src/sample.jl" fn="generate"}
:::

## Temperature in practice {#sec-temperature-practice}

The model was trained at $\tau=1$, so $p=\softmax(z)$ is the distribution that training fitted to the data, and $\tau=1$ is the only temperature at which samples are drawn from the model itself. Every other temperature is a deliberate distortion, in one of two directions (@ch-energies).

As $\tau\to0$ the Gibbs distribution freezes onto its ground state, and sampling becomes *greedy decoding*: always take the likeliest character. Greedy text is the model's most confident guess at each step, but not the most probable text: the likeliest first character may lead into contexts where every continuation is unlikely (@ex-greedy). Greedy text also tends to fall into loops, because a deterministic map from contexts to characters must eventually revisit a context, and from then on it repeats.

::: aside
The code takes $\tau=0$ to mean `argmax(z)`, which returns the *first* largest logit. The true limit of @ch-energies shares the probability equally among tied ground states. With Float64 logits exact ties almost never happen, so the shortcut costs nothing in practice, but it is a convention, not the limit.
:::

As $\tau$ grows, the distribution melts towards uniform and the text becomes more varied and less like the data. In between there is a trade: a little below $1$, samples stay closer to the typical patterns of the training text and make fewer errors; above $1$ they wander.

## The specimen speaks {#sec-specimen-speaks}

@fig-sampling-tree lets the specimen write from a one-character prompt, a space. Each column is one draw: the interval $[0,1]$ cut into the seven probabilities, the random number $u$ as a tick, and the character whose segment it hits. The thin line through the chosen segments is the path the text took; the outlined segments are the paths not taken.

::: {.figure #fig-sampling-tree prompt="␣" count=22}
The trained specimen writes twenty-two characters after the prompt "␣". Each column is the distribution of the next character, stacked along $[0,1]$ in vocabulary order; the tick is the uniform random number $u$, and the segment it falls in (solid) is the character written. [Drag the temperature, draw new random numbers, or play the writing from the start.]{.web} Under the text: where the specimen's contexts leave the set it was trained on.
:::

At $\tau=1$ the first column is cut into three nearly equal parts: after a space at the start of a window, the specimen cannot tell "␣be", "␣or" and "␣not" apart, which is the threefold ground state of @ch-energies and the floor of @ch-surprise seen from the writer's side. After that first choice almost every column is a single solid segment. The specimen has memorised its corpus: once a branch is taken, the rest of the line follows. Different random numbers $u$ choose different first characters, and each choice is followed by the rest of the corpus from that point on.

Then the text runs past the last "be" of the corpus. From there on, every context the specimen sees is one it never met in training; the bracket under the text marks the place. The columns stay just as solid. The specimen is exactly as sure of what it never saw as of what it learned, because nothing in training ever asked it to be unsure: softmax always returns a distribution, and away from the data that distribution is whatever the parameters happen to give. Confidence is not knowledge.

At temperature $2$ the outlined segments widen. Branches that were negligible become possible, the text leaves the corpus earlier, and it decays into strings like those a careless copyist might produce. At temperature zero every column is a single segment: greedy decoding, the same text whatever $u$.

## A real model {#sec-real-model}

The specimen can only recite. To see what next-character prediction can do, the same code trains a larger model on *Tiny Shakespeare*: about a million characters from Shakespeare's plays, $V=65$ distinct characters (letters in both cases, punctuation, space, newline, and one digit, 3). The model has $D=128$, $H=4$ heads, $L=4$ blocks and a context of $T=256$ characters, and its 810,049 parameters are trained in Julia in Float32, with AdamW, a warmup-and-cosine schedule and clipping (@ch-learning). Each step reads 32 windows of 256 characters. The last tenth of the text is held out: the model never trains on it, so its loss there measures prediction rather than memory.

::: {.figure #fig-shakespeare-checkpoints}
The Tiny Shakespeare model in training. Left: its loss, the mean surprise per character, on its training text (solid) and on held-out text it never saw (open circles), with $\log 65$, uniform guessing, for reference; the step axis is logarithmic. Right: what the model wrote at $\tau=1$ at each saved checkpoint, starting from a newline, with the share of its words that occur in the training text.
:::

The loss starts at $\log 65$, where it must: the untrained model's logits are nearly equal, and its first sample is every character of the vocabulary in random order, capitals, punctuation and the one digit alike. Within a hundred steps it has learned which characters are common: spaces, newlines, lower-case vowels. By a few hundred steps the strings have the shape of words and lines. By a thousand the text has the layout of a play, with speaker names in capitals, a colon and a line break, and lines of roughly the length of verse. By the end, most words are real words.

A loss in nats is easier to feel through its exponential. If a model's loss is $\mathcal L$, then $e^{\mathcal L}$, the *perplexity*, is the number of equally likely characters that would cost the same surprise: $65$ at the start, and, at the end, a handful (read it off the figure). Two curves diverge as training goes on. The loss on the training text keeps falling; the loss on held-out text falls more slowly and then flattens. The gap between them is what the model has memorised about its particular million characters rather than learned about Shakespeare's.

::: {.figure #fig-shakespeare-temperature}
The trained model writing from the same prompt at four temperatures. Words that never occur in the training text are underlined with dots; the bar above each column is the share of words that do.
:::

@fig-shakespeare-temperature shows the dial at work. At $\tau=\frac12$ every word is real, and the text is safe and repetitive, leaning on the commonest words. At $\tau=1$ a few invented words appear. At $\tau=1.5$ a third of the words are invented, and spelling decays.

Here, honestly, is what the model has learned. It spells common words. It knows the format of a play: who speaks is set in capitals and followed by a colon and a line break, and speeches are broken into lines of verse length. It knows where capitals and punctuation go. It strings two or three words together with a plausible rhythm. And here is what it has not learned. It does not know what any word means. Its sentences have no syntax beyond a few words. The speakers do not speak to each other, nothing happens, and names are invented as freely as words. Every one of these is a statement about next-character statistics: what can be predicted from the last few characters, it predicts; what needs more than that, it does not.

## Inside the real model {#sec-inside}

What do its sixteen heads read? @fig-shakespeare-attention runs the trained model on a passage of the held-out text in which one line is spoken twice. Above the passage are arcs for one reader in one head, as in @ch-attention; beneath it, the surprise of each character. Below are all sixteen attention patterns, drawn as lower triangles, as in @ch-attention, with three habits measured for each head: how much weight it puts on the reader itself, on the character before it, and on the character a copying model would read.

::: {.figure #fig-shakespeare-attention}
The trained model reading a passage of held-out text that contains a repeated line. Top: the passage, with each character's surprise beneath it; arcs show what the chosen reader reads in the chosen head. Below: every head of every layer, rows readers and columns sources, dot area the weight. Under each head, its mean weight on the reader itself (self), on the previous character (prev), and, over the repeated line, on the character that followed the same character the first time (copy). [Point at a character, or click a head.]{.web}
:::

The first layer spreads its attention widely. In the later layers many heads put a large share of their weight, in several close to half, on the reader itself and on the character before it; their dots crowd the diagonal and the line just below it. That is how a model whose maps act column by column (@ch-block) gets to see pairs and triples of characters at all: a head that reads the previous character copies it into the current column, and the next layer can then react to the pair. One head of the third layer looks different: its dots stand in vertical stripes, because whatever the reader, it puts much of its weight on line breaks and on the capitals that begin lines. A model that writes speeches in lines has use for knowing where the current line began.

The repeated line is a test. A model that noticed the repetition could predict the second "Good morrow, neighbour" almost perfectly: read where the same characters came before, and copy what followed them. The *copy* number measures exactly that, head by head. It is at most a hundredth for every head of this model, and the surprise per character of the repeated line is the same the second time as the first. Within the 5000 steps of its training, this model has not learned to copy from its own context.

::: aside
Heads that do this, called *induction heads*, are well documented in larger transformers trained for longer, where they appear fairly suddenly during training ([Olsson et al., *In-context Learning and Induction Heads*, 2022](https://arxiv.org/abs/2209.11895)). The mechanism needs two heads in sequence: one that reads the previous character, and one that looks for an earlier position whose previous character matches the current one. This model has several heads of the first kind and none of the second.
:::

The two models of this book sit at the ends of the same road. The specimen has learned its eighteen characters perfectly and knows nothing else. The Shakespeare model has learned the statistics of a million characters well enough to imitate their surface, and nothing of what they say. Everything in between, the larger contexts, the more layers, the longer training, the more data, is the subject of the Epilogue.

## Exercises {.unnumbered}

::: exercise
Let $u$ be uniform on $[0,1)$ and $F_a=p_1+\dots+p_a$. Show that the first $a$ with $F_a>u$ is $a$ with probability $p_a$. What changes if the characters are stacked in a different order?
:::

::: {.exercise #ex-greedy}
Construct a model on two characters $\{x,y\}$ and texts of length two in which greedy decoding produces a text that is not the most probable text of length two.
:::

::: exercise
In @fig-sampling-tree at $\tau=1$, how many different texts can the specimen write from the prompt "␣" before it leaves its corpus? [Use "draw again" to check. Then raise $\tau$ to $2$ and count the columns that are no longer a single segment.]{.web}
:::

::: exercise
Show that $e^{H(p)}=V$ for the uniform distribution on $V$ outcomes and $e^{H(p)}=1$ for a one-hot $p$. Read the final held-out loss of the Shakespeare model off @fig-shakespeare-checkpoints and convert it to a perplexity. Between how many characters is the model, on average, still choosing?
:::

::: exercise
Why is the loss on held-out text higher than on the training text, and why does the gap grow? What would you expect if training went on for ten times as long with the same data?
:::

::: exercise
In @fig-shakespeare-attention, pick a head with a large *prev*. [Point at characters in the middle of a word and at the first character after a space.]{.web} What does each read? Why might the model want the previous character to be available in every column?
:::

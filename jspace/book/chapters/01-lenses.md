# Reading the stream early {#ch-lenses}

::: claim
The model's readout is a map defined on the whole of $\R^D$, so it can be applied to the stream at any depth: that is the logit lens. It is exact at the last depth and unreliable before it, because nothing makes the stream use the same coordinates at every depth; the only thing that turns an early column into a final one is the rest of the model, and the rest of the model is not the identity.
:::

::: names
| maths | physics | ML |
|---|---|---|
| $X_\ell\in\R^{D\times N}$, $\ell=0,\tfrac12,1,\dots,L$ | the state at time $\ell$ | hidden states |
| $R(x)=\softmax\bigl(W_{\text{out}}\LN(x)+b_{\text{out}}\bigr)$ | a measurement with $V$ outcomes | LM head |
| $R$ applied to a column of $X_\ell$, $\ell<L$ | measuring early | logit lens |
| $F_\ell:X_\ell\mapsto X_L$ | the propagator from $\ell$ to $L$ | the rest of the model |
| the fraction of columns with $\arg\max R(x_\ell)=\arg\max R(x_L)$ | — | top-1 agreement |
:::

## The stream at every depth {#sec-depths}

Recall the shape of the model from [PicoGPT's chapter on the block](https://tobiasosborne.github.io/PicoGPT.jl/block/). The embedding writes the first state, and each block adds two updates to it, one from attention and one from the MLP. Counting in half blocks,
$$X_0=E[:,x]+P,\qquad X_{\ell-\frac12}=X_{\ell-1}+\operatorname{Attn}_\ell\bigl(\LN(X_{\ell-1})\bigr),\qquad X_\ell=X_{\ell-\frac12}+\operatorname{MLP}_\ell\bigl(\LN(X_{\ell-\frac12})\bigr),$$
for $\ell=1,\dots,L$. Column by column, $x_\ell^{(t)}=X_\ell[:,t]$ is the state of the token at position $t$ after $2\ell$ updates. The specimen has three states per token, $x_0$, $x_{1/2}$ and $x_1$; the Shakespeare model has nine.

The point of the residual form is that every update is *added*. Nothing is ever replaced, so every state at every depth lives in the same space $\R^D$, and $X_L$ is $X_\ell$ plus the updates written after depth $\ell$. @fig-j-depths draws the states of one window of the specimen, depth by depth, on one scale.

::: {.figure #fig-j-depths window=1 column=4}
The stream of one window of the specimen at its three depths. Each glyph is one column $x_\ell^{(t)}\in\R^8$, one spoke per entry, all on one scale, laid out as $X_\ell$ is written: depths down the page, positions across. Under the last row, the readout of each final column, the model's distribution over the next character; the character that actually comes next is underlined. [Click a column for its lengths, and drag the window along the line.]{.web}
:::

On average the columns grow as they go down, most of all in the MLP update, and they turn. Nothing about the residual form stops either. What the form does guarantee is that $x_0$, $x_{1/2}$ and $x_1$ are vectors of the same space, so that any map defined on $\R^D$ can be applied to all of them. The model has one such map that we know how to read.

## The readout as a map {#sec-readout}

The readout of the last column is a map from $\R^D$ to the probability simplex $\Delta_V$ of distributions on the $V$ characters,
$$R:\R^D\to\Delta_V,\qquad R(x)=\softmax\bigl(W_{\text{out}}\LN(x)+b_{\text{out}}\bigr),\qquad z_a=\sum_{i=1}^DW_{\text{out}}[a,i]\,\LN(x)_i+b_{\text{out}}[a].$$
It is the final LayerNorm, with its own gain and offset, followed by an affine map and a softmax. Three properties matter here.

It is defined on the whole of $\R^D$. Nothing in its formula knows which depth a column came from.

It reads a direction, not a vector. LayerNorm removes the component along $\one$ and divides by the length of what is left, so with $\varepsilon=0$
$$R(x+c\one)=R(x)\quad\text{for every }c,\qquad R(cx)=R(x)\quad\text{for every }c>0.$$
With $\varepsilon=10^{-5}$ the second identity holds only approximately, with a correction of order $\varepsilon/\sigma^2$, where $\sigma$ is the root mean square of the centred column. For the columns of the stream that is negligible, as long as nothing shrinks them; @ch-jlens meets a case where something does.

After the normalisation it is affine. The scores $z_a$ are linear in the normalised column: the readout pairs $\LN(x)$ with the $V$ rows of $W_{\text{out}}$, one covector per character, and adds a bias. In the paper's vocabulary $W_{\text{out}}$ is the *unembedding* $W_U$, the name recording that it maps the stream back to the vocabulary, the way the embedding $E$ mapped the vocabulary into the stream.

::: aside
In PicoGPT's code the readout is two lines of `gpt` in `src/model.jl`: the final `layernorm` with $\gamma_f,\beta_f$, then the `affine` map with $W_{\text{out}},b_{\text{out}}$. The softmax is left to the loss.
:::

## The logit lens {#sec-logit-lens}

Since $R$ is defined on all of $\R^D$, apply it early: read the column at depth $\ell<L$ as if it were the last one,
$$\operatorname{lens}^{\text{logit}}_\ell(x)=R(x),\qquad x=X_\ell[:,t].$$
This is the *logit lens*. It amounts to pretending that the rest of the model, everything between depth $\ell$ and the readout, is the identity. At the last depth the pretence is true, and the logit lens is the model.

@fig-j-logit-lens applies it to the specimen. At depth $0$ the column is $Ee_{x_t}+P_{:,t}$: the current character and its position, nothing else, so the logit lens at depth $0$ is a bigram model with a clock, which reads the next character off the current one and its position. At depth $\tfrac12$ attention has added what the earlier characters hand over. Over all {{value:lensSpecimenColumns}} columns of the ten windows, the logit lens names the model's own top character at {{value:lensLogitAgreeZero}} of them at depth $0$ and at {{value:lensLogitAgreeHalf}} at depth $\tfrac12$.

::: {.figure #fig-j-logit-lens window=1 column=7 depth=0.5}
The logit lens on one window of the specimen. Each cell holds the readout $R$ applied to one column at one depth: the probabilities of the seven characters, on one scale. The most probable character stands over its bar, in ink where it is also the model's top character at that position and in grey where it is not; the character that actually comes next is underlined. On the right, the number of positions where the lens agrees with the model. At depth $1$ the lens is the model's own readout. [Point at a cell for its numbers, and drag the window.]{.web}
:::

One disagreement settles the question of where the information is. Take the column of the *o* after *to be* (position 7 of the first window). The model says *{{value:lensExampleModel}}*. At depth $\tfrac12$, after attention has read the earlier characters, the logit lens says *{{value:lensExampleLogit}}*. But in the specimen everything after depth $\tfrac12$, the MLP update and the readout, acts on one column at a time: the model's *{{value:lensExampleModel}}* is a function of this one column $x_{1/2}$ and of nothing else. The information is in the column. The logit lens cannot read it, because it reads the column in the wrong coordinates.

::: aside
In the paper: "Comparison to Related Techniques" introduces the logit lens as the special case $J_\ell=I$ of the Jacobian lens, "reasonable in late layers due to the influence of residual connections", degrading in earlier ones.
:::

## Why it fails early {#sec-why}

The logit lens is right exactly when the rest of the model does not change the readout. Write the rest of the model as a map of the whole window,
$$F_\ell:\R^{D\times N}\to\R^{D\times N},\qquad X_\ell\longmapsto X_L,$$
so that the model's prediction at position $t$ is $R\bigl(F_\ell(X_\ell)[:,t]\bigr)$ and the logit lens's is $R\bigl(X_\ell[:,t]\bigr)$. In the residual form,
$$F_\ell(X_\ell)=X_\ell+\sum_{\text{updates after }\ell}(\text{update}),$$
and the logit lens is the approximation that drops the sum. Late in the model the sum is short and the approximation is good. Early it is a sum of up to $2L$ updates, and nothing makes them small.

There is a more structural reason, and it is the one the paper fixes. A column $x_\ell$ and a column $x_L$ live in the same space, but nothing in training asks the model to use the same *coordinates* in both. The readout pairs the final column with the rows of $W_{\text{out}}$; the loss rewards whatever arrangement of the stream at depth $\ell$ lets the remaining blocks produce a good final column, and it does not care whether that arrangement happens to be legible to $W_{\text{out}}$ already. A direction at depth $\ell$ means what the rest of the model does with it. The logit lens reads every depth in the final depth's coordinates; the only thing that actually translates depth-$\ell$ coordinates into final ones is $F_\ell$, which is not linear and, through attention, mixes positions.

So the repair suggests itself. Replace the identity by the best linear stand-in for $F_\ell$ that we can justify, apply it, and then read out. @ch-jacobian builds that stand-in from the derivative of $F_\ell$, averaged over contexts.

::: aside
The logit lens was introduced in 2020 by the pseudonymous blogger nostalgebraist, in a post called *interpreting GPT: the logit lens*. Belrose and colleagues trained one affine map per layer to make the readout match the model's output, the [*tuned lens*](https://arxiv.org/abs/2303.08112) (2023). The paper argues that this fits correlation rather than cause, and finds that the tuned lens "skips ahead" to the output.
:::

## In the Shakespeare model {#sec-logit-shakespeare}

The specimen has only three depths. The Shakespeare model has nine, and @fig-j-logit-lens-depth runs the logit lens at each of them over {{value:lensShPositions}} positions of held-out text: {{value:lensShPerWindow}} evenly spaced positions in each of {{value:lensShWindows}} windows of the validation set. Two questions are asked of it at every depth: does its top character agree with the model's own, and is it the character that actually comes next?

::: {.figure #fig-j-logit-lens-depth}
The logit lens of the Shakespeare model, depth by depth, on held-out text. Filled dots: the fraction of positions at which the lens's top character is the model's own top character. Open rings: the fraction at which it is the character that actually comes next. The dashed lines mark uniform guessing, $1/V$, and the model's own accuracy, which the rings reach at the last depth, where the lens is the model. [Point at a depth for its numbers.]{.web}
:::

At depth $0$ the logit lens agrees with the model at {{value:lensShLogitAgreeZero}}% of the positions and names the next character at {{value:lensShLogitAccZero}}%: far better than uniform guessing, because the current character alone is a good predictor of the next, and far worse than the model, which is right at {{value:lensShAccModel}}%. Through the first two blocks the agreement hardly moves. It rises in the third, and reaches only {{value:lensShLogitAgreeLast}}% at depth $3\tfrac12$, with one update left to go: the last MLP update changes the model's top character at more than half of the positions. If the logit lens were the whole story, this model would settle its prediction in its last half block.

Is that a fact about the model, or about the lens? The rest of this book is an attempt to answer that question with a better lens.

## In the code {#sec-code-lenses}

The analysis keeps every state of the stream, and the pullback of every half block, from one forward pass:

::: {.code file="jspace/src/JLens.jl" fn="residual_trace"}
:::

and reads them with one function. With no matrix it is the logit lens; @ch-jlens gives it a matrix.

::: {.code file="jspace/src/JLens.jl" fn="lens_logits"}
:::

## Exercises {.unnumbered}

::: exercise
Show that $R(x+c\one)=R(x)$ for every $c$, and that $R(cx)=R(x)$ for every $c>0$ when $\varepsilon=0$. Deduce that, with $\varepsilon=0$, the readout depends on $x$ only through a point of a sphere of dimension $D-2$. What happens at $x=c\one$?
:::

::: exercise
At depth $0$, $x=Ee_a+P_{:,t}$. Show that the logit lens at depth $0$ is a function of the pair $(a,t)$ alone, so that in the specimen it can produce at most $VT$ different distributions. How many distinct contexts of length at most $T$ does the corpus contain, and how many distinct distributions does the model produce on them?
:::

::: exercise
Top-1 agreement is a coarse measure. Give two distributions on seven characters with the same most probable character and total variation distance close to $1$, and two with different most probable characters and total variation distance close to $0$. Which measure would you rather report, and which does the live readout of @fig-j-logit-lens use on the web?
:::

::: exercise
In @fig-j-logit-lens-depth, between which two consecutive depths does the agreement with the model change most? The accuracy against the next character changes much less over the same step. Explain how the last update can change the top character at most positions while improving accuracy by comparatively little.
:::

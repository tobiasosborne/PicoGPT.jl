# The Jacobian lens {#ch-jlens}

::: claim
The Jacobian lens reads a column of the stream at depth $\ell$ by pushing it to the end of the model with the averaged derivative $J_\ell$ and applying the model's own readout. Its lens vectors are covectors, averaged gradients of one character's score, and at the last depth the lens is the model.
:::

::: names
| maths | physics | ML |
|---|---|---|
| $\operatorname{lens}_\ell(x)=R(J_\ell x)$ | averaged linear response, then the measurement | Jacobian lens (J-lens) |
| $v_a=J_\ell^\T W_{\text{out}}^\T e_a$, row $a$ of $W_{\text{out}}J_\ell$, a covector | a static susceptibility | J-lens vector |
| $\inner{v_a}{x}$, a covector paired with a value | the response it predicts | probe |
| the most probable character of $\operatorname{lens}_\ell(x)$ | the likeliest outcome | the top lens token |
:::

## Replacing the rest of the model {#sec-replace}

@ch-lenses applied the readout
$$R:\R^D\to\Delta_V,\qquad R(x)=\softmax\bigl(W_{\text{out}}\LN(x)+b_{\text{out}}\bigr),$$
to a column $x$ of the stream $X_\ell$ at an early depth, and found it unreliable: between depth $\ell$ and the readout stands the rest of the model, $F_\ell:X_\ell\mapsto X_L$, and $F_\ell$ is not the identity. @ch-jacobian replaced $F_\ell$ by one $D\times D$ matrix, the averaged derivative
$$J_\ell=\frac1{BT}\sum_{b=1}^{B}\sum_{t=1}^{T}\sum_{t'\ge t}K^{(b)}_{t't},\qquad K^{(b)}_{t't}=\frac{\partial X^{(b)}_L[:,t']}{\partial X^{(b)}_\ell[:,t]}.$$
The Jacobian lens puts the two together:
$$\operatorname{lens}_\ell(x)=R(J_\ell x)=\softmax\bigl(W_{\text{out}}\LN(J_\ell x)+b_{\text{out}}\bigr),$$
or, logit by logit,
$$z_a(x)=\sum_{i=1}^{D}W_{\text{out}}[a,i]\,\LN(J_\ell x)_i+b_{\text{out}}[a].$$
It is the model's readout, LayerNorm, gains and biases exactly as the model has them, applied to the column $J_\ell x$ instead of to the model's own final column. With $J_\ell=I$ it is the logit lens. The two lenses differ by one matrix per depth, and this chapter is about what that matrix does.

It is worth being precise about what $J_\ell x$ is, because it is not an approximation to the final column $X_L[:,t]$. The column $x$ is not small, and the rest of the model does not send zero to zero. What $J_\ell x$ is, exactly: the change that $x$ would make to the final stream, at its own position and at every later one, summed, to first order, if it were added to the stream of a text at depth $\ell$, averaged over the positions and the texts. The lens reads the column as a disturbance and asks what it would make the model say, now or later, in an average context. And since LayerNorm does not see the length of its input (up to its $\varepsilon$, to which we return in @sec-last), the lens reads only the direction of $J_\ell x$, not its size.

::: aside
In physics this is linear response taken at face value: $K_{t't}$ is the retarded response of the output at time $t'$ to a kick at time $t$ (@ch-jacobian), $J_\ell$ its sum over the future, averaged over where the kick lands. The lens applies the response to the column itself, as if the column were the kick.
:::

Two things are thrown away. The lens reads one column and ignores the others, although $F_\ell$ mixes them through attention: what the later columns would have done with this one has been averaged into $J_\ell$, and what the earlier columns have done to it is seen only through $x$ itself, not again. And $J_\ell$ is linear while $F_\ell$ is not. Whether what is left is useful is a question for the data, and the rest of the chapter asks it of both models.

::: aside
The *tuned lens* of [Belrose et al. (2023)](https://arxiv.org/abs/2303.08112) also puts a matrix in front of the readout at every depth, but fits it: it is trained to make the lens agree with the model's output. The paper argues that this objective is correlational, and that a tuned lens learns to skip ahead to the answer rather than show what the column carries on the way. $J_\ell$ is not fitted to anything. It is a derivative, averaged.
:::

## Lens vectors are covectors {#sec-lens-vectors}

Before the final LayerNorm the readout is linear, and so is $J_\ell$. Composed, they give one linear function of $x$ per character,
$$(W_{\text{out}}J_\ell\,x)_a=\sum_{q=1}^{D}(W_{\text{out}}J_\ell)_{aq}\,x_q=\inner{v_a}{x},\qquad v_a=J_\ell^\T W_{\text{out}}^\T e_a .$$
The $V$ vectors $v_a\in\R^D$, the rows of $W_{\text{out}}J_\ell$, are the paper's *J-lens vectors*, one per character and per depth.

What kind of object is $v_a$? Write $w_a=W_{\text{out}}^\T e_a$ for row $a$ of the readout, the covector that scores character $a$ on the final column before LayerNorm: $s_a(y)=\inner{w_a}{y}$. Then $v_a=J_\ell^\T w_a$ is $w_a$ pulled back through the averaged derivative, in the sense of [PicoGPT, Pulling back](https://tobiasosborne.github.io/PicoGPT.jl/pulling-back/). Spelled out block by block,
$$v_a=\frac1{BT}\sum_{b,t}\ \sum_{t'\ge t}\bigl(K^{(b)}_{t't}\bigr)^\T w_a,$$
and since $\bigl(K^{(b)}_{t't}\bigr)^\T w_a$ is the gradient of $s_a(X^{(b)}_L[:,t'])$ with respect to $X^{(b)}_\ell[:,t]$,
$$v_a=\frac1{BT}\sum_{b,t}\ \frac{\partial}{\partial X^{(b)}_\ell[:,t]}\ \sum_{t'\ge t}s_a\bigl(X^{(b)}_L[:,t']\bigr).$$
It is the gradient, with respect to the stream at depth $\ell$, of the score of $a$ summed over the present and every later position, averaged over positions and texts. A gradient is a covector, and the book draws $v_a$ as one: in covector ink, with bar tips. The column $x$ is a value, in value ink. The lens score $\inner{v_a}{x}$ is a covector paired with a value: by how much the scores of $a$, now and later, would rise if $x$ were added to the stream.

::: aside
Exactly, the final LayerNorm sits between $W_{\text{out}}$ and $J_\ell x$. It centres the column, divides by one number common to all characters and multiplies by the gains $\gamma$, so the lens ranks characters by pairings with slightly different covectors, shifted by constants. @sec-probe writes this out; the paper, and we, call the rows of $W_{\text{out}}J_\ell$ the lens vectors.
:::

@fig-j-lens-vectors draws all of them for the specimen. At depth $1$, the last, they coincide with the readout rows; @sec-last shows why. At the earlier depths they point elsewhere, which is the whole difference between the two lenses, and most are longer, up to {{value:jlLenRatioMax}} times{{value:jlLenNote}}. The lens itself is blind to the length, through LayerNorm. @ch-writing, which adds lens vectors to the stream, is not.

::: {.figure #fig-j-lens-vectors}
The J-lens vectors $v_a$ of the specimen, one row per depth and one column per character, as spoke glyphs in covector ink; at the bottom, the readout's own rows $w_a$, which the logit lens uses at every depth. All glyphs share one scale. Under each lens vector, its cosine with the readout row of the same character. [Point at a glyph for its length and its largest components.]{.web}
:::

::: aside
A covector needs no inner product to be paired with a value. If the stream at depth $\ell$ is written in other coordinates, $x'=Mx$, the lens vectors become $M^{-\T}v_a$ and every pairing $\inner{v_a}{x}$ stays what it was (an exercise below). The J-space of @ch-jspace will need an inner product; the lens does not.
:::

## At the last depth the lens is the model {#sec-last}

At depth $L$ nothing of the model is left. The rest of the model is the identity, so
$$K_{t't}=\frac{\partial X_L[:,t']}{\partial X_L[:,t]}=\delta_{t't}\,I,\qquad J_L=\frac1{BT}\sum_{b,t}\ \sum_{t'\ge t}\delta_{t't}\,I=I,$$
the lens vectors are the rows of $W_{\text{out}}$, and $\operatorname{lens}_L(x)=R(x)$ is the model's own output, exactly. This is why @ch-jacobian sums over targets and averages over sources: the normalisation that makes $J_L$ the identity makes the lens at the last depth the model.

The paper's Methods write $J_\ell$ as an expectation over pairs $(t,t')$ with $t'\ge t$, each pair counted once. A window has $T(T+1)/2$ such pairs, of which $T$ are diagonal, so its pair-weighted matrix is ours divided by $(T+1)/2$, and at the last depth it is
$$J^{\text{pair}}_L=\frac{2}{T+1}\,I .$$
(The paper's pseudocode, which sums over targets and then averages over positions, computes ours.) In exact arithmetic the scalar is harmless: LayerNorm without $\varepsilon$ does not see the length of its input. With $\varepsilon$ it does. For every $s>0$,
$$\LN_\varepsilon(sy)=\LN_{\varepsilon/s^2}(y),$$
so shrinking the column by $s$ is the same as running LayerNorm with $\varepsilon/s^2$ in place of $\varepsilon$: a larger floor under the scale it divides by, which shrinks the part of the logits that depends on the column, as a raised temperature would, though not the constant $W_{\text{out}}\beta_f+b_{\text{out}}$. For the specimen $s=2/9$, and the pair-weighted lens at the last depth differs from the model by at most {{value:jlLitErrSpecimen}} in any probability. For the Shakespeare model $s=2/257$, the effective floor is $\varepsilon\,(T+1)^2/4\approx{{value:jlEpsEffShakespeare}}$, and the pair-weighted lens differs from the model by up to {{value:jlLitErrShakespeare}} in a probability, although its first choice still agrees with the model's at {{value:jlLitAgreeLast}} of the positions we test. A scalar that cancels on paper does not cancel in a model with an $\varepsilon$ in it. @fig-j-lens-depth carries the pair-weighted lens along as crosses.

::: aside
In the paper, the expectation $\E_{t,\,t'\ge t,\,\text{prompt}}$ is in the Methods section "The Jacobian Lens"; the pseudocode is in the appendix "Methodological Details and Ablations", which also lists the variants we mention in @ch-jacobian. Whether the scalar matters in the paper's models depends on the $\varepsilon$ of their normalisation and on the size of their streams, which we cannot see; in ours it is visible.
:::

## Both lenses on the specimen {#sec-specimen-lenses}

@fig-j-readouts reads one window of the specimen with both lenses at every depth and every position, against the model's own output.

::: {.figure #fig-j-readouts window=1}
Both lenses on one window of the specimen. Each column is a position, headed by its character; each row is a depth, and the last row is the model. In each cell the seven probabilities, in the order ␣ b e n o r t: solid bars for the J-lens, open bars for the logit lens. The tick under a cell marks the character that actually comes next. At depth $1$ the two lenses and the model coincide. [Drag the window along the line; point at a cell for its numbers.]{.web}
:::

At depth $1$ the three rows are one row. At depth $0$ the lenses differ most: over all {{value:jlSpecColumns}} columns of the ten windows they name the same character first at only {{value:jlSpecLensesAgreeZero}} of them, and at {{value:jlSpecLensesAgreeHalf}} at depth $\tfrac12$. Against the model, the J-lens's first choice is the model's own at {{value:jlSpecAgreeZeroJ}} of the {{value:jlSpecColumns}} columns at depth $0$ and at {{value:jlSpecAgreeHalfJ}} at depth $\tfrac12$; the logit lens manages {{value:jlSpecAgreeZeroLogit}} and {{value:jlSpecAgreeHalfLogit}}. In the specimen, replacing the rest of the model by its averaged derivative instead of by the identity moves the early readouts towards the model's.

One confound belongs here, because this is the one place where the J-lens wins. $J_\ell$ is averaged over the same {{value:jlSpecColumns}} columns the lenses are then scored on, and for the Shakespeare model over the same $64$ held-out windows that supply the scored positions and the passage of @sec-along-text; the paper averages over its own corpus of prompts and evaluates on others. The overlap can only flatter the J-lens, so the specimen's lead is an upper bound and the Shakespeare model's deficit is, if anything, understated. A leave-one-window-out average, which the export's per-window Jacobians make possible at depth $\tfrac12$, is an exercise.

## Along a real text {#sec-along-text}

The specimen has three depths; the Shakespeare model has nine. @fig-j-grid reads it along an exchange of greetings from the held-out text, one character per cell, the weight of the type standing for the probability.

::: {.figure #fig-j-grid}
The J-lens of the Shakespeare model along a passage of held-out text. Each column is a position of the text, set across the top ($¶$ is a line break, ␣ a space). Under it, one cell per depth, top to bottom, holding the character the lens ranks first; the weight of the type is proportional to its probability. At depth $4$ the J-lens is the model. The bottom row is the character that actually comes next. In print, the logit lens follows as a second plate. [Point at a cell for the lens's first three choices; switch to the logit lens to compare.]{.web}
:::

Read the J-lens plate from the top. Through depth $1\tfrac12$ the lens reads characters that have little to do with the text. From depth $2$ on there is structure, and the clearest is under the two speakers' names. Count, at each depth, under how many of the {{value:jlNameLetters}} capital letters of *GREMIO* and *BAPTISTA* a lens's first choice is a colon: {{value:jlColonTable}}. At depth {{value:jlColonDepth}}, where the effect is largest, the J-lens reads a colon under {{value:jlColonJ}} of the letters and the logit lens under {{value:jlColonLogit}}; at the first four depths the logit lens reads more colons than the J-lens, one or two of them the genuine next character after a name's last letter. The colon is the next character only after the last letter of a name. Under the others it lies further ahead, up to {{value:jlColonAheadMax}} characters. This is, on the natural reading, what the sum over $t'\ge t$ in $J_\ell$ buys: the J-lens reads what a column is disposed to make the model say, now or later, and the logit lens only what the model would say next if it stopped at this depth. Nothing about the model is clever here. In this corpus every line of capitals that opens a speech is a name ending in a colon, and any model of it learns that early. The point is that the J-lens surfaces a character that is coming more often than the logit lens does, at the depths where it does so at all. The reading is a hypothesis; the test would be the present-only lens $J^{\text{self}}$, which was not computed for this model (@ch-epilogue).

::: aside
The paper's interactive figure for its "Count to five" prompt is the same grid: positions along, layers down, the top lens token in each cell. Its tokens are words, and what it finds in the middle layers are words the model is about to need, *halfway* and *done*, that appear in neither the prompt nor the output. Ours are characters, and what we find is a colon.
:::

## Depth by depth {#sec-depth-by-depth}

@fig-j-lens-depth asks both lenses the questions of @ch-lenses, over {{value:jlShPositions}} held-out positions at each depth: does the lens's first choice agree with the model's, is it the character that comes next, and how far is its distribution from the model's?

::: {.figure #fig-j-lens-depth}
The two lenses of the Shakespeare model against the model and against the text, at its nine depths, over the same held-out positions as in @fig-j-logit-lens-depth. Left: how often the lens's first choice is the model's. Middle: how often it is the character that actually comes next, with the model's own rate and uniform guessing for reference. Right: $D(p_{\text{model}}\,\|\,p_{\text{lens}})$ in nats, the model's distribution first. Filled dots, the J-lens; open dots, the logit lens; crosses, the J-lens with the paper's pair weighting. [Click a depth for its numbers.]{.web}
:::

By these measures the J-lens is the worse lens at every intermediate depth of the Shakespeare model. At depth $3$ its first choice is the model's at {{value:jlShAgreeJThree}} of the positions, against {{value:jlShAgreeLogitThree}} for the logit lens; its accuracy against the text trails by between {{value:jlShGapMin}} and {{value:jlShGapMax}} percentage points, and its distribution is further from the model's at every depth before the last. We report this as measured. Two remarks put it in its place without explaining it away. Next-character agreement is the logit lens's home ground: it asks only what the model would say next, while the J-lens is built to read the present and the future at once, and @fig-j-grid shows what that is good for. And the paper's claim is not about next-token agreement: it reports that the two lenses agree in the last layers and that the J-lens finds *interpretable content* earlier, measured on intermediate concepts its prompts are built to contain. A four-block model of characters cannot test that claim, and these figures do not. The crosses follow the filled dots closely before the last depth, where the two conventions differ only through $\varepsilon$, and at the last depth they are the only marks off the model.

## In the code {#sec-code-jlens}

The lens vectors are the rows of $W_{\text{out}}J_\ell$; the analysis keeps them as the columns of a $D\times V$ frame, the shape @ch-jspace needs:

::: {.code file="jspace/src/JLens.jl" fn="lens_frame" doc=true}
:::

The lens itself is the readout of PicoGPT, called with a matrix in front, and with none for the logit lens:

::: {.code file="jspace/src/JLens.jl" fn="lens_logits" doc=true}
:::

## Exercises {.unnumbered}

::: exercise
Show that if the rest of the model acted on each column separately and linearly, $X_L[:,t]=MX_\ell[:,t]$ for one matrix $M$, then $J_\ell=M$ and the J-lens at depth $\ell$ would be the model's output exactly. Which maps of a transformer break each of the two assumptions?
:::

::: exercise
Prove $\LN_\varepsilon(sy)=\LN_{\varepsilon/s^2}(y)$ for $s>0$. Writing $\LN_\varepsilon(y)=\gamma\odot Cy/\sqrt{\lVert Cy\rVert^2/D+\varepsilon}+\beta$, show that the logits of the lens are an affine function of $x$ divided by $\sqrt{\lVert CJ_\ell x\rVert^2/D+\varepsilon}$, and explain in the language of [PicoGPT, Energies and probabilities](https://tobiasosborne.github.io/PicoGPT.jl/energies/) why a larger $\varepsilon$ acts as a higher temperature on the part of the logits that depends on $x$.
:::

::: exercise
Let $n$ be a vector with $W_{\text{out}}\diag(\gamma)\,C\,J_\ell\,n=0$, with $C$ the centring of the previous exercise. Show that adding $n$ to $x$ can change the probabilities of the lens, but never the order in which it ranks the characters, when $\LN(J_\ell x)$ has $\beta=0$ and $b_{\text{out}}=0$. What does a nonzero $\beta$ or $b_{\text{out}}$ change?
:::

::: exercise
Write the stream at depth $\ell$ in new coordinates, $x'=Mx$ with $M$ invertible, so that the rest of the model becomes $F_\ell\circ M^{-1}$. Show that the averaged Jacobian becomes $J_\ell M^{-1}$, the lens vectors become $M^{-\T}v_a$, and $\inner{v_a}{x}$ does not change. Which object of this chapter transforms like $x$, and which like $v_a$? (This is the covector rule of [PicoGPT, Pulling back](https://tobiasosborne.github.io/PicoGPT.jl/pulling-back/).)
:::

::: exercise
A window has $T(T+1)/2$ causal pairs $(t,t')$. Show that weighting every pair equally gives $J^{\text{pair}}_\ell=\frac{2}{T+1}J_\ell$ when every window has the same length $T$. What goes wrong with this relation if the windows of the average have different lengths, as the prompts of a real corpus do?
:::

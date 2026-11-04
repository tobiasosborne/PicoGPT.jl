# Prologue {#ch-prologue}

::: claim
A transformer is a composition of maps between named spaces. Running it pushes values forward; training pulls one number, the loss, back through the transposed derivatives. This book shows it on a model small enough to see every number.
:::

::: names
| maths | physics | ML |
|---|---|---|
| a point $\theta\in\R^p$ | the coordinates of a configuration | parameters, weights |
| $p=\dim\theta$ | the number of degrees of freedom | parameter count, model size |
| a map $f_\theta$ from inputs to outputs | a response function | model, network |
| evaluating $f_\theta$, value by value | integrating forward | forward pass, inference |
| $\ell:\R^p\to\R$, to be minimised | an energy, a cost | loss, objective |
| the covector $\bar x=\partial\ell/\partial x$ | the linear response, the conjugate variable | gradient |
| applying $Df^\T$ in reverse order | the adjoint equation | backward pass, backpropagation |
:::

## What this is {#sec-what}

This is a book about one function. It takes a piece of text and returns, for every position in it, a probability distribution over the character that comes next. It is built from matrix products, a softmax, a normalisation and one elementwise nonlinearity, composed in a fixed order, and it is the function at the centre of every current large language model. The book derives it, writes it in Julia from scratch, and trains it.

The reader we have in mind has a degree's worth of linear algebra, analysis and probability. For that reader the mathematics of this subject is easy. The obstacle is the language: a field that grew up quickly has named familiar objects in unfamiliar ways, and has given several names to some of them. Softmax is a Gibbs distribution. Cross-entropy is a relative entropy plus a constant. Backpropagation is the chain rule, applied in the cheap order. We say so every time, and then use the words of the field, because they are the words of its literature.

## The specimen {#sec-specimen}

The protagonist of this book is a small transformer that we call *the specimen*. It is trained on a single line,

> to be or not to be

eighteen characters over an alphabet of seven. It has the shape of a real model, with every part present: an embedding, one block of attention with two heads followed by a small network, a final normalisation and a readout. Only the sizes are small: a context of $T=8$ characters, a width of $D=8$, and so on (the shapes are fixed in @ch-task and @ch-vectors). With $L$ blocks, a width $D$, an alphabet of $V$ characters and an inner width $F$, the number of parameters is
$$p=2DV+V+2D+L\,(4D^2+2DF+9D+F),$$
and for the specimen's $V=7$, $D=8$, $F=32$ and $L=1$, that is $p=1007$. Here they all are.

::: {.figure #fig-specimen-anatomy}
The whole specimen: all of its parameters, after training, in the order of the computation from top to bottom. Each matrix is drawn as written, rows top to bottom and columns left to right, with one square per entry whose area is the entry's magnitude, filled when it is positive and open when it is negative; vectors are single columns. One scale serves the whole plate. [Point at a block to learn its name and job, or at a square to read its value.]{.web}
:::

Every figure in this book shows this model, or the larger one of @ch-speaking, and every number in every figure is computed from it as the page is drawn: in your browser if you are reading on screen, and when the book was set if you are reading on paper. Nothing is illustrative: when a figure prints a weight to two decimals, the model computed it.

::: aside
One group of these 1007 numbers can never learn. Look closely at the plate. By the end of @ch-attention you can say which group, and prove it.
:::

The specimen learns its line almost perfectly, and the "almost" is instructive, not a defect. @ch-task shows why no model could do better.

## How to read this book {#sec-how-to-read}

Three habits are used throughout.

**Three names.** Every object is introduced as a map between named spaces, with its index form and matrix form side by side, and it is given three names: the mathematician's, the physicist's and the one used in machine learning. The names table at the start of each chapter lists the objects it defines. In it the mathematical name is set in roman, the physical one in italic, and the field's name in this face, between quotation marks: jargon is shown as jargon. The epilogue collects all of them into one dictionary.

**Two inks.** Besides black, this book is printed in two colours, and each has exactly one meaning. *Value ink*, an indigo, marks the quantities computed going forward: parameters and activations. *Covector ink*, a cinnabar red, marks the adjoints pulled back from the loss, the gradients, written with a bar, $\bar x$. The red pen marks the corrections, and gradients are corrections. Forward computation runs down the page, the backward computation runs up, and position in a text runs from left to right. Nothing else is ever coloured. Printed in one colour, the two inks stay apart by shape: value marks are solid, covector marks are barred or hatched.

::: {.figure #fig-two-inks window=2 pos=1}
The two inks on the last map of the model, $z=W_{\text{out}}x+b_{\text{out}}$, applied to one column. Forward, in value ink: the column $x$ enters at the top, and the matrix turns it into seven scores, one per character, from which the loss of this one prediction is computed. Backward, in covector ink: the loss returns its adjoint $\bar z$, and the transpose sends it back to $\bar x=W_{\text{out}}^\T\bar z$; the matrix receives $\bar W_{\text{out}}=\bar z\,x^\T$. [Scrub the computation, or click another character of the window.]{.web}
:::

**Keys.** A glyph is a picture of numbers in which every mark stands for one number. The first time a glyph appears, its figure carries a key in the margin, headed "How to read": one instance of the glyph, enlarged and drawn from the figure's own data, with hairline leaders to plain notes. A key explains how to read the picture, never what it shows. After its first appearance a glyph is used without one.

::: aside
Figures are numbered by chapter. "Figure 5.2" is the second figure of @ch-attention. Sections are numbered the same way, §5.3, and the margin of each section gives its number.
:::

## Two editions {#sec-editions}

The book exists in two forms made from one source. The web edition is interactive: its figures run the specimen in the page, and they respond to the hand. You can drag a window along the text, click a character, move a temperature, or scrub through a computation with the playhead under the figure. The print edition is the same text and the same figures, typeset for paper; each figure appears in a static state that is complete on its own, and computations that unfold in time are printed as small multiples of their stages. Nothing is held back from print: interaction is only for asking the next question.

## Running the code {#sec-running}

The code is a Julia package of a few hundred lines that uses nothing but the standard library. Each file belongs to a chapter, in the order of the book, and every operation returns its value together with its pullback. No automatic differentiation is used anywhere; @ch-pulling-back derives each pullback by hand, and the tests check every one of them against finite differences.

The code shown in this book is cut from the source files when the book is built, never copied by hand, and the line numbers in its margin are the real line numbers of the file. Here is the parameter count above, as the code computes it:

::: {.code file="src/parameters.jl" fn="nparams" doc=true}
:::

To run it, install Julia (1.10 or later) and, from the root of the repository,

```sh
julia --project -e 'using Pkg; Pkg.test()'     # every pullback against finite differences
julia --project scripts/export_specimen.jl     # train the specimen; write artifacts/specimen.json
```

The second command trains the specimen exactly as this book shows it, and writes every number the figures use. The browser has its own implementation of the same model, in JavaScript, which reproduces the Julia numbers to twelve decimal places; it is what makes the web figures live.

## Exercises {.unnumbered}

::: exercise
Check the parameter count. Attribute each term of $p=2DV+V+2D+L(4D^2+2DF+9D+F)$ to the blocks of @fig-specimen-anatomy, and verify $p=1007$ for the specimen. Which terms grow fastest with $D$?
:::

::: exercise
Try it in @fig-specimen-anatomy. One block of the plate is blank, although the model has been trained. Which is it? Write down a guess for why, and check it against @ch-attention.
:::

::: exercise
Try it in @fig-two-inks. [Click through the characters of the window.]{.web} For most of them the covector marks almost vanish. What does a vanishing $\bar z$ say about the prediction, and at which positions is it large?
:::

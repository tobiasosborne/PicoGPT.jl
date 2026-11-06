# Epilogue {#ch-epilogue}

::: claim
Everything the paper does beyond the reach of two small models, from dictionaries of learned features to training that changes what a model's workspace carries, stands on the object this book computed whole: the rest of the model replaced by its averaged derivative, and read as one covector per thing the model can say.
:::

::: names
| maths | physics | ML |
|---|---|---|
| an encoder $f(x)\ge0$, sparse, and a decoder $x\approx\sum_if_i(x)\,d_i$ with $d_i\in\R^D$, fitted together | a sparse expansion in an overcomplete frame | sparse autoencoder (SAE) |
| a decoder column $d_i$ | one atom of the frame | feature |
| $(\Sigma+\lambda I)^{-1}(\mu_w-\mu)$, a regularised, whitened difference of means | a linear response estimated from fluctuations | template lens |
| a trained map from an activation to a few phrases whose templates reconstruct it | an inverse map, learned | oracle lens |
| the parameters after next-token training only | the system as first prepared | base model |
| further training on dialogues and against a learned score of the answers | a second preparation, starting from the first | post-training |
| fine-tuning on the model's own reflections, then evaluating without them | preparing a state through its answer to a question never asked | counterfactual reflection training |
:::

## What the paper does that we could not {#sec-beyond}

The paper's methods section fits in this book; most of its results do not. Five of them need something our models do not have, and each is worth a paragraph in our own terms.

**Sparse autoencoders.** A sparse autoencoder learns an overcomplete dictionary for a model's activations: many more directions $d_i$ than the stream has dimensions, and for every activation a nonnegative code with few nonzero entries, fitted so that the code reconstructs the activation. Its directions, called features, are the paper's stand-in for everything the model represents, in the J-space or not. Projecting a feature's direction through the J-lens gives a distribution over the vocabulary; its excess kurtosis, the paper's *lens kurtosis* $\kappa$ (the readout kurtosis of @ch-workspace, applied to a feature instead of a column), says how sharply the feature names a few tokens, and so how far it lies in the J-space. With it the paper counts what fraction of the model's features are J-space features (about one in seven at most, once those that only predict the next token are excluded), finds that they are amplified by the MLPs more than the others, and finds the others dominated by syntax and bookkeeping. An autoencoder for the Shakespeare model is within reach of a laptop. It is not in this book, and the question it would answer, whether the J-space is a small corner of a much larger repertoire, is the one our $V<D$ geometry makes least natural to ask.

**Words of several tokens.** A J-lens vector exists only for a single token, so the paper's lens cannot name a concept whose name is several tokens long; *blackmail* appears as *black*. Its *template lens* builds a vector for any word $w$ from contexts written to end just before $w$: the mean activation there, minus the mean over all words, multiplied by the inverse covariance of the activations. A lemma of Charles Stein connects this to the J-lens: for Gaussian activations $x$ with covariance $\Sigma$, $\mathbb E[\nabla g(x)]=\Sigma^{-1}\mathbb E[g(x)(x-\mu)]$, so the whitened difference of means is an averaged gradient, as a J-lens vector is. Read the other way, the identity says that the concept vector $m_a$ of @ch-writing, a plain difference of means, is for Gaussian activations $m_a=\Sigma\,\mathbb E[\nabla_xP(a\mid x)]/P(a)$: an averaged gradient, of the probability of $a$ rather than its logit, turned into a vector with $\Sigma$ instead of the identity. It is the conversion $v\mapsto v^\sharp$ of @sec-steer in another inner product. The *oracle lens* goes further and trains two copies of the model, one to turn activations into phrases and one to turn phrases back into directions. Our alphabet is characters, so every word is a word of several tokens, and none of this was attempted.

**Behavioural experiments.** Most of the paper asks a model to do something and looks inside while it does: think of a sport and name it; hold citrus in mind while copying a sentence about a painting; report a thought that was injected into its activations; solve a two-step question without writing the intermediate step; decide what to do in a scenario built to tempt it. The J-lens shows the concept before the model says it, and a swap of lens coordinates changes what it says. These experiments need a model that follows instructions and talks about itself. A model of Shakespeare's characters cannot be asked anything; it can only be continued.

**Comparisons across training.** The paper applies the same lens to a base model and to the model after post-training, and finds that on the user's words the post-trained model already represents the assistant's assessment of them (that a dose is unsafe, say) where the base model represents only the words. Our models have one stage of training, and nobody to talk to.

**Training the J-space directly.** In *counterfactual reflection training* the paper fine-tunes a model on reflections it writes when asked, mid-task, what principles apply, and then evaluates it on the same tasks with no question asked. The fine-tuned model is more honest, its J-space on the unchanged prompts now carries words like *ethical* and *transparent*, and ablating the lens vectors of those words removes almost all of the improvement on one of the paper's two benchmarks and part of it on the other. It is the paper's most direct evidence that what a model is disposed to say shapes how it reasons silently, and it needs everything our models lack at once: dialogue, a written set of principles, and a model that can reflect.

::: aside
Stein's identity is from C. M. Stein, "Estimation of the mean of a multivariate normal distribution", *Annals of Statistics* 9 (1981). The paper's appendix "Extending the Jacobian lens to multi-token concepts" uses it, under idealised assumptions, to motivate the template lens, with $g$ the probability that the continuation is $w$; activations are not Gaussian, and the analogy is a motivation, not an identity.
:::

## What the small models leave open {#sec-open}

Some questions this book raised can be answered with the tools it built, at little cost.

*Depth.* The Shakespeare model showed no band of depths in four blocks (@sec-band). A model of the same width with eight or sixteen blocks, trained on the same text, would say whether depth alone produces one; every statistic of @fig-j-band is a function the analysis already has.

*Vocabulary.* With characters, $V<D$, and the J-space is a subset of a $65$-dimensional subspace. A model of the same text with a vocabulary of a few thousand tokens would have $V\gg D$ and a J-space of thin cones in a space its lens vectors span many times, as in the paper; the questions of capacity in @sec-capacity would then have the paper's shape.

*Averaging.* At every intermediate depth of the Shakespeare model the J-lens predicts the next character a little worse than the logit lens. The J-lens was not built for that: its average runs over every later position a column can affect, so it measures what the column disposes the model to say from here on, not next. The self-only average $J^{\text{self}}$ of @ch-jacobian isolates the next position; the analysis computed it exactly for the specimen and omitted it for the Shakespeare model to save time. Computing it would say how much of the gap is the average over the future.

*Construction.* The attention heads preserve J-lens directions far better than a rotated control (@sec-ov), and the J-lens vectors were defined by pulling back through those heads. The control that separates inheritance from arrangement is described in an exercise of @ch-workspace and not computed.

*Training.* The paper does not know when in training its workspace appears. For the specimen the analysis exported the J-lens vectors at every one of its two hundred steps (`jspace/artifacts/specimen-jlens-training.json`), and how they settle is left to the reader.

The paper closes its list of open questions by asking whether smaller models have a workspace as rich, a smaller one, a less reliable one, or none. A model of a thousand parameters and one of eight hundred thousand cannot settle that. They do show what each piece of the question is made of, down to the last number.

## A dictionary of jargon {#sec-dictionary}

Every term given three names in this book, in alphabetical order, with its names in mathematics and in physics and the chapter that defines it. The dictionary is assembled when the book is built, from the names tables at the head of each chapter; it is exactly as complete as they are.

::: dictionary
:::

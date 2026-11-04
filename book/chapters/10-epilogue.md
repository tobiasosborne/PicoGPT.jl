# Epilogue {#ch-epilogue}

::: claim
What this book left out changes how much is computed, on what hardware and in how many bits, far more than it changes what is computed: a large language model is still a composition of maps between named spaces, pushed forward to predict and pulled back to learn.
:::

::: names
| maths | physics | ML |
|---|---|---|
| an alphabet of strings $\mathcal A\subset\Sigma^*$ over bytes $\Sigma$ | a coarse-graining of the text | token vocabulary, subword tokens |
| repeated merging of the most frequent adjacent pair | a renormalisation step, run greedily | byte-pair encoding, BPE |
| floating-point numbers with fewer bits of mantissa | a coarser grid of representable values | reduced precision |
| rotating $q_i$ and $k_j$ by $R(i)$ and $R(j)$ before $q_i\cdot k_j$ | a phase convention depending only on $i-j$ | rotary position embedding, RoPE |
| the stored columns $k_j,v_j$ of the past, reused | a memo | KV cache |
| minimising a second loss, starting from the first minimiser | relaxing from a prepared state | fine-tuning |
:::

## What we left out {#sec-left-out}

The specimen is complete in its maps and small in everything else. The distance from it to the models in the news is real, and it lies in a few directions that can be named.

### Tokens instead of characters

Our alphabet is the set of characters of the text. Production models use an alphabet of *tokens*: strings of bytes chosen once, before training, so that common words are one token and rare words a few. The usual construction is *byte-pair encoding*. Start from the 256 byte values; count every adjacent pair of symbols in a large sample of text; add the most frequent pair to the alphabet as a new symbol and rewrite the sample with it; repeat until the alphabet has the desired size, typically tens of thousands to a few hundred thousand.^[The algorithm was introduced for data compression by [Gage (1994)](https://jacobfilipp.com/DrDobbs/articles/CUJ/1994/9402/gage/gage.htm) and adapted to language by [Sennrich, Haddow and Birch (2016)](https://aclanthology.org/P16-1162/).] The result is a finite alphabet $\mathcal A$, and nothing in this book depends on what its elements are: $V$ grows, $E$ and $W_{\text{out}}$ get more columns and rows, and every map is unchanged.

What changes is the text the model sees. A token covers about four characters of English on average, so a context of $T$ tokens holds about four times as much text, and the chain rule of @ch-task is applied to fewer, larger symbols. The price is that the model never sees the letters inside a token, which is why large models can be oddly bad at spelling and at counting characters.

### Scale

The specimen has about a thousand parameters, and the Shakespeare model of @ch-speaking a little under a million. Published language models have had from hundreds of millions to hundreds of billions.

::: {.figure #fig-scale}
How big is big. Each panel is drawn the same size; in its corner sits the previous model at its true relative scale. The specimen panel shows one square per parameter, with a minimum visible size for the smallest entries. From the left: the specimen; the Shakespeare model, trained on the works of Shakespeare; the largest GPT-2 (2019) ([Radford et al., 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)); and GPT-3 ([Brown et al., 2020](https://arxiv.org/abs/2005.14165)). These counts are published measurements, rather than computations from our own models.
:::

Size has a second axis, the data. The specimen sees the same {{value:predictions}} predictions at every step. A large model sees trillions of tokens, most of them only once. Over many orders of magnitude, the loss reached by such models falls smoothly, close to a power law in the number of parameters and in the number of tokens trained on,^[Kaplan et al., [*Scaling Laws for Neural Language Models* (2020)](https://arxiv.org/abs/2001.08361); Hoffmann et al., [*Training Compute-Optimal Large Language Models* (2022 preprint)](https://arxiv.org/abs/2203.15556).] and the practice of the field is largely the practice of exploiting that regularity. None of it requires a new map; all of it requires more of the same ones.

### Hardware

Almost all the arithmetic of the model is matrix products: the four maps of attention, the two of the MLP and the readout, in both directions. A graphics processor performs such products thousands of times faster than the loops of our Julia, and production code is organised around keeping it busy: long batches, fused kernels, and parameters and activations split across many devices.

Two costs we never felt dominate at scale. Attention computes a $T\times T$ score matrix for every head and every window, so its cost grows with the square of the context; fast implementations compute the masked softmax of @ch-attention in tiles, never storing the scores, and never computing the upper triangle at all.^[Dao et al., [*FlashAttention* (NeurIPS 2022)](https://arxiv.org/abs/2205.14135).] And when a model writes, each new character needs the keys and values of every earlier position; they are stored once and reused, in a *KV cache*, instead of recomputing the whole window at every step as `generate` does in @ch-speaking.

### Precision

The specimen computes in 64-bit floating point, with the Julia and JavaScript implementations checked against one another. Arrays agree to $10^{-12}$ absolute; the recorded loss curve to $10^{-9}$ absolute. A replay that fails the latter check is discarded. The Shakespeare model trains in 32 bits. Large models train with 16-bit numbers for most of the arithmetic, keeping a 32-bit copy of the parameters for the update and, in the narrowest formats, rescaling the loss so that small gradients survive the coarser grid.^[Micikevicius et al., [*Mixed Precision Training* (ICLR 2018)](https://arxiv.org/abs/1710.03740).] They are often served with fewer bits still. The maps are the same; the rounding is not, and a good part of the engineering is in keeping it harmless.

### Other architectures

Our block is the *pre-norm* transformer: LayerNorm before each branch, the residual stream left untouched between them. The original transformer of 2017 normalised after the sum; the pre-norm order trains more reliably in deep stacks.^[Vaswani et al., [*Attention Is All You Need* (NeurIPS 2017)](https://arxiv.org/abs/1706.03762); Xiong et al., [*On Layer Normalization in the Transformer Architecture* (ICML 2020)](https://proceedings.mlr.press/v119/xiong20b.html).] Since then the parts have changed while the skeleton has not:

- Normalisation without centring, which divides by the root mean square only (RMSNorm).
- Positions as rotations of queries and keys (RoPE), which is the rotation property of @ch-vectors made into the mechanism.^[Su et al., [*RoFormer* (2021 preprint)](https://arxiv.org/abs/2104.09864).]
- A gated MLP, in which the output of the nonlinearity is multiplied, entry by entry, by a second linear image of the input.^[Shazeer, [*GLU Variants Improve Transformer* (2020)](https://arxiv.org/abs/2002.05202).]
- Several query heads sharing one key and value head, to shrink the cache; and *mixtures of experts*, in which each token passes through only a few of many MLPs.
- The embedding and the readout as one matrix, $W_{\text{out}}=E^\T$. Ours are independent.

Further away are models whose mixing step is not attention at all, such as the state-space models, which replace the $T\times T$ kernel by a recurrence that is linear in its state.^[Gu and Dao, [*Mamba* (2023 preprint)](https://arxiv.org/abs/2312.00752).]

### After the first loss

A model trained only to predict the next token of a large text continues text; it does not answer questions. The models one talks to have been trained further: on curated dialogues, with the same loss, and then against a learned score of how well a human would rate the answer.^[Ouyang et al., [*Training language models to follow instructions with human feedback* (NeurIPS 2022)](https://arxiv.org/abs/2203.02155).] These stages change the data and the objective. The pullbacks of @ch-pulling-back, and the optimiser of @ch-learning, are used unchanged.

## A dictionary of jargon {#sec-dictionary}

Every term given three names in this book, in alphabetical order, with its names in mathematics and in physics and the chapter that defines it. The dictionary is assembled when the book is built, from the names tables at the head of each chapter; it is exactly as complete as they are.

::: dictionary
:::

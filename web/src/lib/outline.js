// The book's planned arc (BRIEF §8). A chapter file `book/chapters/NN-slug.md`
// overrides the planned title; until it exists the contents page lists the
// chapter as "in preparation" (never as a link to placeholder text).
// Slugs are the URL path (/attention/) and the file name after the number.
export const OUTLINE = [
	{ num: 0, slug: 'prologue', title: 'Prologue', gist: 'What this is. The specimen. How to read it: three names, two inks.' },
	{ num: 1, slug: 'task', title: 'The task', gist: 'Text as a finite sequence. The chain rule of probability makes the model autoregressive: a language model is a map from contexts to distributions.' },
	{ num: 2, slug: 'vectors', title: 'Symbols become vectors', gist: 'One-hot vectors, the embedding matrix, positions as Fourier features.' },
	{ num: 3, slug: 'energies', title: 'Energies and probabilities', gist: 'Logits are negative energies; softmax is the Gibbs distribution. Temperature. Shift invariance and log-sum-exp.' },
	{ num: 4, slug: 'surprise', title: 'Measuring surprise', gist: 'Cross-entropy, KL divergence, maximum likelihood. The baseline ln V. The gradient: predicted minus observed.' },
	{ num: 5, slug: 'attention', title: 'Attention', gist: 'A row-stochastic mixing of columns; query, key, value; the 1/√d scaling; the causal mask as lower-triangularity; heads as a direct sum.' },
	{ num: 6, slug: 'block', title: 'The block', gist: 'The residual stream as an Euler step. LayerNorm as projection to a sphere. The MLP. The whole model as one composition.' },
	{ num: 7, slug: 'pulling-back', title: 'Pulling back', gist: 'The chain rule as a composition of transposes. Each pullback derived. The gradient check.' },
	{ num: 8, slug: 'learning', title: 'Learning', gist: 'Gradient flow and its discretisation; minibatch noise; Adam as diagonal preconditioning; schedules; clipping; initial variances.' },
	{ num: 9, slug: 'speaking', title: 'Speaking', gist: 'Sampling, temperature, and what the Shakespeare model learned, shown honestly.' },
	{ num: 10, slug: 'epilogue', title: 'Epilogue', gist: 'What we left out — BPE, scale, GPUs — and a dictionary of jargon.', unnumbered: true }
];

export const BOOK = {
	title: 'PicoGPT',
	subtitle: 'A transformer small enough to see every number',
	author: 'Tobias J. Osborne',
	repo: 'https://github.com/tobiasosborne/PicoGPT.jl'
};

/** Chapter number as printed: 0 → "0", epilogue → "" */
export const chapterLabel = (c) => (c.unnumbered ? '' : String(c.num));

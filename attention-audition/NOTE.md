# Attention, from the top down — note for the author

**Read:** `attention-audition/AttentionDraft.pdf` (18 pages; the chapter is pp. 7–18).
Source: `attention-audition/book/chapters/05-attention.md`. The book's own
`book/chapters/05-attention.md` is untouched. Rebuild with
`node print/build.mjs --edition=attention-audition` (see `README.md` here).

## What changed, and why

The old chapter builds attention bottom-up: an average, then a score, then a mask. The draft
follows the route of your ChatGPT conversation. The conditional $P(y\mid c)$ is an exponential
table, and the chain rule gives no recipe for it. Every map so far is column-wise, which makes
a bigram model with a clock. Attention is then reached by five steps, each stated as exact or as
a restriction, with when it holds and when it fails:

1. **An uncertain index.** It is exact and empty. The enlarged law is *chosen*: the uniform choice
   returns $P$ in every term, which answers your turn-7 question. The first real assumption is
   that some choice makes both factors simple, and nothing guarantees that.
2. **One shared rule on position vectors.** This is where the value $v_j$ is named.
3. **A Gibbs distribution over sources with a bilinear energy.** The biases are expanded exactly,
   which shows why $b_k$ never learns.
4. **Rank ≤ d.** The two factors are all that "query" and "key" mean.
5. **The mean-field step.** The vectors are averaged, not the predictions.

Jargon is named only after its object is on the page. The mask comes after the five steps, as
"every reader at once": the batched form of the constraint $j\le i$ that was imposed in Step 1.
Heads are "one index per subspace".

**New content, checked twice:**
- With a readout whose logits are affine in the attention output, the mean-field step turns the
  arithmetic mixture of per-source predictions into their normalised *geometric* mixture. One
  source can vote a character out.
- The real model is not affine, so the draft states the identity with its true scope.
- The specimen measures it over all 80 readers, both heads chosen independently:

  | prediction | mean loss (nats) |
  |---|---|
  | actual | 0.0589 |
  | geometric mixture | 0.0624 |
  | arithmetic mixture | 0.1980 |

  The mean total variation from the actual prediction is 0.0034 for the geometric mixture and
  0.12 for the arithmetic one.
- The draft says plainly that the weights were trained for the mean-field predictor, so this
  measures how far the network sits from the "select one source" picture. It is not an error of
  the network.
- One new figure, Fig. 5.3 (Opus), shows this for one real reader: `or not t`, position 6, head 1.
  A strip below it covers all 70 readers with more than one source; the geometric mixture is the
  closer at every one of them.

## What was kept

Every fact of the old chapter, reorganised:
- every position is a reader, and one pass gives the $T$ conditionals;
- rows are readers and columns are sources, $O=VA^\T$;
- the mask as lower-triangularity, and the causal Jacobian;
- why $\sqrt d$ (its own subsection, so Chapter 8's reference still resolves);
- heads as a direct sum, recombined by $W_o$;
- 288 attention parameters;
- the code section, verbatim;
- the three old figures, with captions rewritten for the route.

There are five exercises. Three are old: two of these had to be corrected, and one is restated
in full sentences. One is new: the data-dependent average that computes $ab$.

## Errors found in the *current* book chapter (fix them whatever you decide)

Astra found these in `book/chapters/05-attention.md` as it stands:
- **The mask aside.** It says half the scores are thrown away. It is 28 of 64 per head and window.
- **The √d paragraph.** It says score variance alone freezes the softmax. Variance does not
  determine row entropy; what concentrates a row is the gaps between the largest score and the
  rest.
- **The permutation exercise.** It holds only for a one-block model; it fails from the second
  block on.

## What the reviewers still object to

Reviewers were Astra for the mathematics and Muse as the intended reader. Both reviewed revision
1. The final readability pass was prose only, with every claim frozen. I checked that pass
mechanically and tightened two sentences myself. Neither reviewer has read the final text.

- **Length (Muse, and your brief).** The draft is **2,288 words against 1,609 (+42 %)**. A
  revision cut to 1,800 words became telegraphic notes, so I traded length back for readable
  prose. Cut candidates:
  - the exact "if and only if" condition in Step 4;
  - the coupling remark in Heads;
  - Step 2's operational definition of $g_\theta$.
- **Steps 1→2 (Muse).** The draft says what the hypothesis excludes, but gives no reason to
  believe text satisfies it. Muse suggests promoting the $ab$ exercise into the text as the
  miniature that shows it can work.
- **Why the specimen sits so close to the geometric mixture (Muse).** The draft leaves this as an
  empirical fact. The quantities that would explain it are defined but not evaluated.
- **The matrix product state aside (Muse).** Muse would delete it, because the reader does not
  know MPS. I kept it, short and corrected (rank across a cut, not a correlation length), as a
  margin note for you to judge.

## Choices for you

1. **The route.** Accept it, reject it, or redirect it.
2. **Length.** Accept about 2,300 words, or cut, using the candidates above.
3. **Matrix product states.** Keep the MPS aside or delete it. The one-particle (direct-sum,
   transfer-operator) picture is *left out*: it needs ket notation that nothing else in the book
   uses. It could become an epilogue section or a supplement instead.
4. **Vocabulary.** The draft says Steps 2–4 "restrict" the family (each is an approximation
   because the restricted family may not contain $P$) and Step 5 "replaces one predictor by
   another". Is that the vocabulary you want?
5. **Fig. 5.3.** It is the densest figure in the book: a plate plus a strip of all readers. Keep
   all of it, or only the plate?
6. **If you adopt the draft.** A short pass would be needed to move it into `book/`: wire the
   figure and values into the main edition, and re-run both reviews on the final text.

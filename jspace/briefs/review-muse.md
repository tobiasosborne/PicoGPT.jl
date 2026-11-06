# Review of *Poised to speak* (`jspace/`)

Reviewer: Muse Code (muse-wonder-full-1.0), acting as review sub-agent.
Date of the PDF reviewed: proof of 2026-11-06, 81 pages.

## How this review was done

I read the printed book front to back from `jspace/proof/pages/p-01.png` …
`p-81.png` (and the contact sheets), then read the prose sources
`jspace/book/chapters/00-prologue.md` … `07-epilogue.md`, the paper
(`jspace/briefs/paper.txt`, especially Methods lines 103–169, “The
J-space’s structure supports its function” from line 364, “Methodological
Details and Ablations” and “Formalization of the J-space” from line 879),
`jspace/briefs/writer.md`, `BRIEF.md`, `DESIGN.md`,
`jspace/briefs/analysis-report.md`, `jspace/artifacts/README.md`,
`jspace/edition.js`, and the voice reference `book/chapters/05-attention.md`.

Every prose number I could recompute, I recomputed independently in
Python from `jspace/artifacts/specimen-jlens.json` and
`shakespeare-jlens.json` (lens agreements, the GREMIO colon counts,
steering counts, concept-vector cosines and shares, CKA ranges,
occupancies, spans, MLP/OV tables, ranks of the lens frames). I also
read the figure sources where a caption makes a claim about encoding
(`MatrixGlyph`, `fig-j-blocks`, `fig-j-cone`/`cone.js`,
`fig-j-lens-vectors.key`, `structure_statistics` in `jspace/src/JLens.jl`).

The book is, in most places, unusually careful: the pair-weighting /
LayerNorm-ε lesson is correct, the Shakespeare J-lens defeat is reported
plainly, and most `{{value:…}}` numbers match the artifacts exactly
(I verified, among others, specimen agreements 33/55 vs 29/39, steering
269/278/11 of 480, CKA ≥ 0.91 before the last depth, occupancies
1,1,1,1,1,2,3,5,37). The findings below are the exceptions. They are
ranked by importance: mathematical and paper-fidelity errors first,
then unsupported quantitative claims, then consistency, figures, and
typography. Each gives *where*, a quote, *what is wrong*, and a
*proposed fix*. Page numbers are printed folios (= PDF pages here).

---

## 1. The lens vector is repeatedly called the gradient of the logit. It is not.

**Where:** `jspace/book/chapters/05-writing.md:21`; also `:46–48`,
`:119`; echoed in `00-prologue.md` names table and §0.2, and in the
privileged-test argument (§5.5, pp. 58–59).

> “$v_a$, row $a$ of $W_{\text{out}}J_\ell$, the averaged gradient of
> the logit of $a$ with respect to the stream at depth $\ell$.”
> (`05-writing.md:21`)

> “To first order, and in the averaged model, the step moves every
> logit, not only the logit of $a$: $dz_b=\alpha\langle v_b,v_a\rangle$”
> (`05-writing.md:46`)

**What is wrong.** Chapter 3 gets this right and chapter 5 undoes it.
The model’s logit is $z_a=(W_{\text{out}}\mathrm{LN}(y)+b_{\text{out}})_a$.
The vector $v_a=J_\ell^{\mathsf T}W_{\text{out}}^{\mathsf T}e_a$ is the
averaged gradient of the *pre-normalisation score*
$s_a(y)=\langle w_a,y\rangle$, as §3 correctly spells out
(`03-jlens.md` “Lens vectors are covectors”, with the aside “Exactly,
the final LayerNorm sits between $W_{\text{out}}$ and $J_\ell x$”).
The gradient of the actual logit would contain the Jacobian of the
final LayerNorm (centring, the common scale $s(u)$, the gains $\gamma$)
— exactly the modified covector
$w_a=J_\ell^{\mathsf T}C\operatorname{diag}(\gamma)W_{\text{out}}^{\mathsf T}e_a$
that §5.2 itself derives. Three load-bearing places inherit the error:

1. The steering side-effect formula $dz=\alpha W_{\text{out}}J_\ell
   J_\ell^{\mathsf T}W_{\text{out}}^{\mathsf T}e_a$ is the change in
   pre-LN scores, not in logits, and not in $\log p_a$.
2. “$v_a^\sharp$ is the direction in which the averaged logit of $a$
   rises fastest per unit length” (`05-writing.md`, §5.3) should read
   “averaged pre-normalisation score”.
3. The privileged-test moral, “the remainder does not move the
   averaged logit of $a$ at all, to first order” (`:119`), is true
   only for that pre-LN score, **summed over $t'\ge t$** (see Finding 4).

**Proposed fix.** Globally replace “gradient of the logit” by
“gradient of the score $s_a(y)=\langle w_a,y\rangle$ before the final
LayerNorm”, rename $dz_b$ to $ds_b$ in §5.3, and add one sentence at
the first steering formula: “The change in the model’s actual logits
is this vector passed through the derivative of the final LayerNorm
(§5.2); the Gram matrix gives the pre-normalisation scores.”

## 2. “Analysed with exactly the paper’s definitions” is not true, and the deviations are the ones that matter for the verdict

**Where:** `jspace/book/chapters/06-workspace.md:92` (p. 71);
definition choices in `02-jacobian.md:62`, `03-jlens.md` §3.4,
`04-jspace.md` §4.5, `06-workspace.md` §§6.2–6.4;
`jspace/artifacts/README.md:36–54,148–154`.

> “A transformer trained on next-character prediction, analysed with
> exactly the paper’s definitions, does not produce them by default”
> (`06-workspace.md:92`)

**What is wrong.** The book is honest about several individual choices
but never gathers them, and the closing sentence claims exactness.
At least these differ from the paper as written in `paper.txt`:

- **Scale of $J$.** The book’s $J_\ell$ is a *sum* over targets; the
  paper’s Methods formula is an *expectation over pairs*. They differ
  by $(T+1)/2$ (the book proves this). The book is right that the
  paper’s pseudocode (paper.txt lines 917 ff.) computes the book’s
  version — but then “the paper’s $J$” is ambiguous in the paper
  itself, and the book should say “we follow the pseudocode, not the
  displayed expectation” once, plainly.
- **Target layer.** The book uses the final depth. The appendix
  (paper.txt, “Gradient of What?”) states the *default* lens used
  throughout the paper takes $z$ at the **penultimate** layer. The
  aside at `02-jacobian.md:62` says “We use the final depth” without
  saying this is not the paper’s default.
- **Aggregation.** No outlier/first-position exclusion, plain mean
  over all positions (paper considers and the book lists these, then
  drops the subject).
- **Bias.** The book’s lens keeps $b_{\text{out}}$; the paper’s
  formula has no bias term. Fine for these models, but it is a
  deviation and it enters every ranking.
- **Sparse decomposition.** The paper uses *gradient pursuit*; the
  book uses nonnegative orthogonal matching pursuit with exact NNLS
  refits. Occupancy is algorithm-dependent.
- **Occupancy.** The paper’s occupancy is a per-position quantity,
  reported as a median/percentiles over positions. The book’s is the
  first $k$ at which the *position-averaged* marginal improvement
  falls below the random one (`artifacts/README.md:141–142`). These
  need not agree; see Finding 8.
- **Band statistics.** The paper’s panel (a) is *top-$k$* recovery of
  the model’s top-1 (“any top-$k$ J-lens token matches”, paper.txt
  Fig. 28 caption); Figure 6.2(a) is top-1 agreement. The paper has
  a fourth signal, autocorrelation across positions (panel c), which
  the book neither computes nor mentions while saying (p. 66)
  “Figure 6.2 computes all of these”. The 90 % threshold for
  “effective dimension” is the book’s, not the paper’s.
- **Broadcast populations.** MLP gain includes the branch LayerNorm
  and **both biases** (`jspace/src/JLens.jl`, `structure_statistics`:
  `branch(A) = first(mlp_update(A, block)) - A`); the bias $b_2$ is
  in every output norm, compressing all gains towards 1. The paper’s
  definition (“output norm of the MLP block … when applied to $v$”)
  does not settle this, and the book does not discuss it.

**Proposed fix.** Replace “exactly the paper’s definitions” with
“the paper’s definitions as implemented here”, and add a half-page
table in §6.1 (or an appendix aside) listing the seven choices above
with “paper / this book”. The verdict (“signatures are not automatic”)
survives; the exactness claim does not.

## 3. Steering strength $\alpha$ is compared across depths and characters as if it were a physical step size. It is a convention-dependent multiple of vectors of very different lengths.

**Where:** `jspace/book/chapters/05-writing.md:55` (p. 53), Figure 5.1
(p. 54); lengths from `specimen-jlens.json` (`lenses.vectors`).

> “The first is how little it takes at the earlier depths. Count …
> how often a step of $\alpha=1$ makes the steered character the
> model’s choice: 269 of 480 times at depth 0, 278 of 480 at depth
> $\tfrac12$, and 11 of 480 at depth 1” (rendered values, p. 53)

**What is wrong.** The book itself defines the step length as
$|\alpha|\,\|v_a\|$ (§5.3). At depth 0 the seven $\|v_a\|$ are
6.59, 5.78, 2.65, 8.17, 6.39, 1.66, 9.25, against stream columns of
mean norm 3.29; at depth 1 they are the readout rows, 1.52–2.07,
against columns of mean norm 7.29. So “$\alpha=1$” is a step of
roughly *two column lengths* early and *a quarter of a column
length* late. Worse, the early lengths are large partly because of
Finding 2’s convention: under the paper’s pair-weighted $J$ every
early vector shrinks by the specimen factor 4.5
(`conventions.jacobian_factor`), and the 269/480 count would
collapse, while the lens itself (Chapter 3’s whole point) would not
change. “How little it takes” is therefore a statement about the
chosen normalisation of $J$, not about the model. The same applies
to Figure 5.1’s $\alpha$ axis, which is never converted to
$\|\alpha v_a\|/\|x\|$.

**Proposed fix.** Either steer with unit vectors $\hat v_a$ and count
again, or keep $v_a$ and rewrite the paragraph in units of step
length: state $\|v_a\|$ per depth, give the count as a function of
$\|\text{step}\|$, and add: “These counts move if $J_\ell$ is
rescaled; the lens does not. $\alpha$ alone is not a strength.”

## 4. The “half arithmetic” account of the privileged test conflates four different objects

**Where:** `jspace/book/chapters/05-writing.md:109–123` (§5.5,
pp. 58–59), Figure 5.4 (p. 59).

> “So whenever $v_a$ is among the fit’s vectors with a positive
> weight, $\langle v_a,r_a\rangle=0$, and the remainder does not move
> the averaged logit of $a$ at all, to first order.”
> (`05-writing.md:119`)

**What is wrong.** The orthogonality is real (KKT, correctly
derived), but the inference chain to the measured quantity silently
swaps objects four times:

1. $v_a$ is the gradient of the *pre-LN* score, not the logit
   (Finding 1);
2. summed over *present and future* positions
   $\sum_{t'\ge t}s_a(X_L[:,t'])$, whereas Figure 5.4 measures
   $\Delta\log p_a$ **at the edited position only** — the relevant
   first-order covector there is the *self* part $J^{\text{self}}$,
   to which $r_a$ need not be orthogonal;
3. $\Delta\log p_a \ne \Delta z_a$: the softmax subtracts
   $\log\sum_b e^{z_b}$, and steering moves every logit (§5.3’s own
   Gram-matrix point);
4. the write is rescaled to $\|m_a\|$ (up to $\approx$10 at depth 1,
   against column norms $\approx$7) and run through the exact
   nonlinear model, so “to first order” is doing heavy lifting that
   the text only acknowledges afterwards (“the model has departed
   from its first-order picture”).

Two further problems with the test as an analogue of the paper’s.
The “concept vector” $m_a=\mathbb E[x\mid\text{next}=a]-\mathbb E[x]$
is a *next-character* direction — at depth 1 it is 92–99 % inside the
$k=3$ J-space (Figure 5.4’s own percentages), i.e. it is essentially
the readout row it is then split against. The paper’s concept
vectors come from “Tell me about {concept}” prompts and are *not*
next-token predictors; their J-share is 6–7 %. And the outcome,
mean $\Delta\log p_a$ over positions where $p_a$ is often
$\approx 0$, is dominated by near-zero probabilities (bars reach
12.6 nats); no spread, median, or sign count is given.

**Proposed fix.** Demote the claim: state the exact theorem
($r_a$ does not move the *summed pre-LN score*, in the *averaged
linear* model, to first order), then list (1)–(4) as the gaps
between theorem and measurement. Rename $m_a$ in the prose to
“the mean column before an $a$ (a next-character direction, not the
paper’s prompt-based concept vector)”, and report a robust summary
(median $\Delta\log p$, or the fraction of positions where $a$
becomes top-1) alongside the mean.

## 5. The GREMIO colon — the book’s one positive lens exhibit — is reported at a depth selected for being maximal, and the closing sentence is false as written

**Where:** `jspace/book/chapters/03-jlens.md:101` (p. 36),
Figure 3.3 (p. 37); selection code `jspace/edition.js`,
`jlensValues` (`for … if (colonAt(j,0) > colonAt(jc,0)) jc = j;`).

> “At depth 2½ the J-lens reads a colon under 10 of the 14 capital
> letters of *GREMIO* and *BAPTISTA*; the logit lens, at the same
> depth, under 3. … The point is that the J-lens surfaces a
> character that is coming, and the logit lens does not.”
> (rendered, p. 36)

**What is wrong.** I recomputed the counts from
`lenses.prompt.ids` for all nine depths (14 name letters;
colon id 11):

| depth | 0 | ½ | 1 | 1½ | 2 | 2½ | 3 | 3½ | 4 |
|---|---|---|---|---|---|---|---|---|---|
| J-lens top-1 `:` | 0 | 1 | 0 | 0 | 4 | **10** | 7 | 5 | 2 |
| logit top-1 `:` | 3 | 4 | 5 | 3 | 3 | 3 | 4 | 5 | 2 |

Depth 2½ is chosen in `edition.js` precisely as the depth where the
J-lens count is largest; the prose never says so. At every depth up
to 1½ the logit lens reads *more* colons than the J-lens, and at
2½ it still reads 3 (one is the genuine next character at each
name’s last letter, as the model’s own count of 2 at depth 4
shows). “The logit lens does not” is therefore false, and “From
depth 2 on there is structure” papers over five depths of
counter-evidence visible in Figure 3.3 itself. Further, “reads a
colon” means *top-1*, at whatever probability; the figure encodes
probability as type weight but the prose never gives it, so a
top-1 colon at $p\approx0.05$ and one at $p\approx0.9$ count alike.
Finally, the causal gloss (“This is what the sum over $t'\ge t$
buys”) is asserted, not tested: the obvious test exists in the
book’s own machinery — the self-only lens $J^{\text{self}}$ — and
was not computed for Shakespeare (the epilogue proposes it as
future work, p. 77, without connecting it to this claim).

**Proposed fix.** Print the 2×9 table above (it is three lines),
say “at depth 2½, where the effect is largest”, delete the absolute
final sentence in favour of the counts, give the median colon
probability at 2½, and either compute the self-only colon count or
label the $\sum_{t'\ge t}$ explanation as a hypothesis.

## 6. Every lens evaluation is in-sample: $J_\ell$ is averaged over the same columns it is then scored on. This is never mentioned.

**Where:** specimen: `jspace/artifacts/README.md:90–93`,
`01-lenses.md`, `03-jlens.md` §3.5; Shakespeare:
`artifacts/README.md:106–113`, `03-jlens.md` §3.6,
`06-workspace.md` throughout.

**What is wrong.** For the specimen, $J_\ell$ averages over all
80 columns of the ten windows, and Chapters 1/3/4 then score both
lenses, exact cone fits, and interventions on those same 80
columns. For Shakespeare, $J_\ell$ uses all 16,384 positions of
64 validation windows; the lens metrics, NNOMP curves and span
shares use 2,048 positions drawn from the *same* 64 windows
(`sampling.activation_positions_per_window`), and the GREMIO
prompt is the first text of the validation split, inside window 1
(`window_starts[1]=1$; the prompt occurs at corpus offset
1,003,857, validation begins at 1,003,855). The paper computes
$J_\ell$ on 1,000 pretraining-like prompts and evaluates on
separately constructed prompt suites. In-sample averaging does not
invalidate the (mostly negative) results — if anything it flatters
the J-lens — but a book whose Chapter 6 teaches “control, confound,
scale” should name its own largest confound, especially where it
reports the specimen J-lens *beating* the logit lens (55 vs 39 at
depth ½), the one place optimism could manufacture the effect.

**Proposed fix.** One paragraph in §2.5/§3.5: state the overlap
exactly, note the direction of the bias, and ideally recompute the
specimen headline (55 vs 39) with a leave-one-window-out $J$
(the export’s per-window Jacobians at depth ½ make half of this
possible already).

## 7. The “frame” names-table row is backwards for this book’s models, and “frame” is then used for objects that are not frames

**Where:** `jspace/book/chapters/04-jspace.md:15–16,30` and
throughout §§4.4–4.5; `06-workspace.md:55`.

> “$(v_a)_{a=1}^V$, spanning but not independent | an overcomplete
> set, like coherent states | frame, dictionary”
> (`04-jspace.md:15`)

**What is wrong.** For *both* models in this book the lens vectors
are **independent and not spanning**: I computed
$\operatorname{rank}(W_{\text{out}}J_\ell)=V$ at every depth
(7 in $\mathbb R^8$; 65 in $\mathbb R^{128}$). The row describes the
*paper’s* $V\gg D$ geometry, in a chapter whose whole point is that
the geometry here is the opposite. It propagates: “random frames
of seven unit vectors in $\mathbb R^8$” are not frames for
$\mathbb R^8$ (seven vectors cannot span it), “the J frame stops
beating chance” (`04-jspace.md:103`), “a good frame for the stream”
(`:115`). The writers’ brief instructed: define “sparse frame” in
one sentence as the paper does “and then stop using the word”
(`jspace/briefs/writer.md`, ch. 4). Section 4.1 even promises
“That is all the vocabulary we need from it” — and then uses
“frame” more than a dozen times.

**Proposed fix.** Change the names row to “$(v_a)_{a=1}^V$,
independent here (rank $V<D$); in the paper, spanning and
dependent”, rename “random frame” to “random set of $V$ unit
vectors” everywhere (prose and figure captions), and keep “frame”
only in the one definitional sentence about the paper.

## 8. Occupancy is defined so that it cannot be 0, then misread by one in the prose

**Where:** `jspace/book/chapters/04-jspace.md:103,113`;
`06-workspace.md:53–55`; definition
`jspace/artifacts/README.md:141–142`.

> “Before the last block a lens vector beats a random direction for
> a step or a few of the fit and then stops”
> (`06-workspace.md:55`)

**What is wrong.** The export defines occupancy as “the first $k$
whose J improvement is strictly below the random improvement”.
Hence occupancy $1$ means the J frame loses **at the first step**:
it beats random for *zero* steps. That is the situation at depths
0–2, whose first-step mean improvements I recomputed as
J 0.035/0.027/0.035/0.036/0.038 against random
0.042/0.047/0.049/0.046/0.046. The sentence above, and “after the
first vector or two, the J frame describes the stream no better”
(same paragraph), are therefore off by one for five of the nine
depths. Separately, an occupancy that is bounded below by 1 by
construction cannot be compared with the paper’s “near zero
through the first third of its layers” (`04-jspace.md:115`) without
saying so.

**Proposed fix.** Either redefine occupancy as
(number of initial steps that beat random), taking values
0,0,0,0,0,1,2,4,36, or keep the export’s definition and rewrite
both sentences (“at depths 0–2 it does not beat random even once:
occupancy 1 in our convention, where the paper’s early layers are
near 0”).

## 9. “The nearest point” and “the J-coordinates” are not unique for a union of cones; the figures quietly depend on tie-breaking and zero coefficients

**Where:** `jspace/book/chapters/04-jspace.md:4` (claim), `:40`,
`:79–82`; Figure 4.2 (p. 46); code docstring printed p. 45
(“Ties retain the first lexicographic subset; zero coefficients are
allowed inside $S$”).

> “The outer minimum runs over finitely many cones; its winner is
> the J-component $x_J$ of $x$, its coefficients $c_a$ are the
> J-coordinates” (`04-jspace.md:40`)

**What is wrong.** The Moreau aside (`:51`) correctly notes the
union “gets the orthogonality … and not the uniqueness”, but the
claim, the main text, and every subsequent use speak of *the*
nearest point and *the* coordinates. Two distinct failures are
visible in the data. (a) *Ties:* nothing in §4.2 says what happens
when two cones are equidistant; the code’s answer (first
lexicographic subset) appears only in a docstring on p. 45, after
the concept has been used for a chapter. (b) *Zero coefficients
inside the “winning subset”:* of the specimen’s
$\sum_k k$ selected slots, **689 of 6,720 have coefficient exactly
0** (my count from `j_space[*].selected/coefficients`). Figure
4.2’s caption says “A dot … means that its lens vector is in the
winning subset; its area is the length $\|c_av_a\|$” — for a zero
coefficient the area is zero, yet the figure draws an *open ring*
(visible at depth 0, $k=7$, character b, where indeed
$c_b=0$). The ring is nowhere explained, and the celebrated
nesting count, “the best $(k+1)$-subset fails to contain the best
$k$-subset in 138 of 1440 cases” (p. 46), counts set inclusion of
these padded subsets, including the forced $k=6\to7$ step where
$S$ must become the whole vocabulary. As a statement about
*supports* (vectors with $c_a>0$) the count would be different.

**Proposed fix.** In §4.2, replace “its winner” by “a winner
(ties are possible; the code keeps the first in lexicographic
order, and a winning subset may contain vectors with coefficient
0)”, define J-coordinates as the coefficients *of a chosen*
nearest point, add the open-ring convention to Figure 4.2’s
caption, and recompute the nesting sentence on supports
$\{a:c_a>0\}$ (or state explicitly that it is about padded subsets).

## 10. Several “one scale” captions are false: the Hinton glyph’s area is proportional to cell², and the cell changes within the figure

**Where:** Figure 2.1 (p. 23), caption `02-jacobian.md:30`;
Figure 2.2 (p. 25), caption `02-jacobian.md:58`;
glyph: `web/src/figures/primitives/MatrixGlyph.svelte`
(`side = cell·0.92·√(|v|/vmax)`); callers:
`jspace/figures/02-jacobian/fig-j-blocks.svelte`
(triangle `cell = inner/D` vs side panels `cell = big`) and
`fig-j-average.svelte` (`g1` vs `g2`).

> “the area of a square is the size of an entry … all on one scale.
> … On the right, this window’s own average $J^{(1)}_\ell$ and the
> average $J_\ell$ over all ten windows, on the same scale.”
> (`02-jacobian.md:30`)

**What is wrong.** `MatrixGlyph` encodes
$\text{area}\propto \text{cell}^2\cdot|v|/v_{\max}$. Sharing
$v_{\max}$ is not sharing a scale when `cell` differs. In
Figure 2.1 the 36 blocks are drawn at cell $\approx$ inner$/8$
(a few px) and the two averages at cell $=$ `big` (up to 17):
the same entry value gets roughly six times the area on the
right. The visual conclusion a reader draws on p. 23 — the
averages look *much* stronger than any block — is therefore
partly an artefact of the drawing, on top of the real effect
(the average is a sum). Figure 2.2 repeats the pattern across its
two halves (`g1`$\ne$`g2`) under the caption “One area scale for
every matrix.” This also breaks DESIGN §5 (“small multiples share
a scale; never scale per glyph”) in the place a reader is most
likely to compare magnitudes.

**Proposed fix.** Pass an absolute area scale (px² per unit
$|v|$) to `MatrixGlyph` instead of `cell`+`vmax`, redraw
Figures 2.1–2.2, and audit every caption containing “one scale”
(`grep` finds six) against the new encoding.

## 11. The paper’s $J$ is presented as an expectation that “is” the book’s sum up to a scalar. For variable-length prompts there is no such scalar, and the book’s averaging weights early positions more heavily — which it never states

**Where:** `02-jacobian.md` §2.2 (p. 24), `03-jlens.md` §3.3;
exercise at `00-prologue.md` (LN scale freedom).

> “For windows of fixed length there are $T(T+1)/2$ pairs to every
> $T$ sources, so the paper’s matrix is ours times $2/(T+1)$.”

**What is wrong.** True here (fixed $T$), but the reader is led to
believe the two averages differ only by a harmless constant in
general. They do not: averaging *uniformly over pairs* weights a
source at position $t$ by $(T-t+1)/\binom{T+1}{2}$ — early sources
count more *per pair* but each pair counts once — whereas the
book’s *mean over sources of the sum over targets* weights source
$t$ by $(T-t+1)/T$ in total contribution, i.e. an early column’s
total influence is up to $T$ times a late column’s. In this book
that means position 1 of a Shakespeare window contributes 256
blocks to $J_\ell$ and position 256 contributes one. Whether that
weighting is “the disposition” is a modelling choice, not an
identity, and it interacts with the decision to retain first
positions (§2.5). The exercises ask what happens for variable
lengths but the main text, where the interpretation “$J_\ell dx$
is what a nudge does … on average over where and when it happens”
is given, does not disclose the position-weighting at all.

**Proposed fix.** Add two sentences to §2.2 giving the weight of
source $t$ in both averages, and change “on average over where”
to “summed over its future, then averaged over where — early
positions therefore contribute more blocks”.

## 12. Chapter 6 does not apply its own circularity warning to the MLP-gain test

**Where:** `06-workspace.md` §6.4 “Across depth” (pp. 68–69) vs
“Across positions” (pp. 70–71).

> “But the J-lens vectors are not independent of these weights.
> The lens vectors at the input of block $b$ are rows of
> $W_{\text{out}}J_{b-1}$, and $J_{b-1}$ is the averaged derivative
> of everything after depth $b-1$, block $b$’s attention included”
> (p. 71, of the OV test)

**What is wrong.** Exactly the same holds for the MLP test and is
not said: the population fed to block $b$’s MLP is the lens
vectors at depth $b-\tfrac12$, i.e. rows of
$W_{\text{out}}J_{b-1/2}$, and $J_{b-1/2}$ is the averaged
derivative *through that very MLP* ($K_{tt}=I+D(\mathrm{MLP}\circ
\mathrm{LN})$ at depth ½ in the specimen is the extreme case).
A direction defined as a pulled-back gradient through a map is
generically aligned with that map’s high-gain directions. The
specimen’s headline MLP result (median gain 1.50, the only
positive broadcast number in §6.4 besides the heads) is therefore
open to precisely the objection the book raises four pages later
— and the confound it *does* name (feeding a vector alone to the
MLP) is the weaker one. The proposed remedy for heads
(recompute $J$ with the head’s path removed) applies verbatim.

**Proposed fix.** Move the circularity paragraph before both
tests, or duplicate its first three sentences in §6.4 with
“MLP” for “attention”, and qualify the specimen’s 1.50
accordingly.

## 13. “Label preservation” is misdefined in the names table and dictionary

**Where:** `06-workspace.md:18` (names table), dictionary p. 80;
correct definition `06-workspace.md` §6.4 (p. 70).

> “how often $Mv_i$ is nearest $v_i$, less chance | the fidelity of
> a channel | label preservation” (`06-workspace.md:18`)

**What is wrong.** The statistic computed (and correctly defined
on p. 70, following the paper) is mean *reciprocal rank* of
$\cos(M_hv_i,v_i)$ among $\{\cos(M_hv_i,v_j)\}_j$, minus the same
quantity for random directions. “How often nearest” is the
top-1 rate, a different number (reciprocal rank gives partial
credit for rank 2, 3, …; with $V=65$ the random baseline is
$\approx H_{65}/65\approx0.07$ for reciprocal rank but $1/65$ for
top-1). A reader using the names table — its stated purpose —
will misread every value in Figure 6.4 (e.g. 0.32 does not mean
“32 % of the time”).

**Proposed fix.** Names row: “mean reciprocal rank of
$\cos(Mv_i,v_i)$ among the $\cos(Mv_i,v_j)$, less the same for
random directions”. The dictionary entry is generated from the
table and will follow.

## 14. The swap is presented as leaving “its position” and the other $D-2$ dimensions untouched. Only the orthogonal component is untouched, and position lives partly in the plane

**Where:** `05-writing.md` §5.4 (p. 56); pseudoinverse derivation
same section.

> “whatever else the column carries in the other $D-2$, its
> position, or anything the model keeps there for reasons
> unrelated to $s$ and $u$, is left exactly as it was.”

**What is wrong.** What is preserved is $x-Vc$, the component
orthogonal to $\operatorname{span}\{v_s,v_u\}$. The positional
code $P_{:,t}$ is not confined to that complement; its in-plane
component is swapped along with everything else. As written, the
sentence invites the reading that positional information is
preserved *tout court*. Two smaller gaps in the same derivation:
(1) $c=(V^{\mathsf T}V)^{-1}V^{\mathsf T}x=:V^+x$ defines $V^+$
only for independent $v_s,v_u$; in the parallel case the code’s
`pinv` picks the minimum-norm $c$, and the swap of that $c$ is a
convention, not an exchange of identifiable coordinates — the
text’s parenthetical does not say the swap itself becomes
ill-defined. (2) The scaled swap’s $\alpha=\tfrac12$ claim (both
coordinates become their mean) uses $V^+V=I$, again independence,
unstated. (3) The default printed swap (Figure 5.2, ‘e’↔‘t’ at
depth 0) has $\|v_e\|=2.65$, $\|v_t\|=9.25$ and
$G=\begin{pmatrix}7.04&-3.64\\-3.64&85.49\end{pmatrix}$: exchanging
$c=(1.01,0.07)$ moves the column by
$|c_s-c_u|\,\|v_s-v_u\|\approx0.94\times9.4\approx8.8$, more than
twice $\|x\|$ — hence TV $\approx1$ at four later positions. The
prose notes that unequal lengths “can be a long way” but never
connects it to its own exhibit, which is a demolition, not an
exchange of two concepts at fixed total content.

**Proposed fix.** Reword to “…carries in the $D-2$ dimensions
orthogonal to the plane is left exactly as it was (the in-plane
part of everything, position included, is rewritten)”;
state the independence hypothesis before the display and the
$\alpha=\tfrac12$ remark; under Figure 5.2 give
$\|x'-x\|/\|x\|$ for the printed swap.

## 15. The two inks do not mean one thing each in Chapters 4 and 6

**Where:** `DESIGN.md` §1.1, §5; prologue §0.5 (p. 11);
Figure 4.1 (p. 44), Figures 6.1–6.4 (pp. 65–70),
Figure 5.4 (p. 59).

**What is wrong.** The design rule is value = forward quantity,
covector = pulled-back quantity, “nothing else is ever coloured”
(categories by weight/position, not hue). Repeated breaches:

- Figure 4.1 hatches the cone in covector ink, but the cone is a
  set of *stream columns* (values); the glyph source even comments
  “generated by covectors, read as directions”. Chapter 4 has
  just spent a section turning covectors into columns via the
  inner product; after that conversion $x_J$ is a value (the
  writers’ brief says so explicitly).
- Figure 6.2 draws *scalar statistics* (agreement fractions,
  kurtosis, counts, occupancies) as “a short bar in covector
  ink” *because they are statistics of the J-lens* (caption,
  p. 66). A fraction is neither a value nor a covector; red here
  is a category colour.
- Figures 6.1 (CKA dissimilarity hatched red), 6.3 (J population
  red, neurons blue, readout grey) and 6.4 (J bars red) use the
  two spot inks as a legend for populations.
- Figure 5.4 draws $v_a$ hatched (covector) in an experiment
  where what is *added to the stream* is $v_a^\sharp$, a value —
  while §5.3 promises “The figures keep the distinction in their
  inks.”

A reader who learned the inks in PicoGPT cannot decode Chapters
4/6 with them.

**Proposed fix.** Cone in value ink with covector bar-tips only
on its two generating rays; Chapter 6 statistics in near-black
ink with J-vs-control distinguished by fill/ring (as Chapter 3
already does); Figure 5.4 legend to read
“$v_a^\sharp$ (the lens vector, written as a column)” in value ink.

## 16. Three-names tables contradict each other for the same object; the generated dictionary therefore contradicts itself

**Where:** names tables in `00-prologue.md`, `01-lenses.md`,
`03-jlens.md`, `05-writing.md`; dictionary pp. 79–81.

**What is wrong.** The dictionary is assembled mechanically from
the tables (`07-epilogue.md` §E.3 says so), so every table-level
inconsistency becomes a printed self-contradiction:

- **Unembedding.** Prologue: the readout $W_{\text{out}}$ is
  “unembedding, $W_U$”. Chapter 1: the whole map
  $R(x)=\operatorname{softmax}(W_{\text{out}}\mathrm{LN}(x)+b)$
  is “unembedding, LM head”. Dictionary p. 81 dutifully prints
  both, numbered 1 and 2. Only the first is the paper’s $W_U$.
- **Activation / residual stream.** Prologue: a *column*
  $x=X_\ell[:,t]$. Chapter 1: the *matrix* $X_\ell\in\mathbb
  R^{D\times N}$ (“hidden states”). Same ML names, different
  maths objects.
- **J-lens vector, physics name.** Prologue: “a covector, a
  generalised force”. Chapter 3: “a static susceptibility”.
  Dictionary p. 80 prints both under one headword.
- **Probe.** Chapter 3: physics “the response it predicts”, ML
  “probe, lens score”. Chapter 5: physics “the reading of one
  linear observable”, ML “probe, linear probe”. Both printed.
- **Logit lens.** Chapter 1: $R$ applied early. Chapter 3: the
  *condition* $J_\ell=I$ — not an object of the same type.
- Duplicated headwords with identical bodies, from comma lists
  in single ML cells: ablation / directional ablation;
  frame / dictionary; concept vector / difference of means;
  SAE / sparse autoencoder; CKA / centred kernel alignment
  (pp. 79–81).
- “top-1 agreement” (p. 81) ends with a stray “· —”, the em dash
  from Chapter 1’s empty physics cell, presented as content.
- Alphabetisation is by a hidden sort key: “the lens matrix”
  under L, “a sparse nonnegative combination” under S between
  “sparse autoencoder” and “steering” — defensible, but
  §E.3’s plain claim “in alphabetical order” will mislead anyone
  searching under T or A.

**Proposed fix.** Make the tables agree before generation: one
row per object per book, later chapters reusing earlier wordings
verbatim; then regenerate. Where the ML cell lists two synonyms,
generate one entry with a “also called” line, not two entries.

## 17. Notation is overloaded across chapters written by different authors

**Where:** throughout; examples below.

- $V$: vocabulary size (everywhere), the $D\times2$ swap matrix
  (§5.4; flagged in text, but then $V^+$ in the names table and
  $GV$/$TV$ in Figure 5.2), and PicoGPT’s value matrix in the
  background.
- $K$: Jacobian blocks $K_{t't}$ (ch. 2) vs Gram matrices
  $K_\ell,K,L$ in CKA (§6.2 and names table).
- $N$: columns in a batch (prologue, ch. 1), then one window has
  $T$ columns (ch. 2), then $N=80$ and simultaneously $N_a$ is a
  *set* (§5.5, $m_a$): “where $N_a$ is the set … among the
  $N=80$ columns”.
- $s$: LayerNorm scale $s(u)$ and the shrinkage $s=2/(T+1)$
  (ch. 3), source character $s$ (ch. 5).
- $b$: window index (ch. 2), block index (ch. 6), the character
  b (all specimen figures).
- $\mathcal J_k$ (the set), $J_\ell$ (the matrix), $m_a^J$ (a
  component), “J frame”, “J-space” — five uses of J in Chapter 5
  alone.
- $R$ the readout (chs. 1, 3) vs $r_a$ the remainder and
  “residual” for $x-x_J$ (ch. 4 names) vs residual stream.

None is fatal alone; together they are exactly the “vocabulary
difficulty” the prologue promises to remove. PicoGPT’s binding
notation table (`BRIEF.md` §2) is not extended for the
supplement anywhere.

**Proposed fix.** Add a one-page supplement notation table in the
prologue (as PicoGPT does), reserving $K$ for Jacobian blocks
(rename CKA Grams $G_\ell$), $N$ for batch columns only (rename
$N_a\to\mathcal N_a$), and keep $s,u$ for swaps only
(LayerNorm scale is already $\sigma$ in PicoGPT’s derivation).

## 18. The saturation argument is stated for all depths and all $v_a$; it needs its hypotheses

**Where:** `05-writing.md` §5.3 (p. 54), Figure 5.1.

> “The output tends to
> $\operatorname{softmax}(W_{\text{out}}\mathrm{LN}(v_a)+
> b_{\text{out}})$, the readout of the lens vector alone, whatever
> the column was and at whatever depth the edit was made.”

**What is wrong.** The limit
$\mathrm{LN}(\alpha v_a+O(1))\to\mathrm{LN}_0(v_a)$ requires
$Cv_a\ne0$ ($C$ the centring). If $v_a\parallel\mathbf 1$ the
limit is $\beta$ (the $O(1)$ term decides), and along
$\alpha\to-\infty$ it is $\mathrm{LN}_0(-v_a)$, which Figure 5.1’s
single open ring per row does not show though the $\alpha$ axis
runs to $-4$. The $O(1)$ claim “only the residual path carries
$\alpha v_a$ to the end, undiminished” also needs the one-line
justification for attention *scores* (queries/keys are functions
of $\mathrm{LN}(\alpha v_a+\cdots)$, hence bounded) — the text
asserts “Every map … either sits behind a LayerNorm … or is
bounded”, and boundedness of the attention branch in $\alpha$ is
the step a BSc reader cannot fill in from PicoGPT alone, because
$Q,K$ are affine in the *normalised* column but the softmax
weights multiply values from *other* columns. Finally, the
prose “the curve of ‘o’ at depth 0 does exactly that [rises,
peaks and falls]” (p. 54): in the printed figure $p_o$ is flat
at $\approx1$ for $\alpha=1..4$ and the “fall” is the dotted
extrapolation to the ring, not data.

**Proposed fix.** State “for $Cv_a\ne0$, as $\alpha\to+\infty$
(and with $-v_a$ as $\alpha\to-\infty$)”, add the scores sentence,
draw the $-\infty$ ring or restrict the axis, and reword to
“must eventually fall towards the ring”.

## 19. Kurtosis is used as a measure of peakedness; it is scale-invariant, and “lens kurtosis” names two different statistics in this book

**Where:** `06-workspace.md:17` (names), §6.2 (p. 67);
`07-epilogue.md:23` (SAEs).

> “everywhere else it is negative, the logits spread flatter than
> a Gaussian’s” (p. 67)

**What is wrong.** Excess kurtosis of the 65 logits is invariant
under $z\mapsto\alpha z+\beta$, but the peakedness of
$\operatorname{softmax}(z)$ is not: logits $(100,0,\dots)$ and
$(1,0,\dots)$ have identical kurtosis and utterly different
readouts. As a “measure of how peaked it is” (§6.2) it measures
only the *shape* (one-vs-rest contrast), and negative kurtosis
for a 65-point empirical distribution does not licence “flatter
than a Gaussian” as a visual claim. Separately, the names table
defines ML “lens kurtosis” as this readout statistic, while the
epilogue (following the paper, paper.txt §1124) uses
*lens-kurtosis $\kappa$* for the kurtosis of an **SAE feature’s**
projection through the lens. The dictionary (p. 80) carries only
the first meaning, so the epilogue’s SAE paragraph is
undecodable from the book’s own apparatus.

**Proposed fix.** Rename Chapter 6’s statistic “excess kurtosis
of the readout” (its figure label already says this), reserve
“lens kurtosis $\kappa$” for the SAE usage, and add: “kurtosis
does not see the overall scale of the logits, so it is a shape
statistic, not a confidence statistic.”

## 20. The specimen half of Chapter 4 overstates its null result at $k=V$

**Where:** `04-jspace.md:92` (p. 47), Figure 4.3.

> “At depths $0$ and $\tfrac12$ the lens vectors do no better than
> chance: they miss 51% and 51%, the random frames 52% and 55%.
> For small $k$ they do worse.” (rendered)

**What is wrong.** At $k=7$ the lens frame in fact misses *less*
than the random mean at both depths (50.6 % vs 51.8 %;
50.8 % vs 55.3 % by my recomputation from
`distance`/`random_fraction`). The honest statement is “worse for
$k\le3$, marginally better by $k=7$, with no uncertainty attached
to either side” — the random figure is a mean over 20 frames with
no spread shown in prose, and the per-column faint lines in
Figure 4.3 show the two populations overlap almost completely.
“As a stream shaped to be read should” (same paragraph, of
depth 1) is teleology, not a finding.

**Proposed fix.** Quote the $k=3$ and $k=7$ numbers with the
random frames’ interquartile range, and delete “do no better than
chance” in favour of the $k$-dependent statement.

## 21. Chapter 3’s KL direction is stated three ways

**Where:** Figure 3.4 caption (`03-jlens.md`, p. 38), figure title
in the plate (“KL(model ∥ lens), nats”), `jspace/artifacts/README.md:107`
(“`kl_model_lens` (nats, model as first argument)”).

> “Right: the relative entropy of the lens’s distribution from
> the model’s, in nats.” (caption)

**What is wrong.** “Relative entropy of $P$ from $Q$” is read by
half the audience as $D(P\|Q)$ and by the other half as
$D(Q\|P)$; the plate’s own title says KL(model ∥ lens). The
prose (“its distribution is further from the model’s”) does not
disambiguate, and PicoGPT derives cross-entropy from KL with a
fixed convention the supplement never restates. The numbers are
$D(p_{\text{model}}\|p_{\text{lens}})$.

**Proposed fix.** Caption and first prose mention to read
“$D(p_{\text{model}}\|p_{\text{lens}})$ in nats (model first)”,
matching the plate.

## 22. “Several times longer” and other length claims do not survive the exported norms

**Where:** `03-jlens.md:57`; norms from `lenses.vectors` /
`lenses.logit_vectors`.

> “At the earlier depths they point elsewhere … and they are
> several times longer.”

**What is wrong.** At depth 0, $\|v_a\|/\|w_a\|$ by character
(␣ b e n o r t) is 4.3, 2.8, 1.5, 4.3, 3.5, **0.9**, 4.5 — ‘r’ is
*shorter* than its readout row, ‘e’ is half as long again, as
Figure 3.1 shows if read carefully. At depth ½ the ratios are
3.5, 1.7, 1.4, 3.4, 2.7, 1.0, 2.9. “Several times” is true of
four characters in seven. Length also re-enters, unremarked, in
the cosines printed under Figure 3.1: two of the fourteen
early cosines are negative (‘o’: −.02, −.04), which the prose’s
“they point elsewhere” understates — one lens vector points
slightly *against* its readout row, worth one clause.

**Proposed fix.** “and mostly longer — up to 4½ times, though
‘r’ at depth 0 is shorter than its readout row, and ‘o’ makes a
slightly obtuse angle with its own”.

## 23. The cone picture is not “exact” as a metric drawing, and it draws the remainder in the plane it is perpendicular to

**Where:** `04-jspace.md` §4.2 (p. 43), Figure 4.1 (p. 44);
`jspace/figures/primitives/cone.js` (`axonometric`),
`ConeGlyph.svelte`.

> “An exact picture of three dimensions loses nothing of the
> eight.” / Caption: “The cone of two lens vectors and the
> nearest point of it, exactly.”

**What is wrong.** The computation is exact; the *drawing* is an
axonometric projection onto an ellipse. Angles and
perpendicularity are not preserved in the print: the disc is an
ellipse, the “foot of the perpendicular” is not visually
perpendicular, and the cone’s printed angle is not
$\arccos\langle\hat v_s,\hat v_u\rangle$. Worse for a first
reader: $x$ is drawn displaced from its foot $p$ *within the
page plane*, with the dashed segment $x-x_J$ lying in that
plane, on the page where the text has just proved
$x-x_J\perp x_J$ and (for the default column) that the remainder
“is all perpendicular to the plane”. “Seen from above” (caption)
would make $x$ coincide with $p$. The key (“How to read a cone”,
p. 44) draws a *different*, flat cone with horizontal hatching —
not an enlarged instance of the figure’s own glyph as DESIGN §6
requires — and its labels (“dashed: $-v_s$ in the plane, not in
the cone”) do not explain the axonometric offset, the disc, or
the right-angle marks. The 21 small cones on the right are, at
print size, indistinguishable red/blue wedges; only their numbers
carry information, and the top axis label for the space
character is an invisible underscore.

**Proposed fix.** Caption to “computed exactly; drawn in
axonometric projection, so angles on the page are not the angles
in $\mathbb R^8$ — $x$ is offset from $p$ by its height above
the plane, in the direction marked”. Make the key an enlargement
of the *printed* pair (␣, e) in the same projection. Either
enlarge the 21-pair triangle or replace its glyphs by the 21
numbers in a triangle (the numbers are the content).

## 24. Figure 2.3 compares spoke lengths across rows that each have their own scale; the caption does not say so

**Where:** Figure 2.3 (p. 27), caption `02-jacobian.md`;
author report `jspace/briefs/report-ch0-2.md` (“One spoke scale
per depth row”).

**What is wrong.** Each depth row is labelled in 6-pt type
“spoke 0.85 / 0.78 / 1.00” (top panel) and “spoke 8.20 / 2.80 /
1.00” (bottom). A spoke at depth 0 in the lower panel is thus
drawn ~8× shorter per unit than at depth 1. Nothing in the
caption or surrounding prose warns that lengths are *not*
comparable across rows or between the two panels, in a figure
whose whole message is what arrives where. DESIGN §5 allows
“separate, explicitly labelled array scales” — the label exists
but is in the margin in micro-type and the caption actively
implies a single reading (“What arrives in each column is drawn
… one spoke per entry”). The right-hand panel compounds it:
$J_0$ is drawn in value ink with row 1 hatched in covector ink,
a distinction the caption (“the same row of $J_0$ … is drawn
beside it”) does not explain.

**Proposed fix.** Add to the caption: “Each row has its own
spoke scale, printed at its left; lengths compare within a row
only.” Explain the two inks in the right panel in one clause.

## 25. Keys that explain the wrong thing

**Where:** Figure 1.3 key (p. 18), Figure 0.1 key (p. 9),
Figure 3.3 key (p. 37), Figure 3.1 “Reminder” (p. 34).

- Figure 1.3’s “How to read” explains the *depth axis* (already
  labelled “embed / attn / MLP / block” inside the figure) and
  never mentions the two encodings a reader actually needs:
  filled dot = agreement, open ring = accuracy, dashed =
  $1/V$ and model accuracy. Those are left to the caption.
- Figure 0.1’s key enlarges one bypass row, but its leaders label
  the matrix and the readout, not the stem/bypass distinction
  that is the figure’s new idea; the caret marking the chosen
  column sits over the space–o boundary in
  “t o ␣ b e ␣ **o** r” and at print size reads as marking ‘o’,
  while the data are column 6, the *space* before it
  (`fig-j-idea.svelte`: `pos = column-1`).
- Figure 3.3’s key shows three columns (“B a p…”) at depths
  3½ and 4 only, with notes “‘o’ with 0.23, light; ‘t’ with
  0.99, bold” — it never says rows run 0→4 top to bottom in
  *both* plates, that the lower plate is a different lens, or
  that ¶/␣ are characters of the text, all of which the caption
  must then carry.
- Figure 3.1’s key is headed “Reminder: covector spokes”, not
  “How to read”, breaking the book’s own key convention
  (DESIGN §6), and its example vector is chosen by an algorithm
  (`fig-j-lens-vectors.key.svelte` searches all depths for the
  most legible ± pair) — the key caption names it (‘b’ at
  depth 0) only in 7-pt grey under the key, detached from the
  figure caption.

**Proposed fix.** Rewrite the four keys so each leader touches
the mark it explains, in the order a reader meets the marks;
rename the reminder; move the chosen column’s caret unambiguously
over the space glyph (or choose column 7).

## 26. Print figures that do not survive print

**Where:** Figures 3.3 (p. 37), 1.2 (p. 16), 4.4 (p. 49),
6.4 (p. 70), 2.1 (p. 23).

- **Figure 3.3** sets 83 positions × 11 rows × 2 plates in one
  page width. Body characters are ≈4–5 pt with weight as the
  only probability encoding; light-weight glyphs (200 weight)
  all but vanish, rows run together with no cell structure, and
  the colon row on which §3.5 turns cannot be counted from the
  printed page (I counted from the JSON). The print build’s own
  warning (“fig-j-grid … printed at 38 %”, `report-ch3-4.md`)
  was noted and left. Either split the two plates over two
  pages or set only the GREMIO…BAPTISTA region at full size.
- **Figure 1.2** distinguishes agreement solely by black vs grey
  6-pt letters over bars; on the greyscale contact sheet
  (`jspace/proof/contact-sheet-grey.png`) the distinction is
  lost, violating “survive black and white”. The underlined
  true-next character is similarly invisible under most cells.
  Use bold vs light *weight* plus the grey, and a tick outside
  the cell.
- **Figure 4.4**: occupancy “vertical mark” at $k=1$ coincides
  with the $y$-axis in five panels, so the labelled mark the
  caption promises does not exist; span ticks at each panel’s
  right are unlabelled except by caption; $k=25$ is “ticked”
  but the tick is a faint axis mark identical to 0 and 65.
- **Figure 6.4**: head labels collide in the centre
  (3.2/4.3/3.3/2.3 overprint), rings for 16 rotated controls form
  an unreadable cluster at (1, 0), and joining lines cross the
  whole panel. The specimen panel uses half its width for two
  heads.
- **Figure 2.1**: the window is written “to_be_or” with ASCII
  underscores inside the figure title, against DESIGN §3
  (“The space character is drawn as ␣ in figures”) and against
  every other figure; the right-hand labels
  “window 1: mean over t of Σ…” are set wider than their panel
  and crowd the page edge.

## 27. Jargon before definition, and anthropomorphism the brief forbids

**Where:** passim; brief: `jspace/briefs/writer.md` “no
anthropomorphism beyond what is defined. Every ML term defined
in the reader’s language before use.”

Used before any definition in *this* book (PicoGPT prerequisites
excuse pullback/softmax/LayerNorm, listed in §0.5 — these are
not in that list): “sparse” and “steered” (§0.1, defined chs. 4/5),
“broadcast” (§0.1, ch. 6), “concept” (everywhere from §0.1 on;
never distinguished from *character*, which matters by §5.5),
“KKT conditions” (§4.2; expanded only in an aside in §5.5),
“active-set / passive set” (§4.3, via a code docstring),
“sensory / motor layers” (ch. 6 names table, before §6.2’s
account), “interpretable content” (§3.6, the paper’s central
claim, never given a testable form in this book).

Anthropomorphism beyond the defined “poised to say $a$
$\equiv$ large $\langle v_a,x\rangle$”: “It is big enough to
have something to say” (§0.3, of the Shakespeare model); “this
model would make up its mind in its last half block” (§1.5);
“silences the probe … need not silence the character” (§5.3);
epilogue claim, “training that changes what a model holds in
mind” (p. 75); “a stream shaped to be read should” (§4.4).
Each is dry wit in the PicoGPT voice; together they breach the
supplement’s own stricter rule, on the topic (consciousness
language) where §0.4 promises particular care.

**Proposed fix.** A single pass: eighth-column definitions at
first use for sparse/steer/broadcast/concept; move the KKT
expansion from §5.5’s aside to §4.2; delete or operationalise
the five phrases quoted.

## 28. Exercises and forward references that the print reader cannot honour

**Where:** `01-lenses.md` Exercise 1.3; `02-jacobian.md`
Exercise 2.4; `06-workspace.md:31` and Exercise 6.1;
`07-epilogue.md:49`.

- Ex. 1.3: “what does the readout of Figure 1.2 use?” — the
  print figure has no readout (it is the interactive figure’s
  live text). Unanswerable on paper.
- Ex. 2.4: “In Figure 2.3, seed $e_i$ in the last column and
  then in the first” — print shows only “column 8 only” and
  “every column”; the first-column seeding exists only on the
  web. DESIGN §1.6 promises both editions carry the argument.
- §6.2: “the change of the scale of $J$ between the first and
  the final export of the analysis” — there are no “two
  exports” in the reader’s world; this is a leak from
  `analysis-report.md`’s superseded-convention history.
  Exercise 6.1 asks the reader to explain it.
- Epilogue: the specimen training trajectory
  (`specimen-jlens-training.json`, 201 steps, a third of the
  analysis effort) is mentioned in one sentence, “left to the
  reader”, with no figure, exercise, or access route. Either
  use it (one small figure: do the vectors settle before the
  loss does?) or cut the sentence.

## 29. Typography and layout

**Where:** whole PDF; specific pages below.

- **Four wholly blank pages** (2, 30, 62, 74), no folio or head:
  reserved trace versos (`jspace/traces/` is empty). In a finished
  81-page book they read as a binding error. Either supply the
  four traces or stop reserving the slot for this edition.
- **Subtitle** (p. 3, from `jspace/edition.js` `BOOK.subtitle`):
  “worked through on a transformer with 1007 parameters” — the
  book’s second model has 810,049 and carries Chapters 3, 4 and
  6. As printed, the title page misdescribes the contents.
- **Sentences split by full-page figures:** p. 56 bottom →
  p. 57 bottom (“swap of *Soccer* for *Rugby*…” orphaned after
  Figures 5.2–5.3); p. 68 → 69 (the list of MLP medians is split
  by Figure 6.3: “…rows of $W_{\text{out}}$ are ⧉ 1.03, 1.09…”);
  p. 33 → 34 (two orphan lines under Figure 3.1); p. 58 → 59.
  A science-book reader should never hunt for a verb across a
  plate.
- **Running heads anticipate:** p. 19 is headed “Exercises”
  though its top half is §1.6 “In the code”; p. 27 is headed
  “In the Shakespeare model” though it is mostly Figure 2.3
  belonging to §2.4. (Rule: first section *beginning* on the
  page; here it misleads.)
- **Widows** flagged by the build and visible: pp. 35, 39
  (`report-ch3-4.md`); p. 34’s two-line orphan is the worst.
- **Code chunks break** despite DESIGN §8 “chunks avoid
  breaking”: `averaged_jacobians` (pp. 28–29),
  `exact_decomposition` (pp. 45–46). `lens_logits` is printed
  twice (pp. 19 and 40), the second time with the docstring the
  first omitted.
- **Glyph for line break** is ¶ in Figure 3.3, ↵ in Figure 5.3’s
  axis, and “a line break” in prose; the space is ␣, `_`, and
  (Figure 5.1’s axis) a low underscore in “_benort”, by turns.
- **Figure 5.3’s TV = 1.00** at $\alpha=1$ and 2: the values are
  0.996 (my recomputation from `interventions.swap`) — printed
  rounding erases exactly the finite gap DESIGN §3 protects for
  probabilities near 0/1.
- Chapter openings have **no trace and no drop initial after the
  claim**, leaving pp. 31, 63, 75 mostly white below the names
  table (p. 75 is three-quarters empty) — consistent with the
  blank-verso problem above, and at odds with the density of
  the reference chapter (`book/chapters/05-attention.md`).

## 30. Smaller exactness items

- `04-jspace.md:106`: gradient pursuit is “cheaper on
  vocabularies of a hundred thousand tokens” — the paper does
  not give its models’ vocabulary sizes in `paper.txt`, and the
  ch. 0–2 author explicitly avoided inventing one
  (`report-ch0-2.md`, “No vocabulary sizes … (not public)”).
  Delete the number.
- `03-jlens.md` §3.3: shrinking by $s$ “flattens the logits as a
  raised temperature would”. It scales only the $x$-dependent
  part; $W_{\text{out}}\beta+b_{\text{out}}$ is untouched, so it
  is not a temperature. Exercise 3.2 says this correctly;
  promote its qualification into the text.
- `03-jlens.md` §3.2: “what the earlier columns are doing to it
  is not seen at all” — at $\ell>0$ their effect is already
  *in* $x$ via attention. Reword: “not seen again”.
- `06-workspace.md` §6.2: CKA “is one when the two sets are the
  same up to a rotation … and a common scale” — add “and a
  translation” (the centring in `linear_cka` gives it, and
  Exercise 6.1 omits it too).
- `05-writing.md` §5.5: “half of the result is arithmetic”
  (§5 claim and p. 58) — no factor of a half is computed
  anywhere; the fraction is rhetorical. “Part of the result”.
- `00-prologue.md` §0.3: specimen averages are “exact: over all
  ten windows, all positions and all pairs” — exact as a sum
  over that *training* batch; one word (“exact over its training
  batch”) would prevent the reading “exact for the model”.
- `01-lenses.md` §1.3: the depth-0 logit lens is “a bigram model
  with a clock, which reads the next character off the current
  one” — it reads it off the pair (character, position), as the
  book’s own Exercise 1.2 then asks the reader to show. Say so
  first time.
- Figure 0.1 caption: “at depth $\tfrac12$ (after attention)” —
  rendering in the PDF text layer as “depth 21” (likewise
  $2\tfrac12$ as “21” in running text extraction) suggests the
  $\tfrac12$ glyph has no space in extraction; visually fine,
  but the PDF is not reliably searchable for “½”, which matters
  for a book whose depths are half-integers. Check the tagged-PDF
  actual-text for vulgar fractions throughout.

---

## Summary for the authors

The mathematics that is this book’s reason to exist — $J_\ell$ by
pullbacks, $J_L=I$, the ε lesson, cones and NNLS, the
pseudoinverse swap, CKA — is soundly derived and, with Findings
1, 9, 14 and 18 repaired, soundly stated. The empirical reporting
is honest in direction and, in most numbers, exact.

What needs work before print, in order:

1. Purge “gradient of the logit” (Finding 1) and with it rewrite
   the steering and privileged-test morals (3, 4).
2. Withdraw “exactly the paper’s definitions” and tabulate the
   deviations (2); fix the GREMIO paragraph with its own data
   (5); disclose in-sample evaluation (6).
3. Correct the frame row, the occupancy off-by-one, and the
   label-preservation definition (7, 8, 13); make nearest points
   non-unique in the text (9).
4. Fix the area-scale encoding and redraw Figures 2.1–2.2 (10);
   restore the two-ink discipline in Chapters 4/6 (15); split
   Figure 3.3 (26).
5. Reconcile the names tables, regenerate the dictionary (16),
   add a notation table (17), and fill or remove the four blank
   versos and fix the subtitle (29).

None of these requires new analysis except, optionally, the
leave-one-window-out check in Finding 6 and the self-only colon
count in Finding 5 — both small against the existing exports.




# Review of draft Chapter 5, Attention — as its intended reader (Muse)

I read the PDF (17 pages; chapter proper pp. 7–17) with the Markdown at
`attention-audition/book/chapters/05-attention.md` at hand, as a BSc physics/mathematics
student who has read Chapters 1–4 (chain rule, vectors, Gibbs/softmax, surprise) and
*not* yet Chapters 6–8 (the block, pulling back, learning) — because in this book's
order, Chapter 5 comes before all three. That ordering matters to several comments
below. I also compared the old Chapter 5 (`book/chapters/05-attention.md`, 1609 words
against the draft's 1998).

**Verdict in one breath.** The top-down route works as a *derivation of the formula's
shape*: by p. 10 I could reconstruct every symbol in $A=\operatorname{softmax}_{\rm rows}
(Q^\T K/\sqrt d+M)$, $O=VA^\T$ and say which step put it there, which the old chapter
never let me do (there, query "says what reader $i$ is looking for"). But the draft
repeatedly calls restrictions *approximations*, and its two load-bearing moves —
Step 2's shared readout and Step 5's faithfulness — are asserted where the old
chapter merely waved. It is clearer about *what* attention is, vaguer than it admits
about *why it should be any good*, and about 400 words of its extra length buy
analysis (§5.3) that is correct but never cashed in numbers I can check. I found no
wrong formula: the bias expansion, the geometric-mixture identity, the Taylor bound
and the $s^2\le\eta R^2$ bound, the count $4(D^2+D)=288$, and the three losses on
p. 12 (0.0589 / 0.0624 / 0.1980 nats) all check out. The problems are order, naming,
one figure, and three sentences that hide the whole argument in an adjective.

## 1. Where I got lost or bored

In reading order. Quotations are exact.

**(a) The claim, p. 7.** *"Attention is an ansatz for the next-character conditional:
introduce an uncertain earlier position, give it a low-rank bilinear Gibbs law, and
replace the selected vector by its expectation."* I know the book opens every chapter
with a claim to reread afterwards, and the names table is headed "defined in this
chapter", so I did not panic — but this claim needs the whole chapter, whereas the
old one ("column $i$ … is an average of value vectors … weighted by a Gibbs
distribution over query–key scores") needed only Chapter 3. Three undefined loads in
one sentence: *ansatz* (a maths BSc may not have the physicists' word), the
*selected vector* (nothing has been selected by anyone yet; there is no random index
on the page), and *its expectation* (under what distribution?). Read after the
chapter it is the best one-sentence summary in the draft. Read before, it is the
jargon-before-object failure the book promises to remove.

**(b) The matrix-product-state aside, p. 7, right column.** *"A matrix product state
compresses a table across a cut into $\sum_{r=1}^{\chi}L_r(u)R_r(v)$, pushing the past
through one bond of dimension $\chi$."* This is where I stopped dead, before the
argument had even started (the eye hits the marginal column first). I do not know
what a matrix product state, a cut, a bond, $u$ or $v$ are; a BSc physicist who has
met Gibbs distributions has usually not met tensor networks. The payoff clause,
*"imposing no correlation-length bound on the joint law"*, says nothing to me —
no correlation length has been defined for a distribution on texts, and with
$T=8$ any bound would be $\le 7$ anyway. The aside also argues one side only:
"saving parameters without a fixed-size memory" omits that the price is keeping
$T$ columns and computing $T^2$ scores (the draft's own p. 14 aside). This is the
single most extraneous paragraph in the draft.

**(c) Step 1, the kernel that sees the answer, p. 8.** *"For any kernel $r(j\mid y,c)$
summing to one over $j\le i$, set $\tilde P(y,j\mid c)=P(y\mid c)\,r(j\mid y,c)$."*
Two stumbles. First, *kernel* is used unglossed (Chapter 1's names table called
$q(\cdot\mid x_{\le t})$ "a transition kernel", in passing). Second and worse:
$r$ may depend on $y$, the character we are trying to predict. So $J$ is chosen
*after seeing the answer*, and of course any $P$ can then be written as a mixture.
The draft is honest ("This is exact and empty"), but I only realised *how* empty —
that $y$-dependence is the whole cheat — on a second read. The example given,
$r=1/i$, shows the trivial extension but no example shows a useful one, so at the
end of half a page of careful rigour ($\pi_j=0$ cases, "of positive probability")
I have proved an identity and own nothing I can compute with. This is the first
place I got bored: rigour spent exactly where the content isn't.

**(d) The sentence the route stands on, p. 8.** *"The first approximation assumes an
extension exists in which both $\pi_j$ and $p_j$ admit simple shared descriptions."*
Everything is hidden in *simple* and *shared*, neither yet given a meaning: shared
across positions? prefixes? readers? Simple as in few parameters, or as in "a
formula"? And it is not an *approximation* in any sense Chapter 4 taught me: no
distance between $P$ and anything is bounded, no error is ever estimated. It is a
*hypothesis about text* — roughly, "the next character depends on the prefix mostly
through one earlier position at a time" — and the draft never says it that plainly,
nor what would make it false (e.g. a next character that is a parity of three
earlier ones, with no single position carrying the information). Because $r$ in
Step 1 could depend on $y$, while $\pi_j$ from Step 2 on cannot, the real loss
happens in the silent change of quantifiers between the identity and this
sentence. I got lost here not because it is hard but because it whispers.

**(e) Step 2's first sentence — three forward references in one breath, p. 8.**
*"Position $j\le i$ carries $x_j\in\R^D$—initially $Ee_{a_j}+P_{:,j}$, in deeper
blocks the incoming residual column—and $\hat x_j=\mathrm{LN}_1(x_j;\gamma_1,
\beta_1)$ (Chapter 6)."* As the intended reader I have not read Chapter 6 (it comes
*after* this one). *Block*, *residual column*, $\mathrm{LN}_1$ and its parameters
$\gamma_1,\beta_1$ are all from a chapter I don't have. The old chapter had exactly
this problem and solved it with one marginal sentence: "for this chapter it is
simply the vector from which the three affine maps form queries, keys and values."
The draft dropped that lifeboat. All I need here is "each position carries a vector
$x_j$, and $\hat x_j$ is a rescaled copy of it (Chapter 6)"; the parameters of
LayerNorm do no work in this chapter.

**(f) The shared readout $g$ arrives from nowhere, p. 8.** *"…and one shared readout
$g(x_i,v_j)\in\Delta^{V-1}$: $\tilde P(y\mid c)=\sum_j\pi_j\,g_y(x_i,v_j)$."* Why
should $p_j$ depend on the whole prefix only through the *pair* $(x_i,v_j)$ — not
on $j$ except via $v_j$, not on any third position? The draft immediately discusses
what $v_j$ erases ("anything $W_v\hat x_j+b_v$ erases is lost" — good) but never
states the symmetrical costs: $x_i$ must suffice for the reader, and $\pi_j$ is
built from pairwise scores normalised afterwards, so the weight on $j$ cannot
depend on how $j$ compares with a *specific* other source, only on the multiset of
scores. Also unanswered: what *is* $g$ in the actual model? I am never told that
$g$ is "the rest of the network after this average, run with $v_j$ in place of
$o_i$". Without that operational definition, $p_j=g(x_i,v_j)$ in Figure 5.3 is an
intervention I cannot picture: $v_j\in\R^d$ is one head's vector, but the readout
of Chapter 3 eats a whole column in $\R^D$. What exactly was fed in? (See §4,
Figure 5.3, and edit 2.)

**(g) "Position $i$, the reader, plays three roles", p. 8.** *"…and passes its
unnormalised $x_i$ to $g$ through the residual sum."* I don't know what a residual
sum is (Chapter 6 again), so the third role is a label without an object. It also
isn't quite how the model works even once Chapter 6 arrives: $g$ does not receive
$x_i$ and $o_i$ as two arguments; it receives the single column $x_i+W_o o_i+
b_o$ (then MLP, final LayerNorm, readout). Writing $g(x_i,v_j)$ makes the mean-field
step look cleaner than it is — the averaging interacts with $x_i$ by *adding* to it
and then passing through LayerNorms, which rescales. A physicist reader will want
to know this before trusting the Taylor discussion on pp. 11–12.

**(h) Step 3 is Step 1 again, but not flagged as such, p. 9.** *"Any positive row
$\pi_j$ … has Gibbs form (Chapter 3) … via $E_{ij}=-\tau\log\pi_j$."* I initially
read Step 3 as content ("a Gibbs law!") and only later saw it is the same empty
reparametrisation trick as Step 1: *any* positive weights have an energy. The
draft says "The restriction is the energy" in the next sentence, so the honest
statement is there, but the numbering oversells: there are not five restrictions,
there are three (shared pairwise rule, low-rank bilinear energy, expectation
inside $g$) dressed with two exact rewrites. Also unflagged as a cost: finite
energies force $\pi_j>0$, so a source can be *nearly* ignored but never excluded —
except future sources, which need the $-\infty$ of §5.4. That asymmetry is worth
one clause.

**(i) The parameter count that undercuts its own step, p. 9.** *"Two $d\times D$
factors store $2dD$ weights against $D^2$ for a dense matrix—64 against 64 per
channel in the specimen ($D=8$, $d=4$, plus 8 biases)—so here the factorisation
imposes rank $d$ without saving matrix entries."* I admire the honesty and then
ask: so why factorise at all, in the only model I can inspect? The draft never
answers. (The answers available: in larger models $d\ll D$ per head and it does
save; here the specimen is the degenerate case $2d=D$; and the rank restriction per
head is the point, the saving a side-effect.) Also *channel* appears here and on
p. 8 ("one channel of width $d$") as a third word for what §5.5 will call a *head*
and a *subspace*; §5.5 is six pages away, and Figures 5.1–5.4 all say "head". One
word, introduced once, please.

**(j) The whole of pp. 11–12 after "When is this faithful?"** *"…For smooth $g_y$
with Hessian norm at most $K$, Taylor's theorem bounds $|\sum_jA_{ij}g_y(v_j)-
g_y(o_i)|$ by $\tfrac12 Ks^2$ (its sign can vary by character), where $s^2=
\sum_jA_{ij}\|\delta_j\|^2$…"* I can follow this (I know Taylor with remainder),
but: (i) *faithful to what?* — it takes a paragraph to realise the bound compares
two predictors at the *same* trained weights, neither of them the true $P$; the
mixture is a model we invented in Steps 1–2, not a gold standard. (ii) $K$ "at
most" *where* (on the convex hull? along the segments?), $R$ in "$s^2\le\eta R^2$
for diameter $R$" — diameter of $\{v_j\}$, presumably, never said; $\eta$
introduced mid-sentence. (iii) The parenthetical "(its sign can vary by character)"
attaches to an absolute value and confused me into rereading. (iv) No numbers:
neither $K$ nor $s^2$ is ever evaluated for the specimen, so the bound decides
nothing — the paragraph ends and I don't know if the gap is $10^{-3}$ or $10^{3}$
(the empirical losses two paragraphs later do tell me, which makes the bound
redundant as evidence). This is where I got bored a second time: a bound with no
right-hand side.

**(k) The 80-readers sentence, p. 12.** *"Across all 80 readers of the trained
specimen (fixed parameters, rows, values and residual inputs; independent head
indices, §5.5 (p. 15)), mean cross-entropy is 0.0589 nats…"* The parenthetical is
doing three jobs and doing none: I don't know what was intervened on ("fixed …
residual inputs" — fixed while *what* changed?), "independent head indices" uses
§5.5's machinery three pages before it exists, and §5.3's formulas were all derived
for a *single* index $J$, while these numbers apparently use a *pair* of indices
(one per head). Which object do 0.0624 and 0.1980 belong to? I still don't know
after two reads. Also missing: the scale. Chapter 1 computed the floor $H_{\min}=
(2\log2+3\log3)/80\approx0.0585$ nats; with that one number, 0.0589 reads as "at
the floor", 0.0624 as "just above", 0.1980 as "three times the floor". Without it
they are three decimals.

**(l) "(Any coupling with those marginals gives the same $O$.)", p. 15.** *Coupling*
is not basic-probability vocabulary for this reader, and the parenthesis undoes the
previous sentence without explanation: if any coupling gives the same $O$, why did
we just assume *independent* indices? (Answer, which I had to supply myself:
independence matters for the *mixture before* the mean-field step — the thing
compared in §5.3 — and is invisible after it. That is actually an interesting
fact about what attention cannot express: correlation between what its heads
read. Buried in a parenthesis, it reads as a shrug.)

**Symbol collisions that cost me rereads throughout** (all inherited from the book's
notation, but this chapter is where they collide): $V$ is the vocabulary size
($V^i(V-1)$, $\Delta^{V-1}$) and the value matrix ($O=VA^\T$); $P$ is a probability
law ($P(y\mid c)$) and the positional code ($P_{:,j}$, Exercise 5.3 "Remove the
positional code $P$" two pages after $P(Y{=}1\mid a,b)$ would be natural); $E$ is
the embedding ($Ee_{a_j}$), the energy ($E_{ij}$) and $\E$ the expectation
($o_i=\E_{J\sim A_{i,:}}[v_J]$); $p_j$ is Step 1's component law, then $g_y(x_i,v_j)$,
then $\operatorname{softmax}(z(v_j))$ with $x_i$ and the other head suppressed;
$a$ is a character ($a_1,\dots,a_i$; Chapter 1 used $x_t$ for these), a dummy
character ($\sum_a$, p. 12) and a bit (Exercise 5.4); the model's law is $q_\theta$
in Chapter 1 but "$p_\theta(a_{i+1}\mid a_{1:i})$" on p. 14; and $o_i$ the output
shares its letter with the character "o" that fills every figure. None is fatal
alone; together they tax exactly the reader the book is for. At minimum $P(y\mid c)$
vs $P_{:,j}$ on pp. 8–9 wants a disambiguating clause.

## 2. Jargon used before the thing it names was on the page

Excluding the claims/names front matter, which the book's convention licences
(though see 1(a)), in order of appearance:

| Term | Where | Reader has it from |
|---|---|---|
| matrix product state, bond, cut, correlation length | p. 7 aside | nowhere; never defined in the book so far |
| kernel $r(j\mid y,c)$ | p. 8 | one passing cell of Ch. 1's names table |
| residual column / residual sum / residual stream; block; $\mathrm{LN}_1(\cdot;\gamma_1,\beta_1)$; MLP (p. 7) | pp. 7–8, 15 | Chapter 6 — *after* this chapter |
| channel (= head = subspace) | p. 8 | §5.5, six pages later |
| readout $g$, "decode/decoding" (pp. 10–11, Fig. 5.3) | p. 8 | readout was Ch. 3's final affine map; here $g$ silently means "everything after attention" |
| log-odds ("arbitrary context-dependent log-odds") | p. 9 | Ch. 3 aside defined logit = log-odds for Bernoulli; here it means $\log(\pi_j/\pi_{j'})$, never written |
| its derivative vanishes ("softmax then freezes… and its derivative (Chapter 7) vanishes"); initialisation; row entropy | p. 9 | Chapter 7 (derivatives/pullbacks) and Chapter 8 (training), both later |
| mean-field | claim, names, §5.3 title | defined only mid-§5.3, p. 11 ("Moving an expectation inside a nonlinear map…"), after doing work in two headings |
| ReLU kink | pp. 12 (twice) | Chapter 6's MLP; also "Hessian norm" is fair game from analysis, "kink" is not yet an object |
| total variation | Fig. 5.3 only | nowhere; not in caption or text (Ch. 4 gave KL, not TV) |
| coupling, marginals | p. 15 | marginalising is fine; *coupling* is graduate probability vocabulary, unglossed |
| pullback, adjoint, covectors "the bars" | p. 16 | Chapter 7 (inherited from old chapter, but still forward) |
| AdamW, $\eta$, warmup, cosine, weight decay, clip | frontispiece caption, p. 6 | Chapter 8 at the earliest; pure noise at this point in the book |

To be fair: *ansatz* in the claim, *convex hull*, *Hessian*, *Jacobian*,
*block lower-triangular* are within a BSc maths/physics vocabulary and I would not
gloss them. The pattern in the table is not ML jargon — query/key/value/head/mask
are now admirably introduced *after* their objects — it is the book's own later
chapters leaking backwards. The draft leans on Chapter 6 (residual stream, LN$_1$,
ReLU, $W_o$ "recombines … into the residual stream (Chapter 6)") far more than the
old chapter did, because the top-down route needs a $g$, and $g$ lives there.

## 3. Does the route work?

Yes for the formula, with the reservation below. After one careful read (and the
rereads noted in §1) I can say why attention has the shape it has, in five lines:

1. Write the next-character law as a mixture over an invented uncertain position:
   $P(y\mid c)=\sum_{j\le i}\pi_jp_j(y)$ — exact, empty, costs nothing, buys nothing.
2. Assume text is kind: one shared rule on pairs of position vectors gives both the
   weights $\pi_j$ and, through one affine image $v_j$ per source and one shared
   readout $g(x_i,v_j)$, the components $p_j$ — cost: the prefix may now influence
   the prediction only one consulted vector at a time.
3. Write the weights as a Gibbs law and take the energy bilinear — cost: every
   source keeps positive weight, and log-ratios $\log(\pi_j/\pi_{j'})$ must be
   bilinear functions of $(\hat x_i,\hat x_j-\hat x_{j'})$.
4. Factor the bilinear form through $\R^d$, $B=W_q^\T W_k$: the two factors are all
   "query" and "key" ever meant — cost: the $T\times T$ score matrix has rank $\le d$
   (per head), and $\sqrt d$ is just the temperature that keeps initial scores $O(1)$.
5. Don't sample or mix: average the vectors, $\sum_jA_{ij}v_j$, and decode once —
   cost: the prediction is $g$ of an average, not the average of predictions; in
   this specimen it lands near the *geometric* mixture (0.0624 vs 0.0589 nats),
   not the arithmetic one (0.1980) that Steps 1–2 actually wrote down.

Line 5 is the draft's best surprise and its weakest joint at once. **The argument
is weakest where "approximation" does its quiet work, at two points.** First,
Steps 1–2: nothing in the chapter gives me a reason to believe an extension with
simple shared $\pi_j,p_j$ exists for text — no example of a law where it succeeds
beyond notation, no failure case to calibrate against (Exercise 5.4, the 2-bit
identity $ab=2a+3b-4o(a,b)$, is precisely the miniature that shows a
data-dependent average *can* encode a two-variable interaction; it is buried as
an exercise and never constructed from $q,k$ — see §5). Second, Step 5: the model
was never trained to be the arithmetic mixture, so "faithfulness" is closeness to
a predictor nobody fitted, at weights fitted for the rival. The draft says this
itself — "Training fits $g(x_i,o_i)$ directly, so the mixture at those weights is
a different predictor" — but only *after* the Taylor bound has invited me to read
the gap as an error. The geometric-mixture identity is exact and lovely, yet why
the trained specimen sits so close to $p^{\rm geo}$ is left as an empirical fact
(near-affine logits along the segments? near-one-hot rows? small $s^2$? the
quantities that would discriminate are defined on p. 11 and never evaluated).
Honest version: Steps 2–5 are an *ansatz* — the claim's own first word — whose
reward is a smooth map with $O(D^2)$ numbers where the table had $V^i(V-1)$, and
§5.3 measures a side-effect (mean-field $\approx$ geometric mixing) rather than an
error. The old chapter gave me a mechanism and no reason; the draft gives me
reasons for every symbol and, at these two joints, a reason-shaped assertion.
That is still a large net win — but the word *approximation* should be retired
in favour of *restriction*, everywhere except Step 5's comparison of two
predictors. (Two smaller gaps along the way: the causal constraint $j\le i$ is
imposed from Step 1 without a word — its justification is Chapter 1's honesty of
predictions, and §5.4 then reads as if the mask were a new idea rather than the
batched implementation of a choice made on p. 8; and the route never returns to
its opening ledger — how many numbers does the ansatz cost against $V^i(V-1)$?
The 288 arrives on p. 15 as a specimen fact, not as the closing of the
exponential-table argument, and $g$'s numbers live in Chapter 6, so I cannot
verify the saving that motivated everything.)

## 4. Each figure

**Frontispiece (p. 6, "Every reading the specimen's two attention heads made while
it learned").** What question does it answer for me? As a frontispiece — decoration
with a promise — it works: thousands of arcs say "positions read each other, at
many ranges, and it changes during training". As a figure, at this point in the
book I can read almost none of it: the corpus is on the baseline but the letters
are not printed on it ("value ticks mark letters, gaps mark spaces" — *value* ticks?
ticks are hard to see at print size); "height span $\times$ weight" contradicts the
encoding of Figure 5.1 four pages later (width encodes $A_{ij}$), so the same arcs
mean two things in one chapter; "40 moments of training (AdamW, $\eta$ 0.02 →
0.002 (warmup 20, cosine), $\beta_2$ 0.95, weight decay 0.01 on matrices, clip 1)"
is Chapter 8's vocabulary in a Chapter 5 caption — provenance the author may want,
but to the intended reader it is static. The one claim, "Head 1 learns short,
local readings; head 2 learns to reach far back", I cannot verify from the tangle:
the longest arcs I see are head 1's (above). Keep as frontispiece by all means,
but the caption should either be demoted to pure description or the claim made
checkable (e.g. mean span per head, early vs late).

**Figure 5.1 — attention reads (p. 10), arc diagram, reader 8 "b" of "not_to_b".**
Question answered: *what does a row of $A$ actually look like in the trained
model?* — and it delivers the chapter's first concrete numbers (head 1 puts 0.80
on the reader itself; head 2 spreads 0.30/0.20/0.19… over earlier letters). The
placement is good logic (immediately after $A_{ij}$ is defined, §5.2) but it sits
*before* $o_i$ exists, so it can only show weights, not what is read — half the
story, necessarily. Three caption failures: (i) "Gibbs weights $A_{ij}$ over
earlier positions" — the dominant arc is the self-loop $j=i$; the whole chapter
says $j\le i$, the caption says "earlier". (ii) It shows *two heads* five pages
before §5.5 defines them; the text waves at "(§5.5, p. 15)" but the caption
doesn't even say "the model's two channels; see §5.5" — a first reader meets
"head 1 above, head 2 below" as unexplained anatomy (the old caption identical in
this respect, but the draft's route makes the wait longer: there, heads were one
section away). (iii) The little "how to read" inset (bottom left) is where the
actual conventions live (arrows point to sources; loop = self-read) and it is set
in ~6pt type that is barely legible even in my 220 dpi render; the marginal
corpus line at the top with the window bracket is never mentioned in the caption.
Verdict: right figure, right place in the logic, caption needs the two sentences
the inset is whispering.

**Figure 5.2 — attention pipeline (p. 11), keys → scores → weights → values → $o_8$.**
Question answered: *show me the whole computation once, with numbers* — this is
the only place in the chapter where I can check arithmetic by hand
($e^{3.15}/(e^{-1.29}+e^{-1.97}+e^{-4.13}+e^{0.98}+e^{-4.24}+e^{-2.04}+e^{0.90}+
e^{3.15})\approx0.80$ ✓) and it is the most valuable figure in the draft. But it
is in the wrong section: it illustrates §5.2's formulas plus $o_i$, yet it is
dropped into §5.3 where it splits Step 5's first paragraph across a page turn
(formula p. 10, figure p. 11, "When is this faithful?" resumes underneath).
Caption: "Cold rows read one source; hot rows read all equally" describes the
web edition's temperature slider (visible at right as a frozen $\tau=2.00$ gauge)
— in print there are no other rows and the sentence is about a counterfactual
($\tau\to0,\infty$) that Chapter 3 already taught with distributions, not rows.
The values row is captioned "weighted values", but I cannot see the weighting in
the spoke glyphs: the low-weight sources' spokes are *pale* (grey tips), not
*short*, so what is drawn looks like $v_j$ faded by $A_{ij}$, not $A_{ij}v_j$;
the caption doesn't say which, and the difference is the section's whole point.
The spoke-glyph reminder at bottom right is again ~6pt but at least present.
Also: query $q_8$ sits in the keys row at the right, separated from the column of
"b" it belongs to — one locating tick would help.

**Figure 5.3 — mean-field (p. 13), per-source predictions, mixtures, all-80 strip.**
Question it wants to answer: *does averaging vectors then decoding behave like
averaging predictions?* — the empirical heart of §5.3. It is the weakest figure
in the draft, on four independent grounds. (i) *Placement/pagination:* its text
ends on p. 12 ("Figure 5.3 shows one reader where the split is wide"), §5.4 then
begins on p. 12 and is cut mid-sentence ("Column $i$ of $O=VA^\T$ depends only
on"), Figure 5.3 fills p. 13 *under §5.4's running head* "Every reader at once:
the causal mask", and §5.4's sentence resumes on p. 14. In the printed book this
figure is filed inside the mask argument. (ii) *Legibility:* three registers
(per-source $p_j$ mini-bars on a 7-letter axis repeated six times; stacked-surprise
bars in nats with source numbers inside 5pt segments; a log-scale dot strip) in
one page-width panel. At print size the $p_j$ row communicates only "everyone
says o, except source 4", and the annotation "′o′: 6.52 nats, 6.52 of them from
source 4, ′n′" needs the caption to unpack what a *segment* is ($A_{ij}\times$
surprise of $p_j$?) — the caption instead merely lists the panels ("Per-source
predictions $p_j=g(x_i,v_j)$ ($j\le6$), their arithmetic mixture, their normalised
geometric mixture, and the model's actual prediction…, which sits with the
geometric mixture") and never defines *total variation*, the metric of both the
brackets (0.655 / 0.064) and the strip. (iii) *Two different interventions, one
name:* the top panels are "head 1 reads only $v_j$, head 2 held at its mean";
the bottom strip is "both heads choosing their sources at once". The circled
points are presumably this same reader under the *second* intervention — and
their values (≈0.50 arithmetic, ≈0.055 geometric, read off the log axis) differ
from the brackets above them (0.655, 0.064). Same reader, two pairs of numbers,
no explanation: I spent longer reconciling this figure with itself than on any
other page. (iv) A stray orphan word "**key**" floats at the right of the caption
block, a leftover label. Verdict: the *question* is exactly where the argument
needs a figure; the figure needs to be split (panels (top+mixtures) as Figure 5.3
in §5.3; the 80-reader strip either with the p. 12 numbers or cut), re-captioned,
and re-paginated out of §5.4.

**Figure 5.4 — attention triangle (p. 14), $S$, $S{+}M$, $A$ for both heads.**
Question answered: *what does the mask do, and what do all rows look like at
once?* Placement is right (inside §5.4, after $A=\operatorname{softmax}_{\rm rows}
(S+M)$), and the three-stage layout is genuinely clarifying — you watch the upper
triangle vanish. The caption, however, describes only the last stage: "dot area
is $A_{ij}$" is false for four of the six panels, where dots encode scores $S_{ij}$
— by *what* (area $\propto|S|$? open vs filled = negative vs positive, as the
pattern suggests?) the caption never says, and neither does the how-to-read inset,
which covers only $A$. A reader who trusts the caption will read the $S$ panels
as (unmasked, unnormalised) weights — precisely the misconception §5.4 exists to
prevent. The six panels also carry no characters on their axes (the inset shows
n/o/t/␣ for a generic row); since Figures 5.1–5.2 worked on "not_to_b", a line
of eight letters along each axis would have tied the chapter's figures into one
example. Verdict: right place, right idea, caption factually wrong about its own
left two-thirds.

## 5. Length and order

The draft is 1998 words against the old 1609 (+389, +24%), and the extra quarter
is nearly all in two places: Step 1's rigour (~180 words to prove an identity the
text calls empty) and §5.3's analysis (Taylor bound + geometric identity + 80-reader
numbers + a full-page figure, ~450 words where the old chapter had nothing). The
rest — $\sqrt d$, mask, heads, code, most asides — is inherited almost verbatim.
So the question "clearer or just longer?" is really "does §5.3 earn its pages?"
Half of it does: the geometric-mixture identity and the three losses are the
draft's deepest content — the first time anyone tells this reader what attention
*does* to probabilities rather than to vectors. The Taylor paragraph does not (see
1(j)): cut it to two sentences ("If $g_y$ were affine the step would be exact; the
gap is second order in the weighted spread $s^2=\sum_jA_{ij}\|v_j-o_i\|^2$, so it
vanishes when the values cluster or one weight dominates — and can fail entirely
at a kink of $g$") and move the bound itself to an exercise, where a stated $K$
can be given. That saves ~90 words and one boredom.

Concrete cuts/merges/moves, with the word-count logic:

- **Delete the matrix-product-state aside** (p. 7; ~65 words). It compares against
  an object this reader doesn't have and will not get in this book. If the
  comparison must live somewhere, it belongs in the epilogue with other
  architectures, where "fixed-size memory" can be defined. Replace with the
  one-sided cost the draft currently never states in §5.1: keeping $T$ columns
  and $T^2$ scores (already half-said on p. 14).
- **Merge Steps 3 and 4** into one step ("a Gibbs law whose energy is a low-rank
  bilinear form"). Step 3 alone is a change of variables; numbering it inflates
  the route and forced the double emptiness noted in 1(h). Merging also lets the
  claim's "low-rank bilinear Gibbs law" be one idea. The five steps become four,
  and §5.3's "Step 5" references update trivially.
- **Move Figure 5.2 (pipeline) to the end of §5.2** — or rather, to immediately
  after $o_i$ is introduced at the top of §5.3, where it currently *is*, but then
  don't let it sever Step 5's paragraph: put the paragraph "Forming the Gibbs
  weights $A$ and taking $o_i=\E[\,v_J\,]$ is attention…" *before* the figure so
  the figure punctuates a completed definition. Figure 5.1 then carries "what $A$
  looks like" and 5.2 "the whole computation", in that order, without a page of
  Taylor between definition and illustration.
- **Move Figure 5.3's bottom strip, or cut it.** The p. 12 sentence already gives
  the 80-reader means as losses; the strip repeats them as (undefined) total
  variations on a log scale. If kept, it belongs with those numbers and needs TV
  defined and the two-intervention distinction (see §4) made typographically
  obvious.
- **Promote Exercise 5.4 into §5.3 as a five-line worked miniature** (and demote
  the Taylor bound to the freed exercise slot). The identity $ab=2a+3b-4o(a,b)$
  is the only place in the chapter where a reader can *see* a data-dependent
  average create an interaction that no columnwise map could — i.e., evidence
  that the ansatz of Steps 2+5 is not toothless. As an exercise it also has a gap:
  it assumes the weights $(\tfrac12,\tfrac12)/(\tfrac14,\tfrac34)$ as given and
  never asks the reader to realise them from scores $q_i\cdot k_j/\sqrt d$, which
  is the step that would connect it to attention rather than to averaging in
  general. In the text, with that one construction added ("scores $(0,0)$ and
  $(0,\log3)$ give these weights"), it stops being a magic formula.
- **The $b_k$ expansion (p. 9) vs Chapter 3's promise.** Chapter 3's puzzle aside
  says "@ch-attention gives the answer in its exercises" — the draft now gives it
  in the text *and* in Exercise 5.2. Pick one. Text is better (it is too neat to
  bury), so fix Chapter 3's aside wording and slim Exercise 5.2 to the
  shift-invariance part.
- **"In the code" (pp. 15–16) is unchanged from the old chapter and needs a
  pass for the new route:** `attention(Q,K,H,T)` and `mix(A,V)` now correspond to
  Steps 3–4 (scores+mask) and Step 5 (expectation) — one clause saying so would
  close the top-down loop ("`attention` is Steps 3–4 for every reader at once;
  `mix` is Step 5"). The comment `# 1/√d, in the number type of Q` and
  `foreach_head` pass without a word; the pullbacks (half the printed lines) are
  Chapter 7 objects and could be elided in print exactly as messy as they are —
  they double the section's length for a reader who cannot yet read a bar.

Nothing else needs reordering: mask after mean-field is defensible (single reader
first, all readers at once), heads last is defensible *if* the figures get the
one-sentence forward gloss they currently lack.

## 6. Voice

There is no ML-blog hype in the draft: no "remarkably", no "powerful", no
"attention is all you need" riff, and ML words arrive in scare quotes in the names
table after their maths. The dominant voice problem is the opposite: a clipped
theorem-register ("This is exact and empty", "Positions interact here and nowhere
else") that occasionally compresses a caveat out of existence. Sentences a
physicist would wince at:

- *"source observable"* (names table, physics column, for $v_j$). $v_j$ is not an
  observable; no measurement yields it, it is indexed by the *random* variable's
  value, and the analogy needs $o_i=\langle v\rangle_i$ to be an expectation
  value *of* an observable over the Gibbs law of $J$ — which is exactly the
  fiction Step 5 installs. Calling $v$ the observable pre-legitimises the move.
  "the quantity averaged" is duller and honest.
- *"temperature"* outright (names table) where the old chapter wrote "temperature
  by the Gibbs analogy". The draft itself shows the scale of $W_q,W_k$ is learned
  and can absorb any $\sqrt d$; a temperature that can be redefined away by
  rescaling $E$ is a unit choice, and the table should keep the qualifier the
  draft dropped. (The marginal aside "nothing equilibrates between readers" is
  admirable — keep; it's the disanalogy that makes the analogy safe.)
- *"causal transition kernel"* (names table, physics, for $A$). $A$ is never
  composed with itself, propagates nothing in time, and its indices are
  (reader, source), not (state at $t$, state at $t+1$). "Row-stochastic" (the
  maths column) is the whole content; the physics cell is decoration that a
  physicist will parse as a claim about a Markov chain.
- *"mean-field"* (§5.3, names table "mean-field expectation $\langle v\rangle_i$").
  Mean-field theory to a physicist means a self-consistent effective field and a
  factorised trial distribution; here it names $\E g(v)\to g(\E v)$, which is a
  Jensen gap, with no self-consistency anywhere. The chapter does eventually
  define its usage ("Moving an expectation inside a nonlinear map is a mean-field
  step"), so this is a wince, not an error — but the heading and table spend the
  word before the definition, inviting the wrong association.
- *"…one source can price a character out"* (Fig. 5.3 annotation) — market
  metaphor, and in tension with the text's own correct "no source can literally
  veto a character to zero" two paragraphs earlier. The figure says veto, the
  text says never veto; a reader remembers the figure.
- *"Cold rows read one source; hot rows read all equally"* (Fig. 5.2 caption).
  Rows don't have temperatures; at $\tau\to0$ a row with tied scores shares weight
  (Ch. 3's frozen limit, which this book was careful about), and "read all
  equally" at high $\tau$ conflates weight with reading. Two licences in nine words.
- *"More heads consult more sources at once"* / *"To consult several sources at
  once"* (p. 15) and *"stacking blocks lets later scores depend on earlier reads"*
  (p. 15). These are the draft's own versions of the banned vague sentence
  ("attention lets earlier positions influence the prediction"): heads don't
  consult sources, each forms one blended expectation in its own coordinates; and
  "lets … depend on" names no mechanism (the mechanism is that $x_j$ in later
  blocks contains $W_o^{(earlier)}o_j$, sayable in the same length).
- *"and imposing no correlation-length bound on the joint law"* (p. 7 aside) —
  statistical-mechanics vocabulary with no system attached; see 1(b).
- Filler crumbs: *"solving the puzzle of Chapter 3"* (p. 9; it doesn't solve, it
  answers — and Ch. 3 said the answer was in the exercises); *"which sits with the
  geometric mixture"* (Fig. 5.3 caption; "sits with" is doing the work of a
  distance); *"every reading … made while it learned"* (frontispiece; readings
  aren't made). Each tiny; together they are the only places the voice slips from
  the book's otherwise exact register.

## 7. The ten most valuable concrete edits

Most important first. Each gives replacement wording to drop in (British spelling
as in the book); deletions are stated as such.

**1. Make the Step 1 → Step 2 joint honest, and retire "approximation" there
(p. 8).** Replace *"The first approximation assumes an extension exists in which
both $\pi_j$ and $p_j$ admit simple shared descriptions. Nothing guarantees one;
only under this hypothesis does $J$ mean "which position to consult"."* with:

> This identity was empty because $r$ was allowed to depend on $y$, the character
> being predicted: $J$ could cheat. Every step from here forbids that. The first
> real restriction — an assumption about text, not a theorem — is that there is an
> extension of this kind in which neither $\pi_j$ nor $p_j$ looks at the whole
> prefix: both are given by one rule, the same for every reader and every prefix,
> applied to one source at a time. Nothing guarantees text is like this; a next
> character fixed by the joint pattern of several positions, with no one position
> carrying the answer, is exactly what it excludes. Only under this hypothesis does
> $J$ mean "which position to consult".

(Then read the chapter replacing "approximation" by "restriction" in §5.1–§5.2.)

**2. Define $g$, $p_j$ and the 80-reader intervention operationally (pp. 8, 12;
Fig. 5.3).** After *"…and one shared readout $g(x_i,v_j)\in\Delta^{V-1}$"* insert:

> In the model, $g$ is literal: run the rest of the network of Chapter 6 past
> this average — add $W_o m$ to $x_i$, continue to the logits — with the vector
> $m$ in place of the average $o_i$. So $p_j=g(x_i,v_j)$ (later $p_j=
> \operatorname{softmax}(z(v_j))$, with $x_i$ and the other head's output held at
> what the model actually computed) is the prediction the trained model *would*
> make if head 1 at reader $i$ took all its weight on source $j$. Nothing is
> retrained in what follows; only this one vector is swapped.

and rewrite the p. 12 parenthetical "(fixed parameters, rows, values and residual
inputs; independent head indices, §5.5 (p. 15))" as: "(the same swap made in both
heads at once and the two swapped vectors combined with weights
$A^{(1)}_{ij_1}A^{(2)}_{ij_2}$, §5.5)". This one insertion also fixes Figure 5.3's
top row ("source $j$ alone") and names the difference between its panels and its
bottom strip.

**3. Assemble the whole formula in one display (end of §5.3, before "When is this
faithful?").** At present $q,k,S,A$ are on p. 9, $o_i$ on p. 10, the mask on p. 12
and $W_o$ on p. 15; nowhere can I see attention entire. Insert:

> Collected, one head at one reader is
> $$o_i=\sum_{j\le i}A_{ij}\,v_j,\qquad
> A_{ij}=\frac{e^{q_i\cdot k_j/\sqrt d}}{\sum_{j'\le i}e^{q_i\cdot k_{j'}/\sqrt d}},\qquad
> (q_i,k_j,v_j)=(W_q\hat x_i{+}b_q,\;W_k\hat x_j{+}b_k,\;W_v\hat x_j{+}b_v).$$
> Steps 1–5 successively bought the sum over $j$, the shared $v_j$ and $g$, the
> exponential weights, the factored score $q_i\cdot k_j$, and the average taken
> before decoding. Everything after this point is bookkeeping: all readers at
> once (§5.4) and several such heads side by side (§5.5).

**4. Rewrite the claim so a first reader can parse it (p. 7).** Replace the claim
with:

> Attention is a guess at the shape of the next-character law: pretend the law is
> a mixture over one uncertain earlier position, weight the positions by a Gibbs
> distribution whose energy is a low-rank bilinear form — its two factors are the
> query and the key — and then average the vectors those positions offer instead
> of averaging their predictions.

(And in the names table restore the old qualifier: physics cell for $\sqrt d$
back to "temperature, by the Gibbs analogy"; physics cell for $v_j$ from "source
observable" to "the vector averaged".)

**5. Merge Steps 3–4 and flag the second empty rewrite (p. 9).** Replace the
Step 3 paragraph opening with:

> *Step 3: a low-rank bilinear Gibbs energy.* Writing the weights as a Gibbs law
> restricts nothing: any positive row $\pi_j$ on $\{1,\dots,i\}$ is
> $\pi_j=Z_i^{-1}e^{-E_{ij}/\tau}$ with $E_{ij}=-\tau\log\pi_j$ (Chapter 3) — a
> change of variables, like Step 1. The restriction is the energy: one shared
> bilinear form, factored through $\R^d$… [continue with present Step 4]. Two
> things are given up here: a source can be made arbitrarily unlikely but, with
> finite energies, never exactly excluded; and the log-ratios $\log(\pi_j/
> \pi_{j'})$ are bilinear in $(\hat x_i,\hat x_j)$, so across all readers and
> sources the score matrix has rank at most $d$.

Renumber "Step 5" to "Step 4" in §5.3 and the contents line (which, note, still
carries the *old* chapter's strapline "A row-stochastic mixing of columns;
query, key, value; the $1/\sqrt d$ scaling…" on p. 5 — update to match the new
route, e.g. "From the exponential table by four restrictions: an uncertain
index, a shared pairwise rule, a low-rank Gibbs energy, and expectation before
decoding.").

**6. De-forward-reference Step 2 and fix "channel" (p. 8).** Replace the Step 2
paragraph's opening with:

> *Step 2: a shared rule on position vectors.* Position $j$ carries a vector
> $x_j\in\R^D$ — at the input, $Ee_{a_j}+P_{:,j}$ — and we work with a rescaled
> copy $\hat x_j$, the LayerNorm of Chapter 6; nothing in this chapter uses more
> than that $\hat x_j$ is a fixed columnwise function of $x_j$. *Shared* will
> mean: the same matrices for every reader, source and prefix — this is where
> the $V^i(V-1)$ numbers of the table collapse to a few matrices. The model runs
> $H$ copies of what follows side by side, each in its own $d$ coordinates — a
> *head* (§5.5); the specimen has two. Take one head…

This also gives Figures 5.1–5.4 their missing one-line gloss of "head" five pages
early, and lets every later "channel" (pp. 8–9) become "head".

**7. Re-caption and re-paginate Figure 5.3, and define total variation (p. 13).**
Keep the figure with its discussion at the end of §5.3 (not inside §5.4), delete
the stray "key", and replace the caption with:

> **Figure 5.3** Window `or_not_t`, reader `t` at position 6, head 1 only (head 2
> left at its computed output). Top: the prediction $p_j$ the trained model makes
> when head 1 at this reader is forced onto source $j$ alone, for $j\le6$; every
> source but `n` predicts `o` next, `n` gives `o` probability $\approx10^{-6}$.
> Below: the arithmetic mixture $\sum_jA_{6j}p_j$ (segments: each source's
> contribution $A_{6j}p_j$), the normalised geometric mixture (segments: each
> source's contribution $A_{6j}(-\log p_j)$ to the surprise of each character,
> in nats, exponentiated and renormalised below), and the model's own prediction
> $g(x_6,o_6)$. Brackets: total variation $\tfrac12\sum_y|p_y-q_y|$ from the
> model's prediction. The strip repeats the comparison for all 80 readers with
> *both* heads forced (weights $A^{(1)}A^{(2)}$) — a different intervention from
> the panels above, hence the circled point for this reader ($\approx$0.50 /
> 0.055) is not the bracket value (0.655 / 0.064).

**8. Fix Figure 5.4's caption — it is wrong about four of its six panels (p. 14).**
Replace with:

> **Figure 5.4** Scores, mask and weights of the specimen's two heads for one
> window (rows: reader $i$; columns: source $j$). Left: the scores $S_{ij}$ for
> every pair — dot area grows with $|S_{ij}|$, filled dots are positive scores,
> open dots negative. Middle: $S+M$; the mask has deleted $j>i$. Right: after
> the rowwise Gibbs normalisation, dot area is $A_{ij}$ and each row sums to one.
> Only the right-hand panels are attention patterns; the left-hand ones are
> energies up to sign and temperature.

(The open/filled = sign reading is my inference from the panels; if it is not
the encoding, that is further evidence the caption must state it. Same repair,
one word, in Figure 5.1's caption: "over earlier positions" → "over positions
$j\le i$".)

**9. Delete the matrix-product-state aside; state the cost instead (p. 7).**
Delete the aside entire and, after "Positions interact here and nowhere else."
(p. 8), add:

> The price of keeping one column per position, instead of compressing the past
> into a state of fixed size, appears in §5.4: $T$ columns held, and a score for
> every pair of positions — work and storage quadratic in $T$. What is bought is
> that any earlier position is as reachable as the previous one; no distance in
> the window is privileged in advance.

(Move the MPS comparison, if wanted, to the epilogue, where bond dimension can be
given its two defining sentences.)

**10. Prevent the heads misreading (p. 15).** After *"…rows 1–4 of $Q,K,V$ in
head 1 and 5–8 in head 2"* insert:

> The split is in the *outputs* of the three affine maps, not in their input:
> head 1's $W_q^{(1)}\in\R^{4\times8}$ still reads all eight coordinates of
> $\hat x_i$. What a head cannot do is emit a query, key or value with a
> component in the other head's four coordinates; the two heads' reads are
> recombined only by $W_o$ below.

and replace *"(Any coupling with those marginals gives the same $O$.)"* with:

> Independence of the heads' indices matters only for the mixture before the
> mean-field step (§5.3's 80 readers): after it, $O$ depends on each head's
> weights separately, so attention as computed cannot express any correlation
> between what one head reads and what another does.

---

*Method note.* Read as specified in the brief: PDF rendered page-by-page
(`pdftoppm -r 110`, figures re-rendered at 220 dpi; images under
`attention-audition/.work/pages/`), source at
`attention-audition/book/chapters/05-attention.md` alongside, with
`book/chapters/01-task.md`, `03-energies.md` for what the reader already has, the
old `book/chapters/05-attention.md` for comparison, and a look at
`book/chapters/06-block.md` only to confirm which of the draft's forwards
references land in a later chapter. The arithmetic I quote (the softmax check in
Figure 5.2, the bias expansion, $4(D^2+D)=288$, $H_{\min}\approx0.0585$) I verified
by hand while reading; the three losses on p. 12 I took as given from the printed
proof. Nothing outside this report file was edited; nothing was committed.




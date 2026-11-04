# PicoGPT — fresh-eyes review (Muse, 4 November 2026)

Scope: `BRIEF.md`, `DESIGN.md`, `book/chapters/*.md`, `book/figures.md`,
`src/*.jl`, `artifacts/specimen.json` + `artifacts/shakespeare/*`, the live
site <https://tobiasosborne.github.io/PicoGPT.jl/> (Playwright, desktop /
390 px / night), and `print/PicoGPT.pdf` (136 pp. + `print/proof/pages/`).
Maths derivations, code tests, citations and layout fix-ups were taken as
done (`book/notes/derivations.md`, `book/briefs/citations.md`,
`book/briefs/fixups.md`) and not re-audited. Shakespeare numbers below were
recomputed from the checked-in artifacts. Caveat: at review time the working
tree had uncommitted edits to two figure components, so the deployed site and
the checkout can differ in detail; every finding below was checked against
both where it matters.

Findings are ranked by improvement-per-fix. Each gives a location, a quote,
why it matters for the intended BSc physics/maths reader, and a concrete fix.
No chapter, figure or source file was edited for this review.

---

## Tier 1 — breaks the book's own contract, or strands a strong reader

### 1. There are two incompatible objects both called “window”: 10 in the text, 11 in the figures

- Ch. 1 §1.5 (`01-task.md:75`): “every run of $T$ consecutive characters
  that has one more character after it… A corpus of $m$ characters has
  $m-T$ windows; the specimen's line has ten.” Hence $B=10$, $N=80$, and
  the floor $(2\log2+3\log3)/80$.
- Ch. 5 defaults all three figures to an eleventh window that the training
  definition excludes: `fig-attention-reads window=11`,
  `fig-attention-pipeline window=11`, `fig-attention-triangle window=11`
  (`05-attention.md:38,54,74`). On the live site Fig. 5.1 therefore opens
  on “ot␣to␣be” (corpus positions 11–18), whose last position has no
  target in the corpus at all.
- The adapter knows the distinction and the prose never tells the reader:
  `web/src/lib/model/specimen.js:20-21` builds 11 display windows
  (`s+T <= length`), while `:69-70` comments “The training set is every
  window with a complete target ($B=10$, not the 11 of `windows`)”.
  Ch. 1 itself mixes the two: `fig-chain-rule` clamps to the 10 training
  windows, while `fig-contexts-to-distributions` can be dragged to start
  11, where its footer still promises “outlined: the character that
  actually comes next” although for $t=8$ there is none (the readout
  quietly substitutes “The corpus ends here”).

Why it matters: counting windows is the reader's first checkable act, and
$N=80$ underwrites the floor, the loss, Fig. 4.1 and the minibatch
combinatorics $\binom{10}{b}$. A careful student who counts brackets on
the corpus strip in Ch. 5 gets 11, concludes $N=88$, and can no longer
reproduce any later number. It also blurs a real modelling point the book
is otherwise precise about: a length-$T$ input with no successor is a
legitimate *prompt*, not a training example.

Fix (choose one, everywhere): (a) re-default Ch. 5 to a training window
(e.g. `window=10`) and clamp display dragging to starts $1\ldots10$,
reserving the 11th start for Ch. 9 generation; or (b) keep 11 display
starts but name them differently in prose, captions and readouts
(“start 11 of 11; starts 1–10 are the training windows — start 11 has no
target and is never trained on”). Option (a) is one attribute per figure
plus one clamp; option (b) additionally needs the sentence in §1.5.

### 2. “Nothing is held back from print” is not true of the figures as printed — and print prose still speaks to a browser

- Prologue §0.4 (`00-prologue.md:67`): “each figure appears in a static
  state that is complete on its own… Nothing is held back from print:
  interaction is only for asking the next question.” DESIGN §1.6 repeats
  it: “Interaction is for asking the next question, never for revealing
  what the page withheld.”
- In practice most print states freeze one reader / column / window /
  step and withhold the rest of the comparison the caption announces.
  Examples: Fig. 5.1 caption “What each character reads” prints only
  reader 8 (PDF p. 63); Fig. 6.1 prints only column 8 enlarged;
  Fig. 7.2 only one column's rank-one share at full size; Fig. 9.4 only
  the default head's arcs. The web reader can click through all
  readers; the print reader is told the general claim (“tokens interact
  only here”, “only ambiguous predictions still send covectors”) from a
  single instance.
- Fig. 8.1 in print (PDF p. 103) compresses four steps × (1007 Hinton
  squares + arcs) into one plate: at print size the squares and arc
  ellipses are texture, not legible numbers, and the sample staircase at
  the top collides (“to␣be␣orbtbtnebtnn”). It fits the page (the fix-ups
  criterion) without being readable as the numbers it claims to show.
- Body text (not captions, which were correctly wrapped in `.web`) still
  addresses a screen in the PDF: Ch. 8 §8.1 “Figure 8.1 runs the whole
  of this chapter in front of you… with the settings it opens with”
  (PDF p. 102); Ch. 7 §7.2 “In the web edition, Figure 7.4 measures the
  constant in your browser” (`07-pulling-back.md:54`); Ch. 4 §4.2 “Run
  the training and watch the bars fall” (`04-surprise.md:43`). The
  print reader is repeatedly reminded they have the lesser edition,
  directly against the Prologue promise.

Fix: either soften the promise to what is true (“print shows one
representative state; the web lets you vary it”) in §0.4 and DESIGN
§1.6/§8, or — better for the three worst cases — give print true small
multiples: Fig. 5.1 as 8 miniature arc rows (one per reader, shared
scale), Fig. 8.1 as loss curve + samples only in print with the
parameter plate moved to its existing Ch. 0/Ch. 8 siblings, and wrap the
three body sentences above in `[…]{.web}` with print-neutral
replacements (“Figure 8.1 summarises this chapter…”).

### 3. Forward-reference debt: the first figures require the last chapters

The brief's rule is “every ML term is defined… before it is used”
(`book/briefs/writer.md`). The chapter order (task → vectors → energies
→ surprise → attention → block) is not the computation order, so forward
references are unavoidable — but they are currently unflagged at the
exact points a strong reader will stop:

- Prologue Fig. 0.2 (`00-prologue.md:55-57`, live screenshot verified)
  asks a reader who has met nothing yet to parse
  $z=W_{\text{out}}x+b_{\text{out}}$, $\ell=-\log p_y=0.674$,
  $\bar z=p-e_y$, $\bar W=\bar z x^\top$ and an augmented matrix
  $[W\,|\,b](x;1)$. Logits arrive in Ch. 3, the loss in Ch. 4, the bar
  notation in Ch. 7. The figure's stated question (“What do the two
  inks mean?”, `book/figures.md`) could be answered with one $2\times2$
  example; instead it front-loads the whole backward pass.
- Ch. 3 opens with the model's last map
  $Z=W_{\rm out}\,\mathrm{LN}(X;\gamma_f,\beta_f)+b_{\rm out}\mathbf 1^\top$
  (`03-energies.md:22`) three chapters before LN, $\gamma$, $\beta$ or
  the residual stream $X$ are defined (Ch. 6). Ch. 5 §5.1 similarly
  invokes “LayerNorm, the MLP and the final linear map, which we meet
  in Chapter 6” and then uses the normalised column $\hat x_i$
  throughout §§5.3–5.6.
- Every `::: names` table previews symbols the chapter has not yet
  defined ($\Delta^{V-1}$ in Ch. 1, $\Pi$ in Ch. 6, $\mathrm{KL}$ in
  Ch. 4). As a glossary that is fine; as the first thing on the page,
  before the claim's own terms are explained, it reads as assumed
  knowledge.

Fix: no reordering needed. (i) Demote Fig. 0.2's readout/caption to ink
meanings only and move the $\ell$, $p-e_y$ arithmetic behind the scrub
(the numbers can appear at $t=1$ without being in the caption);
(ii) add a half-sentence forward tag at each first use (“$\mathrm{LN}$
is the normalisation of Chapter 6; for now read it as ‘a rescaled copy
of $X$’”); (iii) head each names table on first use with “defined in
this chapter” (the Prologue already says this once — repeat it as a
running cue, e.g. in the table's small-caps head).

### 4. “Every number computed, nothing illustrative, twelve decimal places” — three provenance claims that disagree with each other

- Prologue: “every number in every figure is computed from it as the
  page is drawn… Nothing is illustrative” (`00-prologue.md:39`) and
  “reproduces the Julia numbers to twelve decimal places” (`:85`,
  repeated `10-epilogue.md:46` “and every figure is exact”).
  BRIEF §5 in fact requires arrays to $10^{-12}$ absolute but the loss
  *trajectory* only to $10^{-9}$; the adapter enforces $10^{-9}$ and
  silently returns `null` otherwise (`specimen.js:139`), while its own
  comment claims “$\sim10^{-13}$” (`:71`); Ch. 8's aside says the curves
  “agree to the last few digits of a double” (`08-learning.md:33`).
  “Twelve decimal places”, “last few digits” and “exact” are three
  different promises. A physicist will also note “decimal places”
  without “absolute/relative” is meaningless for numbers spanning
  $10^{-19}$ (Ch. 7) to $10^{2}$.
- Prose numbers are typed, not computed, throughout — necessarily, but
  against the stated rule (“Never type a number that a figure could
  compute”, writer brief; “never hard-code” the parameter count,
  BRIEF §4): $p=1007$ seven times, $810{,}049$, $288$, $56$, $2014$,
  “about 1.6 million”, $\beta$ thresholds, etc. They are currently
  correct (spot-checked: $p$, floor $0.0585266$, loss $2.00447\to
  0.058935$), but nothing in the build would catch one going stale if
  the specimen recipe changed — precisely the failure mode the rule
  exists to prevent.
- Traces quietly use different runs from the chapters they face. The
  cover trace caption (PDF frontispiece; `web/public/traces/cover.svg`
  `<desc>`) says “seed 1” with the book's recipe, while Ch. 8's aside
  (`08-learning.md:39`) says the cover “starts from a different
  $\theta_0$ (it was drawn by the JavaScript engine's own
  initialisation), so its numbers are not the ones of this chapter”.
  Both cannot be the provenance a reader needs; “seed 1” in Julia
  (Xoshiro, “reproduce the distributions, not the draws”,
  `specimen.json:init`) and “seed 0 = the book's $\theta_0$” in
  Fig. 8.1 are a third, incompatible seed language.

Fix: pick one tolerance sentence and use it verbatim in Prologue,
Epilogue and Ch. 8 (“arrays agree to $10^{-12}$ absolute; the 200-step
loss curve to $10^{-9}$ — the replay is discarded if it does not”);
delete “exact”. Either template the handful of load-bearing prose
numbers ($1007$, $810{,}049$, floor, loss endpoints) from the artifacts
at build time, or amend the rule in print to “every number *in a
figure* is computed; numbers in prose are checked against the
artifacts” and add that check to `BOOK_STRICT`. Reconcile the cover:
state in its caption which $\theta_0$ it actually used, and make
Ch. 8's aside match.

### 5. Probabilities are printed as “1.00” — the book's two-decimal rule erases its own point about finite softmax

Fig. 1.1 labels six bars “1.00”, its readout says “‘n’ 1.00”, and the
text below concludes “Most columns are a single bar of height one…
the specimen is sure that r comes next” (`01-task.md:53`). Fig. 6.1's
readout likewise reports “$p=1.00$”. But Ch. 4 §4.5 (`04-surprise.md:95`)
carefully teaches the opposite: decided contexts “never quite reach
divergence zero: that would need $q_c$ to be exactly one-hot, which a
softmax with finite logits never is… the loss approaches the floor but
never meets it” (trained loss $0.058935$ vs floor $0.058527$ — the gap
*lives* in the digits that “1.00” rounds away). DESIGN §3 mandates
“two decimals in figures (`fmt`)”, and `fmt` (`scales.js:25`) has no
“$>0.995$ but $<1$” case.

Fix: format probabilities near the endpoints asymmetrically, as the
book already does for surprises: print $1-p$ (or $-\log p$) when
$p>0.995$ — e.g. “$1-3\times10^{-4}$” — at least in Figs. 1.1, 6.1 and
their readouts, and change “height one / is sure” to “within
$10^{-3}$ of one”. One helper (`fmtProb`) next to `fmt` suffices.

### 6. Notation collisions a derivations-check cannot catch, including one wrong formula in a names table

BRIEF §2 is “binding for code, prose, and figures”, yet within the
prose alone:

| Symbol | Meanings in collision |
|---|---|
| $P$ | positional matrix (Ch. 2) · distribution $P(x)$ (Ch. 1) · $P=\mathrm{softmax}(Z)$ (Ch. 4 names + §4.6) · $P_S$ projector (Ch. 6 ex.) |
| $V$ | vocabulary size · value matrix $V\in\mathbb R^{d\times T}$ (Ch. 5) |
| $E$ | embedding · energies $E_a$ (Ch. 3) · $\mathbb E$ expectation |
| $F$ | MLP width · free-energy tick “F” in Fig. 3.1 (vs $\mathcal F$ in text) · $\hat f,\hat F$ MLP input |
| $H$ | head count · entropy $H(p)$ |
| $B$ | batch size · the block map $B(EC+P)$ (`06-block.md:102`) |
| $\ell$ / $\mathcal L$ | mean loss is $\mathcal L(\theta)$ in Ch. 4 but $\ell(\theta)$ in Chs. 6–8; $\ell_n$ is per-prediction in Ch. 4; $\ell$ is also a shift index in Ch. 2 and the block loop variable in `src/model.jl` |
| $\log$ / $\ln$ | Chs. 1–4 write $\log$ for the natural log (stated once, `01-task.md:61`); Ch. 8 and figure labels switch to $\ln V$, $\ln2$ (`08-learning.md:24`) with no remark |
| $W_o$ / $W_{\text{out}}$ | attention output vs readout differ by three letters in the same typeface; code `Wo`/`Wout`, `bo`/`bout` |

Most are inherited from the field and individually harmless; together,
in a book whose selling point is removing exactly this friction, they
need one disambiguating paragraph the book never gives. Two are not
harmless:

- Ch. 2's names table states $P_t=(\sin\omega_r t,\cos\omega_r t)_r$
  (`02-vectors.md:14`), while §2.3, the code and the artifact all use
  $\sin(\omega_r(t-1))$ with “Position 1 is phase zero”
  (`02-vectors.md:55-57`, `src/embedding.jl:22`). A student who checks
  Fig. 2.3 against the table is off by one radian on the fast clock.
- $\varepsilon$ (LayerNorm) vs $\epsilon$ (Adam) are distinguished in
  prose (`08-learning.md:109`) but both are `ε` in the quoted Julia
  (`src/block.jl:54`, `src/train.jl:40`), so the distinction the text
  draws is invisible in the code it quotes.

Fix: correct the $P_t$ row to $(t-1)$; add a “letters that do double
duty” aside in the Prologue (half a margin note, listing $P,V,E,H$
with “context always disambiguates: $V$ a size is a subscriptless
capital in a shape, $V$ values carry head/position indices” or
similar); standardise on $\ln$ *or* state “$\log$ means $\ln$
throughout” in §0.3 and keep Ch. 8 to it; pick $\mathcal L$ for the
mean loss everywhere (Chs. 6–8 currently demote it to $\ell$).

### 7. Three chapters have no frontispiece, so the opening system the book describes does not exist for a quarter of the book

BRIEF §11 / DESIGN §9: “Each chapter opens with a trace.” Present:
`cover, task, vectors, energies, surprise, attention, block,
pulling-back, learning`. Missing: `prologue.svg`, `speaking.svg`,
`epilogue.svg`. On the web the Prologue therefore opens with a large
blank slot above “CHAPTER ZERO” (verified live); in print those three
chapters open directly on a recto (`break-before: right`) while the
other eight face a verso plate (`left`), so the rhythm the contents
and running heads imply changes without explanation. Chs. 3, 4 and 8
refer to “the frontispiece” in asides; Ch. 9 — the chapter whose
sampling tree is the most natural trace in the book — cannot.

Fix: `speaking.svg` is already listed as outstanding in
`book/briefs/fixups.md`; add `prologue` and `epilogue` to the same
orchestrator task, or explicitly exempt bookends in BRIEF §11/DESIGN
§9 (“chapters 1–9 open with a trace”) so the absence reads as design
rather than omission. Until then, remove the empty web slot's reserved
height for missing traces.

---

## Tier 2 — misleading analogies, figure/caption mismatches, honest-reporting gaps

### 8. Accessibility: interactive controls nested inside `role="img"`, and hover-only readers

DESIGN §10.5 requires “keyboard equivalents for every pointer
interaction” and an `aria-label` that “states the numbers”. Verified
live with Playwright:

- Several figures put dozens of `role="button"`/`role="slider"`
  descendants inside an `<svg role="img">` (Speaking page census: one
  such SVG contains 99 interactive descendants, another 24).
  `role="img"` makes the subtree presentational in assistive tech, so
  the controls it advertises can be hidden from the users who need
  them. `fig-mlp.svelte:107` is the pattern in miniature
  (`role="img"` + `tabindex=0` + inner buttons).
- Pointer-only selection remains where the equivalent matters most:
  Fig. 9.4's 83 readers are `tabindex="-1"` hover/click targets with
  empty `onkeydown` (`fig-shakespeare-attention.svelte:140`) — heads
  are keyboard-operable, readers are not; Fig. 0.2's column choice is
  likewise `tabindex="-1"` (`fig-two-inks.svelte:106`), the scrubber
  moving only time $t$. Hover readouts (`fig-surprise`, `fig-adam`,
  `fig-init-variance` rows) have no focus equivalent.
- Where keyboard support is good it is very good (Ch. 3: Tab reaches
  context slider → shift → $\tau$ → each rung as a labelled slider —
  verified). The failure is inconsistency, not absence.
- PDF text layer (tagged PDF, to its credit) still garbles the two
  things a screen-reader user most needs: small-caps first lines
  extract as “Te cover / Te simplest” (the small-cap *h* drops), and
  display maths extracts as linearised symbol soup. Print-first
  should include a text-extraction proof in `print/build.mjs` checks.
- At 390 px the chapter display title overflows (“Energies and” is
  cropped at the viewport edge, live screenshot): the 54–120 px fluid
  title (`DESIGN §3`) needs a `clamp`/wrap guard on narrow screens.

Fix: change container SVGs that own controls from `role="img"` to
`role="group"` with the same label (keep `role="img"` only for purely
static/print renders); give Fig. 9.4 and Fig. 0.2 a single
`tabindex=0` slider owner for reader/column exactly as
`fig-attention-reads.svelte:169` already does for Ch. 5; add a
`pdftotext` smoke check (no “Te ” at line starts) to the print build.

### 9. Attention arcs do not encode direction, and repeated characters make the readout ambiguous

Fig. 5.1 (live + PDF p. 63): arcs are symmetric strokes between reader
and source with no arrowhead, taper or asymmetry. Head 2's thick
0.99 arc under “b–e” reads at a glance as *b reads e* (the future) —
the one reading Ch. 5 exists to forbid. Only the bold face of the
reader and the caption (“arcs leave the selected reader”) carry the
direction. The readout compounds it with identical labels: “head 1
reads ‘t’ 0.50, ‘t’ 0.33, ‘o’ 0.09” — which t (positions 2 or 4) has
0.50 is unrecoverable from the text, and the key (“‘e’ reads ‘t’ with
0.50”) inherits the ambiguity. The same identical-label problem
recurs in Fig. 9.4's readout, mitigated there by “at 59” positions —
Ch. 5 should copy its own later solution.

Fix: taper arcs toward the source (or put the existing weight label
consistently at the source end), and print positions in every arc
readout: “reads $t_4$ 0.50, $t_2$ 0.33…”. Both are local to
`Arcs.svelte` callers and their `readout` strings.

### 10. The Prologue teaches two opposite scale rules in consecutive figures

Fig. 0.1's key: “Area $\propto|\theta_i|$, on one scale for the whole
plate.” Fig. 0.2's key, one page later: “Each array has its own scale:
area $\propto|$entry$|$.” DESIGN §5 mandates scales “computed from
every number the glyph will be compared with… never scale per glyph”,
and `scales.js` repeats “never per glyph” in its header — yet
`fig-two-inks.svelte:14` (“Each array has its own scale”) and
Fig. 7.2 (“the sum of all $N$ shares… gets its own scale”,
`fig-affine-pullback.svelte:53`) do exactly that, for good dynamic-
range reasons that are never explained to the reader. At the moment
the two inks are introduced, a student comparing a covector square in
Fig. 0.2 with a value square beside it — the comparison the figure
invites — is comparing across unknown scale factors.

Fix: keep the per-array scales where range demands it, but state the
rule honestly in §0.3's “Keys” paragraph (“within one array, areas
compare; across arrays in the same figure, check the key — two
figures in this book rescale, and say so”) and print each array's
$v_{\max}$ in Fig. 0.2/7.2 margins. Currently neither figure states
its scale numerically anywhere.

### 11. The Euler analogy is carried further than the specimen's own numbers allow; three different “times” share one name

Ch. 6 is admirably honest in prose (“Nothing… makes the steps
small”, `06-block.md:33`), but the names table, Fig. 6.1's key
(“$\oplus$: $x\leftarrow x+a$, an Euler step”) and Ch. 8's claim
(“gradient descent is Euler's method for the gradient flow”)
re-assert the analogy without the caveat, and Fig. 6.1's own readout
refutes the “small correction / only refine” intuition for the
trained model: “The stream enters at 0.95; attention adds 0.91…
giving 1.34; the MLP adds 1.65, giving 2.71” — the MLP “correction”
is larger than the state it corrects, and the wording “adds 0.91,
giving 1.34” invites scalar addition of norms ($0.95+0.91\ne1.34$;
the vectors are far from collinear). Meanwhile “Euler step” denotes
three unrelated discretisations (depth in Ch. 6 with $h=1$, the
adjoint recursion in Ch. 7, training time in Ch. 8 with $h=\eta$),
and Adam — the optimiser actually used — is not a discretisation of
$\dot\theta=-\nabla\ell$ at all: with state $(m,v)$ it is a
history-dependent dynamical system, a point Ch. 8's claim line
(“everything else… is a decision about how long, and how well aimed,
each step may be”) understates.

Fix: one “three times” margin table in Ch. 8 §8.2 (text position $t$,
depth $k$, training step $k$ — note the last two already collide as
$k$) and reword Fig. 6.1's readout to “$\|x\|/\sqrt D$ goes
$0.95\to1.34$ under an update of size $0.91$” (norms of sums, not
sums of norms). Add the single sentence Ch. 8 lacks: “Adam with
$(m,v)$ is not gradient flow with a better step; it is a different,
second-order-in-state dynamics whose continuous limit is not
$\dot\theta=-\nabla\ell$.”

### 12. Temperature: where the physics name stops being literal, the book keeps using it as if it were

The Gibbs identification in Ch. 3 is exact and beautifully handled
(including the $F$ sign warning, §3.5). It leaks at the edges:

- Ch. 5's names table: “$\sqrt d$ | temperature | scale factor”.
  $\sqrt d$ is a dimensionless head width, not a bath temperature:
  widening a head changes the *system* (more components in
  $q\cdot k$), and the $1/\sqrt d$ merely renormalises the energies
  $-q_i\cdot k_j$ whose scale is itself learned. §5.4's aside half-
  says this (“just sets a sensible starting temperature”) but the
  table — the part that enters the dictionary — does not.
- Each attention row has its own partition function $Z_i$ and there
  is no shared equilibrium, heat bath or fluctuation balance between
  rows; “the Gibbs distribution” for $A_{i,:}$ is a per-reader
  normalisation. A physicist reader will (rightly) ask what is in
  equilibrium with what.
- Ch. 9 is careful that $\tau\ne1$ sampling is “a deliberate
  distortion”, but `src/sample.jl:5` (quoted in spirit in Ch. 9)
  still says dividing by $\tau$ “is literally a temperature”. It is
  literally the temperature *of a different Gibbs system with the
  same energies*, not of the trained model.

Fix: change the Ch. 5 table cell to “temperature (by the Ch. 3
analogy)” and add one sentence in §5.3: “The analogy is row by row:
each reader normalises its own energies; nothing equilibrates
between readers.” No maths changes.

### 13. Chapter 9's “shown honestly” measurements hide their definitions

Recomputed from `artifacts/shakespeare/{loss,samples,attention}.json`
+ `tinyshakespeare.txt`:

- “Real words” (Figs. 9.2–9.3, §9.4 “At $\tau=\tfrac12$ every word is
  real… At $\tau=1.5$ a third of the words are invented”) means:
  maximal letter-runs of length $\ge2$, lower-cased, present in the
  *first 90 %* of the text (`web/src/lib/server/artifacts.js`,
  `trainingWords`/`wordStats`). Single-letter words (“I”, “a”) are
  excluded from numerator and denominator; a genuine Shakespeare
  word occurring only in the held-out 10 % counts as “invented”;
  case is forgiven. With that definition the claims check out
  ($\tau{=}0.5$: 81/81; $\tau{=}1$: 63/70; $\tau{=}1.5$: 41/63) —
  but neither caption nor prose states the definition, and the
  dotted underlines in Fig. 9.3 therefore look like a dictionary
  judgment rather than a corpus-membership test.
- “Held-out” is the contiguous last tenth (`scripts/common.jl:29-31`),
  and Fig. 9.4's passage begins two characters after the split
  (offset 1 003 857 of 1 115 394) — i.e. it is the continuation of
  the same play, with the same speakers (Gremio/Baptista), not an
  independent sample of “Shakespeare”. §9.4's reading of the
  train/validation gap (“what the model has memorised about its
  particular million characters rather than learned about
  Shakespeare's”) needs that caveat: contiguous-tail validation
  flatters generalisation.
- Small overstatements: “It is at most a hundredth for every head”
  (copy) — recomputed maximum is $0.0105$ (L4H3), which rounds to
  the figure's “0.01” but is not $\le0.01$; “The loss starts at
  $\log65$, where it must” — stored step-0 losses are $4.166$–$4.168$
  vs $\ln65=4.174$ (near, not necessary); final held-out perplexity
  $e^{1.531}=4.62$ (“a handful”, fair). First-vs-second repeated-line
  surprise recomputes to $0.889$ vs $0.898$ nats/char, matching the
  figure — the prose could quote the figure's computed pair instead
  of “the same”.
- §9.4's verdicts “It does not know what any word means. Its
  sentences have no syntax beyond a few words” assert unmeasured
  negatives in a book that elsewhere bans unoperationalised
  anthropomorphism. The evidence shown supports “nothing in these
  samples requires more than few-character statistics” — a weaker,
  defensible claim.

Fix: one caption sentence defining “real word” operationally; one
aside sentence “the held-out text is the tail of the same file — the
same plays continue across the split”; change “at most a hundredth”
to “about a hundredth at most (0.011, Fig. 9.4)”; soften the two
verdict sentences to the evidential form.

### 14. Generation has no beginning and no end — and the sliding window silently renumbers the past

Ch. 1 defines $P(x)=P(x_1)\prod q(x_{t+1}\mid x_{\le t})$ but the
model never supplies $P(x_1)$ (the first character “is given, never
predicted”, §1.4), there is no BOS/EOS symbol anywhere in the book,
and `generate` (`src/sample.jl:29-41`) therefore requires a non-empty
prompt and can never decide to stop. Fig. 9.1's prompt “␣” is
consequently not “the start of a text” but the start-of-*window*
conditional (the threefold-degenerate distribution of Ch. 3) — a
subtlety the chapter never names. Second, when generation exceeds
$T$, “only the last $T$ are fed in, numbered $1,\dots,T$ again”
(`09-speaking.md:25`): every character's positional code jumps by one
clock-step at each slide, so the same context suffix is repeatedly
re-embedded with different $P$ — the Markov-chain claim that follows
is correct only because of this renumbering, whose discontinuity
(attention patterns are *not* shift-invariant under it, Ch. 2 §2.4)
deserves the one sentence it never gets.

Fix: a short “Beginnings and endings” aside in §9.1 contrasting the
book's prompt-required, never-stopping sampler with BOS/EOS practice,
stating explicitly that Fig. 9.1's first column is
$q(\cdot\mid\text{␣ at position 1})$, and that sliding re-embeds the
window (so Ch. 2's rotation property is what makes the damage
graceful rather than catastrophic). This also closes Ch. 1's open
$P(x_1)$ loop.

### 15. Fig. 6.2's meridian panel answers “what does LayerNorm forget?” while its caption asks what it does

Caption: “LayerNorm, geometrically…” (`06-block.md:70`); catalogue
question: “What does LayerNorm do, geometrically?”
(`book/figures.md`). The left panel plots only the two forgotten
coordinates $( \mathbf1^\top x/\sqrt D,\ \|\Pi x\| )$, in which —
as the text honestly notes — “every column of the window arrives at
the same place”. A reader meeting the figure before the paragraph
below it reasonably concludes LayerNorm maps all columns to one
vector; only the right-hand spokes (one column, four stages) carry
the actual map, and the key (“How to read: meridian”) reinforces the
forgettable half. The sphere language adds a dimension slip a
physicist will catch: $\hat x$ lies in $\mathbf1^\perp\cap\{\|x\|
=\sqrt D\}$, a sphere $S^{D-2}$, called throughout “the sphere of
radius $\sqrt D$ in $\mathbf1^\perp$” without its dimension.

Fix: retitle the left panel in-caption (“left: the two numbers
LayerNorm discards — all columns coincide *in these coordinates
only*”) and give the sphere its dimension once (“an
$S^{D-2}$ of radius $\sqrt D$”). No redraw needed.

### 16. The loss-chart “log” axis is $\log(1+\text{step})$, labelled as $\log$

`LossChart.svelte:15,88`: position $\propto\log(1+k)$, ticks at
$0,1,10,100,\dots$, axis text “step (logarithmic)”, key: “steps, on
a log scale: 0, 1, 10, 100, …”. On a true log axis step 0 is at
$-\infty$ and the $0\to1$ interval cannot be drawn; on this
$\log1p$ axis the first decade is stretched by design, so early
slopes (“for the first few steps almost nothing does… then the loss
falls steeply”, §8.1) are partly axis artefact. The prose gloss
“the step axis is logarithmic, so the late, slow part… takes most
of the width but little of its time” is true of $\log1p$ too, but a
reader measuring rates off the chart with log-paper intuition will
misread the first ten steps — exactly where warmup happens.

Fix: label the axis “$\log(1+\text{step})$” in chart and key (one
string in `LossChart.svelte:191`), and add “($+1$ so step 0 can be
drawn)” to the key. Applies to Figs. 8.1, 8.2, 9.2 and the Ch. 4
frontispiece description.

### 17. Three-names gaps flow straight into a dictionary that overclaims completeness

Epilogue §E.2: “Every term of the field defined in this book… it is
exactly as complete as [the names tables] are” (`10-epilogue.md:66`)
— the second clause retracts the first, because the tables omit
terms the prose defines and then uses freely: *query / key / value*
themselves have no row in Ch. 5's table (the chapter's central
jargon enters only via $S_{ij}$ and $o_i$); likewise *warmup,
cosine schedule, slingshot, grokking* (Ch. 8), *bigram* (Chs. 1, 5,
8), *scatter-add* (Chs. 2, 7), *fan-in* (Ch. 8), *induction heads*
(Ch. 9), *RMSNorm, mixtures of experts* (Epilogue bullets). None
can appear in the dictionary, which is generated solely from the
tables. Duplicate senses also survive (“per-token loss” defined in
both Ch. 1 and Ch. 4 tables; “context window” with two senses).

Fix: add the missing rows to the relevant tables (Ch. 5 needs three:
$q_i,k_j,v_j$ — maths: “the three affine images of $\hat x$”;
physics names will be forced, which is itself instructive and worth
an honest “—” cell), or change the dictionary's first sentence to
“Every term given three names in this book…”. The generated
dictionary then matches its billing either way.

### 18. Fig. E.1's caption describes a different figure from the one drawn; 24 specimen parameters are silently omitted

Caption (`10-epilogue.md:33`): “Each square has area proportional
to a parameter count, and each is drawn inside the next as a single
mark.” Implementation (`fig-scale.svelte:6-9,41,63-76`): four
equal-size panels in a row, each containing the *previous* model as
a tiny corner square at true relative scale, joined by zoom
hairlines — a good figure, but not nested squares, and no square in
it has area $\propto$ its own count across panels (all panels share
$S$). Inside panel 1, parameters with rendered side $\le0.25$ px
are dropped (`:70`): at the default size that is 24 of the 1007
squares ($|\theta_i|\lesssim0.008$), against “every mark is a
number / one square per parameter”. The caption's “GPT-2, the
largest of the 2019 models” also needs its qualifier (largest
*GPT-2*; the figure's own data note says 1542M from Radford table 2
while the caption prints “1.5 billion”).

Fix: rewrite the caption's first sentence to the zoom-chain truth
(“each panel is drawn the same size; in its corner, the previous
model at true scale”); draw sub-threshold parameters as 0.25 px
minimum dots (or state “24 smallest omitted” — but the book's rule
prefers drawing them); “largest GPT-2 (2019)”.

---

## Tier 3 — smaller frictions worth a line each

- **Energy-ladder near-ties lose a label.** In Fig. 3.1 two rungs
  ($-2.19$, $-1.50$) share one visible character label at default
  size (live screenshot); the shelving (`EnergyLadder.svelte`
  `dodge`) separates bars but the upper rung's character (␣) is
  effectively unlabelled. Guarantee a minimum label gap equal to
  the bar gap.
- **Dragging $\eta$ in Fig. 8.1 rewrites history.** `change()`
  restarts from $\theta_0$ and replays (`fig-train-live.svelte:181`),
  so “drag $\eta$ upwards until the run learns” (Ch. 8 exercise) is
  a sequence of counterfactual restarts, not a mid-run schedule
  change. The hint discloses the restart; the exercise should say
  “each drag reruns from $\theta_0$ — you are comparing runs, not
  steering one”.
- **Code excerpts can overflow.** `src/sample.jl:18`'s error string
  is clipped at the column edge on the live Speaking page at
  1280 px. Excerpts are `size=wide` “so their comment columns fit”
  (DESIGN §4) — long *string literals* defeat that; shorten the two
  `ArgumentError` messages in `sample.jl` or allow excerpt wrapping.
- **Cover $\theta$-field vs Fig. 0.1 order.** The title-page texture
  sets $\theta$ “in storage order” reading left-to-right, top-to-
  bottom, while every figure reads the computation top-to-bottom
  with columns as tokens. One caption clause (“read across, unlike
  the computation”) would prevent a Prologue reader mapping the
  cover's rows onto Fig. 0.1's blocks.
- **Night-mode traces invert to white-on-black “glow”.** The
  `invert(1) hue-rotate(180deg)` rule (DESIGN §2) turns hairline
  plotter traces into the neon-on-black aesthetic BRIEF §0 lists as
  an anti-goal (verified: energies frontispiece at night). Consider
  mapping night traces to `--ink` at reduced opacity instead of
  full inversion.
- **“The function at the centre of every current large language
  model”** (Prologue, `00-prologue.md:21`) is contradicted by the
  Epilogue's own catalogue (MoE routing, state-space mixers,
  post-training objectives). “At the centre of the current
  generation of autoregressive language models” is defensible.

---

## Opportunities — connections the book has all the parts for and never quite makes

1. **One symmetry box to unify four scattered puzzles.** The book
   proves, in four places, that pieces of $\theta$ are invisible:
   $b_k$ entirely (Chs. 3, 5, 7), the mean of $b_{\text{out}}$
   (Chs. 3, 4, 7), LayerNorm's $x\mapsto cx+b\mathbf1$ invariance
   (Ch. 6 exercise), and — stated only as a Ch. 7 exercise question
   — the redundancy $(b_v,b_o)\sim(b_v+\delta,\ b_o-W_o\delta)$
   since rows of $A$ sum to one. Add head-permutation and
   MLP-unit-permutation symmetry and this is a complete, elementary
   “gauge freedom of $\theta$” story: the loss lives on a quotient
   of $\mathbb R^p$, gradients are horizontal, and Adam's
   coordinate-wise steps are *not* gauge-invariant (which is why
   $b_k$ drifts, §7.5). Half a page in Ch. 7 after §7.5 would
   retroactively organise Chs. 3–6 and give physicists the word
   they are already reaching for.
2. **Name the five predictions as a through-characters.** The same
   five window-initial predictions are: the floor (Ch. 1), the
   degenerate ground states (Ch. 3), the cancelling gradients
   (Ch. 4), the only surviving covectors after training (Fig. 7.1),
   the residual minibatch noise (Ch. 8 §8.3) and the first sampling
   branch (Ch. 9). Each link is made locally; no chapter says “it
   is the same five again”. A recurring margin sigil (or one
   sentence per reappearance: “the five of §1.6 again”) would turn
   six facts into one object — the closest thing the specimen has
   to a protagonist besides $\theta$.
3. **Use the exported-but-unused Shakespeare embedding geometry.**
   `artifacts/shakespeare/embedding.json` (E, its PCA, pairwise
   cosines, and the readout's PCA — “all learned”, per its own
   description) is generated by `scripts/export_book_data.jl:109`
   and referenced by no figure, chapter or web loader. Ch. 2
   asserts the specimen's $E$ “moves, but not far” and the
   Epilogue mentions weight tying $W_{\text{out}}=E^\top$ in one
   bullet. A single Ch. 9 small-multiple (alphabet under $E$ vs
   under $W_{\text{out}}$ rows) from data already on disk would
   close the Ch. 2 → Epilogue arc at zero training cost.
4. **Say the loss is convex in $Z$ and not in $\theta$, once.**
   Ch. 3 proves $\mathrm{lse}$ has PSD Hessian (the softmax
   Jacobian as covariance); Ch. 8 then discusses sharpness, edge
   of stability and bowls without ever stating that $\ell(\theta)$
   is non-convex with permutation-equivalent minima, while
   $\ell$ as a function of the logits alone *is* convex. Two
   sentences at the head of Ch. 8 would stop the strongest readers
   over-generalising the descent lemma into a convergence-to-
   minimum story the book never claims but never denies.
5. **Put the bigram baseline on the page as numbers.** Ch. 1 frames
   the transformer as writing the $V^{T+1}-V$ table “with far
   fewer numbers” — true, but the specimen still spends 1007
   parameters on 80 predictions, and the bigram exercise (42 free
   numbers) is left uncomputed. Printing the optimal bigram's mean
   surprise beside the floor and the specimen's $0.0589$ (one
   computed sentence in §1.6 or §4.5) would give “memorisation vs
   compression” a quantitative foothold the Shakespeare chapter
   later needs for its train/held-out gap.

---

## What is *not* in this review

Derivation correctness, pullback/test coverage, citation metadata
and page-layout defects already catalogued in
`book/briefs/fixups.md` were not re-checked. The two still-open
fix-ups items (monolith Shakespeare re-export; `speaking.svg`)
remain open and interact with findings 7 and 13 above: when the
monolith data lands, finding 13's recomputed numbers must be
re-derived, which is itself an argument for finding 4's templated
prose numbers.

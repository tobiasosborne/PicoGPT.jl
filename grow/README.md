# Growing a transformer — exploration and prototypes

Branch `explore/growing`, started 24 November 2026. **Goal: quickly try a bunch of ideas.** This
is a scratch branch for experiments and derivations, not for the books. Speed and honest verdicts
matter more than polish.

If you are an agent starting here (locally or in a cloud session): read this file, pick up the
idea you were given (or the first unclaimed one in the table at the end), work in your own
subfolder `grow/<idea>/`, and follow "Rules" below.

## The idea

We want to *grow* a transformer, guided by renormalisation.

Take two transformers T1 and T2, with T2 bigger on any axis: vocabulary, embedding width,
context length, heads, MLP width, depth, number of passes of a looped block. We want two things:

1. **A lift** T1 → T2 that changes nothing at birth: the grown model computes exactly what its
   parent computed, and training then learns only what is new.
2. **A coarse-graining** T2 → T1, so that we can say precisely in what sense T1 is the
   coarse-grained version of T2.

The second principle: **the training environments grow with the model.** There is a tower of
worlds, each a coarse-graining of the next, and a model for each level. The hope, by analogy with
DMRG, is that one can grow scalably (longer context, larger vocabulary, wider stream) because each
refinement adds only a bounded amount of new structure.

### A working definition

Let π map a fine history to a coarse one (merge tokens, shorten the context, drop a modality).
T2 is a finer-grained T1 when pushing T2's predictions through π gives T1's predictions on the
coarse history, exactly or to a stated error. A stronger, internal version asks for a map R on the
residual stream with R∘F2 = F1∘R, block by block.

The worlds are not an add-on. Renormalisation needs something held fixed while detail is
discarded; here that is the coarse world's predictions (and, with actions, its returns). A
distinction is *relevant* if forgetting it changes a prediction or an action that matters one
level down.

### What is believed so far, axis by axis (all to be checked)

| Axis | Lift that changes nothing at birth | Coarse-graining back | Closes exactly? |
|---|---|---|---|
| Vocabulary | split token v into (v, b); predict p(v\|h)·q(b\|v,h) | sum over b | yes; the loss splits into the parent's surprise plus one new term |
| Depth, or passes of a looped block | new block or pass behind a gate that starts at zero | close the gate | yes |
| Heads | new head with zero output weights | drop it | yes |
| Width | duplicate coordinates (commutes with LayerNorm and ReLU) | average the copies | believed yes at birth, with one constant rescaled in the attention scores; unchecked |
| Longer context | mask the extra positions, extend the position code | truncate | yes on short histories |
| Blocked positions | none known | merge a block B into one position | no: needs the score log Σ_{j∈B} exp(s_j), which is not bilinear in query and key |

The last row is where coarse-graining generates couplings outside the family, as real-space
renormalisation does.

### The DMRG reading

- White's criterion: the block alone cannot say which of its states to keep; the rest of the
  chain decides, through the reduced density matrix. Here the world, and the layers downstream,
  decide which distinctions the model keeps.
- DMRG scales because of an area law (bounded bond dimension). The analogue is a hypothesis about
  the tower of worlds: each refinement adds a bounded amount of new relevant structure.
- DMRG has a canonical form and an optimal truncation by SVD. Transformers have neither. Finding
  the coarse-graining R is the open problem; a candidate is to keep the stream directions the
  output is sensitive to (compare `jspace/`, the Jacobian lens).
- Growing site by site is only DMRG's warm-up; accuracy comes from sweeping back. Freezing the
  parent and training the child is the warm-up alone.

### Evidence already in hand (from an earlier outside project, small scale)

- A frozen 1,273-parameter two-pass transformer for motion on a line gained a second sense
  through two new weights fitted on eight examples (reproduced exactly on 24 November).
- A shell policy gained a third pass through a scalar that starts at zero; the extra pass barely
  helped (71.1 % against 70.0 %), and training from scratch on the whole budget reached 93 %.
- Whole-lineage cost was about equal to a joint fit. **The claim that growing is cheaper than
  training the final model directly is open, and so far unsupported.**

Prior art to check, cited from memory and not verified: function-preserving growth (Net2Net,
Chen, Goodfellow and Shlens 2015; Gesmundo and Maile 2023 for transformers; bert2BERT; LiGO),
lumpability of Markov chains (Kemeny and Snell), open-ended environment generation (POET).

## The repository in one minute

PicoGPT is a GPT written from scratch in Julia with no autodiff: every operation returns its
value and its pullback, and all parameters live in one vector θ (`src/`, tests in
`test/runtests.jl`, `julia --project -e 'using Pkg; Pkg.test()'`). Only the standard library is
needed. "The specimen" is the 1,007-parameter model trained on `to be or not to be` (V=7, D=8,
H=2, F=32, L=1, T=8), whose loss has an exact floor, (2 ln 2 + 3 ln 3)/80 nats. `BRIEF.md` is the
spec and notation. `web/src/specimen/` is a JavaScript twin of the specimen that runs in Node.
`scripts/train_shakespeare.jl` trains the 810,049-parameter model (about 36 minutes on 12
threads; its checkpoints are git-ignored and exist only on the author's laptop). The branch
`t3/prototype-looped-transformer` holds a looped specimen (`loop/`) with analysis of the gate and
of passes.

**In a cloud session** Julia is not preinstalled and the default network rules may block
julialang.org. Try, in order: `julia --version`; `docker run --rm julia:1.11 julia --version`;
installing with juliaup. If none works, prototype in Python with NumPy, with hand-written
pullbacks checked by finite differences, or use the JavaScript twin. Say in your result which you
used.

## Rules

- Work only inside `grow/<idea>/` (code, data, notes). Do not change `src/`, `book/`, `web/`,
  `print/`, `jspace/`. Copy what you need into your folder.
- Every number comes from a script in your folder that reruns in under a few minutes. Every
  gradient you write is checked by finite differences. Every "exact" claim is tested to rounding.
- Write `grow/<idea>/RESULT.md`: what you tried, the numbers, what is verified and how, what
  failed, and a one-line verdict: **promising**, **dead**, or **needs more**. A clean negative
  result is a good result.
- Commit early and often, and push after each real result, because the machine you run on may go
  away at any time. Push only your own branch, by name:
  `git push origin HEAD:refs/heads/grow/<idea>`. Never `git push --all`, `--mirror`, or tags.
- Do not put private material in the repository: no chat transcripts, no personal data.
- Small results files only (under 1 MB each).

## Ideas

| Idea | Folder | Kind | What to find out |
|---|---|---|---|
| Exact lifts on every axis | `grow/lifts/` | prototype | Implement each lift for PicoGPT; test that the logits are unchanged to rounding; then grow the specimen (for example D 4 → 8) and compare total work to reach the floor with training at full size from scratch. |
| A tower of worlds | `grow/tower/` | prototype | Coarse-grain the specimen's seven characters to three classes; compute both floors exactly; train T1 on the coarse text, lift the vocabulary, and test the consistency condition and the split loss. Then a synthetic hierarchical world whose new structure per level can be dialled, staged against scratch. |
| Coarse-graining back | `grow/truncate/` | prototype | Given a trained model, find a smaller one: truncate the stream by activation covariance and by output sensitivity; plot loss against kept dimension. Is there a spectrum, and does it fall fast? |
| Definitions and exact statements | `grow/notes/definition.md` | derivation | Make the definition precise; prove or refute each row of the table, LayerNorm and the √d in the scores included; say when an exact coarse-graining exists (lumpability). |
| Attention under blocking | `grow/notes/blocking.md` | derivation | The log-partition score: cumulant expansion, the generated couplings, error bounds, when one key per block suffices. |
| The DMRG analogy made precise | `grow/notes/dmrg.md` | derivation | What plays the reduced density matrix; the optimal truncation for a linearised model; the "area law" as a statement about the tower; when staged training provably costs less. |
| Longer context | `grow/context/` | prototype | Train at T=8, extend to 16 and 32; what is inherited, what must be learned. |
| A world with actions | `grow/doors/` | prototype | The directory world: `Ada → red door`, refined to `Ada → oak`, `oak → red door`. Does a child with one more pass reuse the parent's lookup? |
| Prior art | `grow/notes/prior-art.md` | reading | Verify the citations above and find what already exists on each axis. |

## Growth need not be exact

Added on 24 Nov 2026 after the first results, at the owner's prompting. In evolution a change
does not have to leave the organism unchanged; it has to leave it viable, and selection does the
rest. The same holds here. A lift is good when the child at birth is *close* to the parent and
training closes the gap quickly; a coarse world is good when it is *nearly* closed. Exactness is
the special case with gap zero. It remains useful as a test of the bookkeeping and as a reference
point, but it is not a requirement, and the first results say it can even be a poor place to
start:

- Exact duplication of width is a symmetry that training never leaves (`grow/lifts`); breaking it
  with noise, which gives up exactness, is what brings the child level with scratch
  (`grow/hierarchy`).
- A gate that is exactly closed has zero gradient on the end-only objective (proved for the
  looped specimen).
- No coarse-graining of real text is exactly closed (`grow/lumpable`), and no truncation is
  exactly lossless below full rank, yet fifty training steps recover a cut to about rank 5
  (`grow/truncate`).

So measure, do not demand: the loss at birth above the parent's, the work to recover, and for a
world its defect in nats. The questions become quantitative: how large a gap at birth can training
absorb cheaply, and which approximate lifts leave the child best placed to learn?

## Access is earned (the rolling-ball picture)

The owner's image, 24 Nov 2026: a ball that rolls through one world and picks up what is small
enough to stick; as it grows, things that were obstacles become things it can pick up, and the
view zooms out. Read as a design for the training environment:

- **One world, a moving cutoff.** The levels of the tower are not separate environments. The fine
  world is there from the start; the model sees it through a coarse-graining that relaxes.
- **Competence sets the reach.** A degree of freedom (a class split, a stream dimension, a pass, a
  context position) opens when the model has earned it, not on a schedule. Candidate signals: the
  loss on what it already sees is near its floor or has stopped falling; or the model pushes
  against a nearly closed gate (a large gradient on it), the analogue of bumping into something
  still too big.
- **Growth is local.** The ball picks up single objects, not whole layers. Split the classes the
  model already predicts well and leave the others; lengthen the context where the full context is
  already used; add a pass for the inputs on which the last pass still changed the answer.
- **It is approximate.** Things stick imperfectly and can be knocked off: see the section above.

All experiments so far used global lifts on a fixed schedule. Untested: gating by competence, and
local refinement. A first test: the synthetic tower of `grow/hierarchy`, with a coarse level that
is expensive to learn, where each class is split when its own class-level loss comes within a few
per cent of its floor; compare work against the fixed schedule and against scratch.

## Status (24 Nov 2026, 15:50 UTC): first round finished

Thirteen ideas ran, each on its own branch `origin/grow/<name>` with a `RESULT.md` (`git fetch
origin`, then `git show origin/grow/<name>:grow/<folder>/RESULT.md`). Nothing is merged. Read
the results before starting anything; a new idea goes in a new folder `grow/<new-idea>/`.

| Branch | Verdict | The one thing to take away |
|---|---|---|
| `grow/notes-definition` | promising | Exact at birth: vocabulary split, MLP width, gated passes, identity blocks, context on old prefixes. Refuted: more heads at fixed D; width by duplication with the stock position code. Closed coarse world = predictive sufficiency (strong lumpability for Markov chains). |
| `grow/notes-blocking` | needs more | A block of positions is replaced exactly by (Σ e^score, Σ e^score·value); one ordinary key per block is exact only if the keys coincide. |
| `grow/notes-dmrg` | needs more | White's rule has a local counterpart (weighted PCA) only when covariance and sensitivity separate; no exact two-site update; a frozen parent can leave a gap that one inherited bias closes. |
| `grow/notes-prior-art` | reading | Lifts exist as function-preserving expansion (Net2Net, Gesmundo and Maile, LEMON); not found: the paired map back, a tower of worlds with the model lifted along it. Abstract-level verification only. |
| `grow/lifts` | needs more | All exact lifts implemented and tested to 1e-15. Duplicated width is a symmetry training never leaves. Growing width on the specimen costs 1.0 to 2.4 times scratch. |
| `grow/tower` | needs more | Vocabulary lift exact; on the specimen text staging costs 3.4 times scratch because the coarse text is the harder one. |
| `grow/hierarchy` | needs more, leaning negative | The lifted child beats scratch only when a class's children are interchangeable as inputs; a defect makes the parent expensive. |
| `grow/lumpable` | needs more | No coarse-graining of the specimen text or of Shakespeare is closed; a Brown-style tower adds about ln 2 of new structure per doubling. |
| `grow/truncate` | needs more | Spectrum falls slowly; cuts need retraining; about rank 5 suffices at D=8 and D=16 after 50 steps; the internal intertwiner fails. |
| `grow/blocking-loss` | needs more | Blocked context loses to a plain window on Shakespeare at T=32 and wins where evidence is spread over the whole context. |
| `grow/passes` | inconclusive | Model rewritten without gate or start mark; neither staged nor scratch trained. Redo on the looped specimen behind a gate. |
| `grow/doors` | needs more | One more pass behind a zero gate: children learn the two-hop world in 7 of 18 runs, scratch in 1 of 9; the parent's lookup is not handed over in query form. |
| `grow/context` | promising | T=8 then T=32 reaches the floor in about 1.7M tokens; scratch at T=32 sits on a plateau to 5M (two seeds, one world). The T=8 model fails beyond position 8 untrained: the absolute position code. |

Where staging paid (context, doors) the child had to find something scratch gets stuck before
finding. Where it did not (width, vocabulary) the coarse stage was no cheaper than the fine one.
Untested: approximate lifts, growth gated by competence, local refinement (the two sections
above).

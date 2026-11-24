# Growing in passes while the world lengthens — result

**Verdict: dead in this form (ungated extra pass, 270 parameters); needs more before the idea itself is judged.**
Neither regime reached the floor at 7 passes on length 8, so the work comparison asked for in
question 1 does not exist. What the runs do show is that an *ungated* extra pass is a bad birth.

Written 24 Nov 2026, 15:40 UTC, under a 40-minute limit. Julia 1.13, standard library only.

## Setting

`passes.jl`: one block with shared weights applied k times (V=2, D=4, one head, F=16, 270
parameters, no position code), attention over the position and its predecessor, final LayerNorm
and unembedding. Task: prefix parity over {be, not}; full batch of all 2^T sequences; Adam, learning
rate 0.01. Written afresh, not copied from `loop/`. Growing = one more pass and one more position,
same weights, **no gate**.

**The floor is not zero.** Without a position code a first position is identical to a position
whose predecessor is in the same state, so `not`, `not not`, `not not not` at the end of an
all-`not` prefix can never be told apart, though their parities differ. `floor_exact(T, k)`
computes the floor by refining a colouring k times (colour ← own colour and predecessor's) and
taking the entropy of the target given the colour. Floors in nats per position, k passes on
length k+1: 0.23869, 0.25406, 0.27429, 0.28159, 0.28832, 0.29234, 0.29569 (k = 1..7).
"Reached" means loss ≤ 1.01 × floor.

## Numbers

**Question 1 (7 passes, length 8), five seeds each** (`staged.txt`, `scratch.txt`):

| Regime | Cap | Seeds reaching the floor | Final loss / floor |
|---|---|---|---|
| Staged, k = 1..7 | 2,500 steps per stage | 0 of 5 | 2.04 to 2.32 |
| Scratch | 4,000 steps | 0 of 5 | 2.03 to 2.39 |

Work spent without success (block applications × positions × steps): staged about 401,000 to
415,000 per seed, scratch 224,000; times the 2^T sequences of the batch, 56.2 million and 57.3
million, so the two regimes had equal budgets in that measure. In the staged runs stage 1 was
reached by 5 of 5 seeds (100 to 167 steps), stage 2 by 4 of 5 (118 to 324 steps), stage 3 and
every later stage by none.

**Short tower with a long cap** (`probe3.txt`: up to 3 passes on length 4, cap 20,000 steps per
stage or run, five seeds):

| Regime | Seeds reaching the floor | Work of the successes |
|---|---|---|
| Staged 1 → 2 → 3 | 1 of 5 | 58,000 |
| Scratch, 3 passes | 2 of 5 | 38,700 and 78,012 |

So the floor at 3 passes is reachable by this model (three runs sit at 1.010 × floor), but most
runs stall at 1.4 to 2.1 × floor, and staging did not help: fewer successes, and no less work.

**Question 2. Is each stage's start a good one? No.** Loss on the new length at the moment of
growing was 2.1 to 9.0 × the new floor in the main run (up to 17 × in the probe), with accuracy
0.27 to 0.79; a random start gives 2.4 to 4.6 × floor. This holds also when the parent had
reached its own floor (stages 2 and 3: 3.0 to 10 × floor at birth). The parent never solved the
longer world with the extra pass. The one stage-3 success needed 4,708 steps after parents that
took 167 and 195: the stage start cost more than it gave. Reason, read off the construction: the
extra pass is applied at every position, including those the parent already answered, so the
parent's answers are destroyed at birth. This lift does not satisfy "changes nothing at birth".

**Question 3. Reverse map: not recovered.** The final models run with k passes on length k+1
give loss / floor between 1.7 and 50 at k = 1 and 2.0 to 13.6 at k = 2..6. This says little,
because those final models had not learned the task at 7 passes either. The reverse map of a
*solved* 7-pass model was not reached.

## Verified, and how

- Gradient: finite differences on all 270 parameters (T=4, 3 passes), worst relative error
  2.0e-7; printed at the top of every run.
- Floor: the colouring argument gives a lower bound by construction. It is attained to within 1%
  at 1, 2 and 3 passes (training loss approaches it from above and stops there, e.g. 0.23870
  against 0.23869 at one pass after 3,000 steps). At 4 passes and more its attainability by a
  D=4 model is **not** verified.

## Not reached, open

- No run solved 7 passes, so there is no staged-against-scratch work ratio. Unknown whether the
  cause is optimisation (plateaus near 2 × floor, seen already at 3 passes) or capacity at D=4.
- Only one learning rate was used for the results (0.03 was tried first and was worse at one pass).
- The obvious next step: grow behind a gate that starts at zero (the README's lift), so old
  positions keep the parent's answers and only the new position must be learned; and give the
  model a start-of-sequence mark or position code so the floor is zero and the general rule
  (one XOR per pass) is expressible. With five seeds and 20,000-step caps the short tower runs
  in two minutes.

## Rerun

    julia passes.jl staged 5 2500 > staged.txt     # about 10 minutes on one thread
    julia passes.jl scratch 5 4000 > scratch.txt   # about 12 minutes
    julia probe.jl 3 5 20000 > probe3.txt          # about 2 minutes

The two long runs exceed the "few minutes" rule; the probe does not.

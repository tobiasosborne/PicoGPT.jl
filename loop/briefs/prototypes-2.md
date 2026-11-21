# Brief: prototypes round 2 — make the loop *trainable*

## Where we are

Round 1 (your own work: `loop/notes/prototypes.md`, `loop/briefs/prototypes-report.md`,
`loop/src/`, `loop/scripts/`) established the machinery and one sobering fact: with the stock
block and the BRIEF §13 recipe, **training from random initialisation does not find a durable
recurrence**. Prefix parity at T=5, K=5 is fitted (once) but the front stalls at position 4 and
does not extend to more passes or longer inputs; hand-initialised "local" references work but
do not survive fine-tuning. Astra's mathematics is in `loop/notes/derivations.md` (read §1, §2, §5,
§6, §7): an exact stock-block previous-token head for 64 bits exists, but it needs score sharpness
~10³, a large carrier, and a *register refresh* update X_{k+1} = U + C[B(X) − X]; the plain
residual loops have a Jacobian eigenvalue 1 from LayerNorm's invariance to adding a constant
column. A book specimen must **train from random initialisation in seconds, reliably across seeds,
and then work beyond its training length and loop count**. Finding a setup where it does is the
whole job of this round. Changes to the stock block or loop are allowed if they are small, can be
stated in one sentence, and are reported as such; the fewer the better.

## Hypotheses to test (diagnose first, then fix; record negative results)

1. **The task at T=5 is too small to need the loop.** 32 words, and full causal attention can read
   every bit at once, so the optimiser finds counting shortcuts. Train at larger lengths (n up to
   12–16, mixed lengths per batch) where one pass provably cannot keep up, with the number of passes
   tied to the input: K = n (Fan et al. 2024, arXiv:2409.15647, trained looped models with the loop
   count set per example), and optionally K = n + random extra passes so that the right state must
   be a *fixed point* (stability beyond the trained count; Anil et al. 2022 "path independence").
2. **A length/pass curriculum** (grow n and K together during training).
3. **The stream drifts** (additive injection grows |X| linearly; LN hides it until it does not).
   Compare post-addition, pre-addition, and a refreshed/normalised state (e.g. Astra's register
   refresh, or LN of the state each pass), and Loop-μP-style 1/K scaling of the residual update.
4. **The previous-token head is hard to learn with D=8 sinusoids.** Check whether trained models
   learn previous-token attention at all (inspect A). Try D=16, or a causal *relative* bias, or
   NoPE, and say what helps. A specimen of up to ~3000 parameters is acceptable if that is what it
   takes; smaller is better.
5. **Supervision.** Final-pass vs deep supervision vs deep supervision only inside the light cone
   (at pass k, score positions t ≤ k; honest because nothing else can be known yet under a linear
   front, but say that it presumes the speed).
6. **Optimiser.** More steps (thousands are fine if each takes milliseconds), learning rate, batch
   size, weight decay. Keep it to AdamW.

Do the same for **rule 60** (autonomous; train on k-step targets, K random in a range) once P1
works, and check its rows against Lucas' theorem beyond the trained K.

## Success criterion and what to report

A configuration is a *specimen candidate* if, from random initialisation, **at least 4 of 5 seeds**
reach ≥ 99 % at the training lengths and stay ≥ 99 % at **4× the longest training length with
K = n passes**, and running extra passes (up to 2n) does not degrade it; training ≤ ~60 s per seed
on this machine. For every candidate and the best failures: parameter count, the exact change to
the stock block/loop, training curve, accuracy per position per pass (the front: linear or
doubling?), behaviour with extra passes, attention patterns (is it a previous-token head?), and
whether the trained state converges to a fixed point (|X_{k+1} − X_k| per pass). Then the controls
at the same setting: one pass (same parameters), untied L = K, wide one pass; one sentence each on
why they fail, with numbers.

## Rules

Own the same paths as before (`loop/src`, `loop/test`, `loop/scripts`, `loop/artifacts`,
`loop/notes/prototypes.md`, append a "Round 2" section). Keep the tests passing; any new loop
variant gets a gradcheck test. Do not edit `src/`, `book/`, `web/`, `jspace/`, `loop/notes/derivations.md`
or `loop/scratch/`. Do not commit. Do not start Muse or Gemini agents yourself. Use your time on the
search, not on prose. Report back (copy to `loop/briefs/prototypes-2-report.md`): candidates found
(or not), the setting that made the difference, the numbers, and your recommendation.

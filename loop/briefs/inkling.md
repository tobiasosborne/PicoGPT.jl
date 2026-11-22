# Brief: a short source note on Inkling 2 Mini (Muse)

Plain-text copies of four sources are in `/tmp/claude-1000/-home-tobias-Projects-PicoGPT-jl/ffbe9658-9762-4fc0-a2bf-fb9b7fd59bdd/scratchpad/web/inkling/` (`page.txt`, `goodfire.txt`, `varloop.md`,
`mini.md`): the Thinking Machines announcement (https://thinkingmachines.ai/news/inkling-2-mini/),
Goodfire's interpretability write-up (https://www.goodfire.com/research/interpreting-inkling-2-mini/),
and the model cards of the released model and its variable-loop research variant.

Write `loop/notes/inkling.md` (about 800–1200 words): what Inkling 2 Mini and its research variant actually
do (architecture: blocks, which run once and which loop, shared weights, the message and its gate with the
exact formula if given, decoding, pass counts in training and at test, what improves with passes and what
does not, the stated motivation against chain of thought, any numbers), with links to the originals. Verify,
do not trust, this summary of them by the advisor: "24 layers in four blocks of six; the first and last block
run once, the middle two run three times with shared weights; every pass restarts from the first block's
output; passes two and three also read the previous pass's output (the message) through a learned gate that
starts closed (x' = κ⊙x + γ⊙m, κ = 1 − tanh(Fz), γ = tanh(Gz)); decoded once, after the last block; three
passes, fixed; the research variant trained with 1–5 passes (ratio 1:2:6:2:1) and runs at ten without
degrading; chained table lookups (N-hop lookup) improve to ten passes, factual recall is flat; written
reasoning fell from 3,703 to 1,661 tokens between one and five passes." Say which parts the sources confirm,
which they contradict, and which they do not state.

Then a row-by-row comparison with our specimen (read `loop/CONCEPT.md` §1, §2, §6): shared weights, fresh
input each pass, normalisation/gate, readout per pass vs once, pass counts, extrapolation, task, what passes buy.

Paraphrase; quote only single short phrases; do not copy the pages into the repository. Write only
`loop/notes/inkling.md` and a three-line summary in `loop/briefs/inkling-report.md`. The laptop is on
battery: no heavy computation. Do not commit. Do not start any subagents. Do not change system settings.

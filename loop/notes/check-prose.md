# Mathematical review of the revised prose

I checked the two prose samples and the 22 November additions to the concept against the [analysis](written-vs-looped.md), its [independent review](check-wvl.md) and the [Inkling source note](inkling.md). For the selection screen and fresh seeds I used the [window study](window.md#fresh-seeds). The [earlier concept review](review-concept.md) identifies the selected-seed diagnostics and the scope of the Rule 60 control.

## Corrections in the prologue

[00-prologue-sample-2.md](../book/00-prologue-sample-2.md)

| Old statement | Corrected statement | Reason |
|---|---|---|
| Each pass moves news one word and the dependence has the listed support. | News can move at most one word. Possible inputs stop at the left boundary. | A mask bounds reach. Particular weights can discard a visible input. |
| The variant trained on one to five passes and extra passes bought serial work rather than stored facts. | Variable counts were sampled in later midtraining. The reported gains are on N-Hop Lookup, with little change on the measured recall tasks. | Factual QA scores measure performance on those tasks. The variant was post-trained at three passes. |
| A fixed first-word marker replaces sinusoidal positions in the working recipe. | The marker replaces positions and only the first word seeds the stream. | The reachability argument and the measurements use that initial condition as well as the marker. |
| Single-pass attention cannot compute parity at growing length. | A fixed-size, one-layer transformer cannot compute parity at all lengths. | The asymptotic sensitivity bound fixes the architecture. [Kozachinskiy, Steifer and Wałęga](https://arxiv.org/abs/2602.05896) |
| The exact loss belongs to a model that is right inside the cone. The figure has blank paper outside it. | The available words determine labels inside. The loss approaches the floor as correct probabilities tend to one and unresolved forecasts tend to one half. Only the optimal forecast is necessarily blank outside. | A correct finite argmax can have positive CE. Identical visible inputs require identical forecasts but do not force calibration. |
| The two mixed floors accompany a range of lengths and passes and the optimiser settings. | The averaging is uniform over lengths, then over the allowed horizons at each length, with mean supervision over passes. The update settings describe the screen. | The weighting determines the population objective. Optimiser settings describe how closely the trained models approached it. |
| Four seeds are exact at lengths 32 and 64, at 5⅓ times the trained length and depth. | They are exact on the test words, at up to 5⅓ times the maximum trained length and depth. | Each length had 256 test words. The full ratios are 64/12 and 128/24. |
| 25 of 30 biased seeds meet the 1% loss criterion. | 27 meet the loss criterion. 25 also decode every tested rollout exactly. | The 65–94% interval belongs to the joint outcome. |
| The predecessor weights and centred steps describe the trained block generally. Transformer settling is measured. | The quoted numbers belong to the selected seed. The measurement is the step between passes. | Other seeds have different attention weights. A small nonzero step does not establish a fixed point. |
| Writing and looping share an error law. | The law concerns independent flips on a single path. | The soft block can combine several paths. |

## Corrections in the written-feedback chapter

[sample-written-vs-looped.md](../book/sample-written-vs-looped.md)

| Old statement | Corrected statement | Reason |
|---|---|---|
| Pairing opposite labels forces a uniform forecast, zero logits and loss ln 2. | The minimum population terminal CE is ln 2 when the start is invisible. Equal logits produce the minimizing forecast. | Given visible context and forecast q, the conditional loss is −½ log(q(1−q)). Any common logit offset yields the same probabilities. |
| The final observations are the last two or K+1 words, or a prior written answer. The first word reaches column K+1. | The suffix lengths are upper bounds, and the start can reach that column. At T=2 the first pair uses the supplied first word. | Reach is a property of allowed paths. No intermediate token has been written at T=2. |
| Free decoding is proved exact, and all rows are achieved. | Getting the four XOR pairs right proves exact free argmax by induction. The table gives population CE infima approached by the witness. The exterior diagnostic is an absolute gap. | Finite correct margins suffice for the emitted bit. Zero CE requires a limit, and either logit can be larger on an unresolved input. |
| Writing saves columns and the loop preserves transcript length or emits zero. A register can trace the diagonal. | Column savings start at T>2. The dense loop emits zero scratch tokens and one final answer. A predecessor-only register can trace the diagonal. | At T=2 both schedules evaluate two columns. All schedules emit the final answer. The soft block also reads self. |
| Both exact schedules execute the same serial chain and share an error law. | The dense schedules have the same serial depth. The product law applies to independent flips on the written chain or an ideal predecessor-only diagonal from an exact seed. | Equal call counts do not give a dense block a single noisy path. |
| Transition indices also denote word positions. Emitting the first prefix adds a noisy edge. | The error recurrence indexes the state after each transition. A separately emitted seed adds an edge to the noise count if its emission is noisy. | With an exact first bit, transition one processes the second word. A deterministic copy adds no noise. |
| The physical correlation is given without specifying when the intended sign is favoured. Every interior rate damps the signed factor exponentially. | For fixed words the intended edge sign multiplies 1−2ε. It is favoured below ½, tied at ½ and disfavoured above ½. Rates zero and one are deterministic limits. The magnitude decays for interior rates and vanishes after one step at ½. | The physical coupling changes sign at ½. Above ½ the error-sign expectation alternates with the path length. |

## Corrections in the concept

[CONCEPT.md](../CONCEPT.md) · Locations below refer to the 22 November revision.

| Location | Old statement | Corrected statement and reason |
|---|---|---|
| Headnote | Numbers pending independent check. | The independent analysis review is complete and linked, and its corrections apply here. The closing source reference names the analysis and Inkling note explicitly. |
| §§1, 5, 7 | Each pass closes one word of distance, sees k+1 words or takes the covector one word back. The written block is called once per word. | Each local call can move information or a cotangent at most one word. The written schedule calls once per fresh pair, with the first call processing two words. |
| §§1, 5 | Optimal losses and a witness attaining the infimum. The written row always sees a written parity. | The table gives population CE infima. The witness approaches the losses while decoding exactly. At T=2 the prior parity is the supplied first word. |
| §§1, 5 | Looping keeps the original transcript or writes nothing, and writing saves columns. | Looping writes the final answer. Writing saves dense columns for T>2, with equal work at T=2. |
| §§1, 5 | Both paths obey the flip law, and 1−2ε is the physical edge correlation. | Independent flips from an exact seed give the product law on the written path and first predecessor-only diagonal. The physical correlation includes the intended sign. Its preference changes at ½. |
| §§1, 5, 6 | g=1 exactly reproduces the old recurrence and closure is an identity. | The gain-one equality is algebraic for the same stream normalization. Literal subtraction followed by addition can round. Closure recovers the one-pass predictor because the old-state map is constant. |
| §5, Negation | One pass of attention fails asymptotically, and 0.4975 is the exact floor. | The parity result fixes model size. The one-pass floor on the training mixture rounds to 0.4975. |
| §5, Two numbers | Both gains learn in a second. | The five recorded update loops took 1–2 seconds. |
| §5, Pulling back | Sum the same transposed map at every pass. | Sum the parameter cotangents from each call of the shared block. Its Jacobian is evaluated at the saved state of that call. |
| §5, Learning to loop | Deep supervision and the sampling schedule make the right state a fixed point. | They reward correct later readouts. The state can stall, settle on a wrong answer or drift. Readout supervision imposes no equality of successive states. |
| §5, Rule 60 | A one-pass model asked two horizons has floor (7/16) ln 2. | This is the whole-word, horizon-blind floor on fair length-eight words with equally weighted passes 1 and 2. Those choices fix the fraction of conflicting labels. |
| §5, Shakespeare and epilogue | The restart goes through the gate. The Inkling result generalizes to stored knowledge and shorter reasoning. The 27,980-parameter construction gives log depth without a length qualification. | The gate scales the previous core output. The Inkling claims concern measured recall, N-Hop Lookup and a selected GPQA subset. The stock doubling parameter count covers four bits with a preparation pass. |
| §6, pullback | Send the seed cotangent through the initial-stream map before the embedding scatter. | Add that contribution to the direct anchor cotangents from every call before the scatter. The embedding enters by both routes. |
| §12, Inkling | Read out after pass three. Every looped layer mixes previous-pass sums. The rationale for middle looping is stated as a finding. | Read out after the last block. The cross-pass sources enter on passes two and three. The abstraction rationale is the authors' intuition. The 27-of-28 comparison is their measured result. |
| §12, historical comparison | All three Shakespeare results came from 215,233-parameter runs. Passes buy computation rather than knowledge. The released pass count is set externally. | Only the two single-block runs have 215,233 parameters. Book 1 has 810,049. The Inkling paragraph reports the observed benchmarks. The release has a fixed three-pass schedule, while the variant supports a choice. |
| §13, timings and order | About 10 s per seed and 72–90 min for prior runs, presented as elapsed times. Runs 1–3 follow chapter order. | The seed timing covers updates. The 72–90 min range includes update and wall measurements. The stock tied wall-time comparator is 75.1 min. Book 1 logged 36.2 min of updates. The gate and distance experiments appear in a different order from their chapters. |
| §13, runs 1 and 4 | Matched sentences and seeds describe the comparison, which decides the trained block's behaviour. Matched applications are reported without their token ratio. | Gold training pairs supply an extra parity input to the written model. Free-rollout results apply to the tested seeds. At equal batch shape, 2,500 eight-application updates match 5,000 four-application updates and use half the training tokens. |

## Statements checked and found correct

- The parity encoding, sign recurrence and both Hamlet examples agree. The three written calls on `not not be not` emit `be`, `be`, `not`.
- A width-w causal stack with L layers repeated K times reaches at most KL(w−1) columns left with pointwise injections.
- Outside the cone, flipping the first fair bit pairs opposite labels. Exactly max(T−k−1,0) prefix sites remain unresolved at pass k.
- At T=K=8 the mean prefix floor is 21 ln 2/64 ≈ 0.227438919 nats. The nested uniform training mixture gives 0.161350952 nats and its one-pass control gives 0.497544536.
- The one-pass prefix accuracy ceiling at T≥2 is ½+1/T, or 53.1% at length 32 and 51.6% at length 64.
- Blank terminal CE infima are [T>2] ln 2 and [T>K+1] ln 2. Written feedback has infimum zero at every length. One call suffices at T=1.
- For T≥2 both minimum-depth exact schedules make T−1 dependent calls. At T=8, writing uses 14 columns and six scratch tokens, and dense looping uses 56 columns and zero scratch tokens.
- The T=4 figure uses three calls and two written intermediates. Its curves over T=2…12 count columns and scratch separately. Returning requested prefix tokens incurs no additional scratch charge for those outputs.
- The 64-position, 128-pass plate uses area 2|p−½|. This is zero at indifference and one at certainty.
- The D=4, H=1, F=16, V=2, L=1 specimen has 244 block parameters and 26 for the embedding, final LayerNorm and readout. The boundary marker, +4 bias and fixed stream LayerNorm add none.
- The independent exhaustive witness check covered lengths 1–8 and passes through T+1. The inside-margin/outside-gap extrema round to 7.03532/0.099995 at M=100 and 70.70679/0.00999995 at M=10,000.
- The unit LayerNorm bound in the proof is on coefficients in its centered orthogonal basis, e.g. aᵀX/4. No target text asserts that every raw normalized feature has magnitude at most one. [Witness review](check-wvl.md)
- In the 1,250-update, batch-eight screen, four of five seeds passed, the best loss was 0.161356316 (+0.0033%), and the fifth missed by about 20%. Timed updates took 6.5–9.9 seconds.
- The fresh biased joint rate was 25/30 (83%, 95% Clopper–Pearson interval 65–94%). Without bias, 20/30 met the loss threshold and 27/30 decoded all test rollouts exactly. [Counts and timings](window.md#fresh-seeds)
- Seed 3's pass-128 predecessor weights round to 0.967–1.000 (mean 0.989), with centred RMS steps about 0.0053 and 9×10⁻¹⁶ into passes 64 and 128.
- All five two-gain fits came within roughly 0.07–0.18% of the mixed floor in 1–2 s. Boundary constancy propagates by induction, so their state is fixed by pass T. [Gain-cell audit](audit.md#5-the-two-gain-prefix-register)
- Independent flips give error probability (1−Πᵢ(1−2εᵢ))/2, including unequal rates. The rational check covers equal rates 0, 0.1, 0.5, 0.8 and 1 through eight transitions.
- Pairwise composition can double the span each simultaneous round from single-edge summaries. A serial address oracle, full attention and a writable parallel scan have different access assumptions. [Analysis, §3](written-vs-looped.md#3-how-far-a-lookup-goes-in-one-pass)
- At scalar gate closure every call receives S. The final predictor, final loss and backbone gradient match one call with the same normalization, as does a mean of identical pass losses with the same targets. K=1 is independent of g.
- With V the cotangent of the mixed input, the restart receives (1−g)V, the previous state gV, and the scalar gain the inner product of V and the message. Channel gains sum over positions. A tanh gain contributes its usual 1−g² factor.
- At fixed input the scalar state Jacobian is gAᵧ. A uniform Lipschitz bound |g|L<1 gives contraction, with an invariant complete domain for the fixed-point conclusion. A plain pre-LN residual preserves feature translations and has eigenvalue one on them.
- At closure a uniform terminal forecast on fair words with T>3 has zero population end-only gain gradient. Its first-order logit perturbation can see at most the last three words.
- The analysis reports gain gradients 1.81722078 and 0.00234017 on its three-word final-pass prefix loss at g=0 and 0.2. Its reported gain/backbone finite-difference discrepancies are 1.29×10⁻⁷ and 3.81×10⁻¹¹. The independent review used a different seed.
- The DEQ adjoint solves (I−Jᵀ)λ=∇ℓ on a differentiable equilibrium branch with I−J invertible. The full Neumann expansion requires ρ(J)<1. Huginn truncates its unroll gradient, HRM/TRM supervise outer steps, and TRM drops HRM's one-step equilibrium approximation.
- Inkling stores 24 layers in four blocks of six. Its 6+3×12+6 schedule gives 48 applications. Its 26.4B stored, 3.4B one-pass active and about 5.4B three-pass effective active counts agree with the sources. [Release](https://thinkingmachines.ai/news/inkling-2-mini/) · [Released model card](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
- Goodfire gives separate Q, K, V and MoE mixtures, two previous middle-block sources, a 64-dimensional bottleneck, κ=1−tanh(Fz) and γ=tanh(Gz). Zero F and G suffice for identity mixing. Their gradients can be nonzero at closure, while this route contributes zero to z̄. Zeroing the bottleneck projection too removes that opening signal. [Architecture appendix](https://www.goodfire.com/research/interpreting-inkling-2-mini/)
- The research variant sampled passes 1–5 in ratio 1:2:6:2:1 during later midtraining. N-Hop Lookup improved through ten passes. SimpleQA Verified scored 6.3%, 7.3% and 6.7% at one, three and five passes. [Variant card](https://huggingface.co/thinkingmachines/tml-d24-varloop-research)
- Mean GPQA reasoning lengths were 3,703, 1,731 and 1,661 tokens at one, three and five passes on questions answered correctly at all five counts. [Release](https://thinkingmachines.ai/news/inkling-2-mini/)
- The Shakespeare sandwich keeps D=128, H=4, F=512, V=65 and length 256. The 16,961 embedding/readout parameters plus four blocks of 198,272 give 810,049. The 128 gains take the total to 810,177. Eight untied blocks would total 1,603,137.
- The prelude, three calls to the two-block core and the coda give eight applications per token and 2,048 per window. The zero initial message and closed gate recover Book 1 for arbitrary existing weights.
- Six core tapes or recomputations feed two shared block-gradient sums. All three mixed inputs contribute to the prelude. Channel gain gradients sum over token columns. Cached inference can need distinct activations for each application once the gate opens. [Sandwich review](check-wvl.md)
- The held-out losses round to 1.5313 for Book 1, 1.654 at stock tied pass four and 1.626 at windowed pass six. The single-block runs used 215,233 parameters and one seed each. [Shakespeare records](shakespeare.md)
- The four proposed time budgets are 5–15 min, 20–60 min, 10–30 min and 2–3 h per 5,000 sandwich updates plus controls. Five seeds per gate condition followed by 30 fresh seeds is the proposed design. These are allowances and planned samples. [Experiment plan](written-vs-looped.md#7-tests-to-schedule-on-mains-power)

For this review a single-thread Julia arithmetic check reproduced the three prefix floors, both model counts, the block-application budgets and the equal-rate flip probabilities in under a second. The forward extrema, training measurements and gradient checks above are comparisons with the cited records. A separate replay of the original seeded gate derivatives reached the seven-second battery cutoff before producing a result.

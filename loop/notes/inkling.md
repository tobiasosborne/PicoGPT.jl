# Inkling 2 Mini and its variable-loop variant: a source note

Sources: Thinking Machines, [Introducing Inkling 2 Mini](https://thinkingmachines.ai/news/inkling-2-mini/) (20 Nov 2026); Goodfire, [Interpreting Inkling 2 Mini](https://www.goodfire.com/research/interpreting-inkling-2-mini/); model cards for [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) and [TML-d24-varloop-research](https://huggingface.co/thinkingmachines/tml-d24-varloop-research). Paraphrased throughout.

## What the released model does

Inkling 2 Mini is a multimodal autoregressive Mixture-of-Experts model: 26.4B parameters, 3.4B active per token before looping, about 20 GB in native NVFP4/MXFP8 (last four layers MXFP8), 1M-token context, each token routed to 8 of 128 experts per layer.

Its 24 layers form four blocks of six. The first block (layers 1–6, the "prelude") and the last (19–24) run once per token; the middle two (7–18) run three times with shared weights — 48 layer applications from 24 stored layers, about 5.4B effective active parameters per token (roughly +1.0B per pass). Looping the middle rather than the last 12 won on 27 of 28 evaluations in their test (compute-equivalent gain 1.51–1.91× vs 1.24–1.36×), motivated by the outer layers translating to/from embeddings while the middle does more abstract work.

There is no single residual stream. Each block sums its own layers' outputs from zero, and every sublayer reads a learned softmax mix over RMS-normalised block sums available to it: embeddings (block 0), completed blocks, its own running sum. Decoding happens once, after the third pass and final block: a mix over all block sums, RMSNorm, unembedding. No intermediate readout or per-pass loss is described; Goodfire's lenses are after-the-fact probes.

## The message and its gate

Each pass again has embeddings and prelude among its sources and rebuilds the middle block sums. On passes two and three, each of the four inputs of each looped layer (Q, K, V, MoE) also reads a *message*: its own learned mix over the previous pass's two middle-block sums. Goodfire's appendix gives the exact gate, with z a 64-dimensional bottleneck from RMS-normalised input x and message m:

x′ = κ⊙x + γ⊙m,   κ = 1 − tanh(Fz),   γ = tanh(Gz)   (κ ∈ (0,2), γ ∈ (−1,1)).

The announcement says the gate at first blocks the message entirely, opening during training. In practice later passes read mostly previous block 3, helped by a few very large activation coordinates (Goodfire).

## Passes: what they buy, what they do not

The released model always runs three passes. The variant TML-d24-varloop-research sampled 1–5 passes (ratio 1:2:6:2:1) in later midtraining only; post-training/RL used three passes, so other counts are undertrained. Pass count at test is set externally; there is no halting.

Passes buy serial computation, not knowledge. With no written reasoning (Think Fast), its 50% horizon rose 16 → 37 human-seconds from one to three passes, was roughly flat beyond, and at ten passes — twice the training maximum — was not significantly worse. N-Hop Lookup (silent chained lookups in fictional tables) kept improving to ten. With reasoning allowed, gains were largest on long-context retrieval, then maths/science and agentic tasks (variant card, 1/3/5 passes: MRCR v2 27.2/89.2/92.1%; GraphWalks BFS 23.5/64.3/63.8%). Closed-book factual recall barely moved (AA-Omniscience, SimpleQA Verified 6.3/7.3/6.7%, SimpleVQA, WorldVQA).

The motivation is against *needing* chain of thought: a fixed-depth model must write out reasoning needing more serial steps than its depth, while looping does more per token internally and needs fewer written tokens — at the cost, they note, that unwritten reasoning evades CoT monitoring. Their checks found, at this scale, no clear monitorability loss: on GPQA Diamond questions correct at every pass count, mean written reasoning fell 3,703 tokens (1 pass) → 1,661 (5 passes; 1,731 at 3) with legibility measures unchanged; self-control of the CoT was near zero; agentic-sabotage stealth showed no trend with passes.

## Checking the advisor's summary

**Confirmed.** 24 layers, 4×6 blocks; outer blocks once, middle two three times, shared weights; fixed three passes when released; variant trained 1–5 at 1:2:6:2:1; N-Hop improving to ten, factual recall flat; 3,703 → 1,661 tokens (on that GPQA subset). The gate formula is verbatim — but only in Goodfire's appendix; announcement and cards give no formula.

**Over-simple.** (i) "Every pass restarts from the first block's output": the announcement's own wording, but Goodfire shows each input is a learned mix also including embeddings and the current pass's in-progress sum — prelude is a standing source, not the sole restart state. (ii) "The previous pass's output": not one output, but separate per-input mixes over *both* previous middle blocks. (iii) "Starts closed": stated in words in the announcement; no source states the initialisation (e.g. zero F, G). (iv) "Runs at ten without degrading": only claimed for Think Fast (plus N-Hop gains); the variant card warns of unpredictable behaviour away from three passes, and RL was three-pass only.

**Not stated.** Any per-pass loss/readout, halting, or fixed-point/exactness property; the 4×6 grouping in the cards (they say only "middle 12 layers"). No headline number in the summary is contradicted.

## Against our specimen (CONCEPT.md §1, §2, §6)

| aspect | Inkling 2 Mini / variant | specimen |
|---|---|---|
| shared weights | middle 12 of 24 layers tied; outer blocks untied, once | sole 270-parameter block tied every pass |
| carried state | pass rebuilds block sums; past enters as gated message | stream itself: Xₖ₊₁ = G(Xₖ) |
| fresh input | embeddings + prelude are standing mix sources each pass | X₀ hard-wired into first residual each pass (§6.3) |
| norm / gate | RMSNorm in mixes + learned κ/γ gate on message | parameter-free LayerNorm of stream per pass (§6.4); no gate |
| readout | once, after final block, per token | every pass, every position (deep supervision) |
| pass counts | fixed 3; variant train 1–5, test to 10 | train K ≤ 24 (lengths 4–12), test to 128 |
| extrapolation | 2× trained passes: robust on Think Fast, gaining on N-Hop, flat elsewhere | exact at 5⅓× trained length/passes, fixed point, exact floor |
| task | general multimodal LM / agentic work | prefix parity, window of two |
| passes buy | serial reasoning per token; recall flat | one word of reach per pass; loss 0.4975 → 0.1614 nats |

Both buy sequential steps, not storage, by re-reading fresh input under tied weights. They differ in mechanism — gated message into a rebuilt pass, read out once, versus direct iteration read out every pass — and in what is claimed beyond training: empirical robustness at twice the depth, versus exact extrapolation far beyond it.

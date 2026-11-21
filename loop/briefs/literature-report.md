# Literature Survey Report

Written [loop/notes/literature.md](file:///home/tobias/.t3/worktrees/PicoGPT.jl/t3-5e634fe4-74d00890eb462b9c/loop/notes/literature.md) covering all four requested areas plus a closing section on implications for a ~1,000-parameter specimen. Every source (45+ arXiv papers and primary technical posts through November 2026) was directly opened and verified via arXiv abstract/HTML/PDF or primary URL.

## Key Highlights

1. **Six families mapped:**
   - *Weight-tied / recurrent depth:* Universal Transformers (Dehghani et al. 2018), ALBERT (Lan et al. 2019), Huginn-3.5B (Geiping et al. Feb 2025), and controlled architectural ablations (Huang et al. Part I, Oct 2026, showing prelude–core–coda + input injection beats Ouro across all 10 benchmarks at matched scale).
   - *Programmable / algorithmic / expressivity:* Giannou et al. (2023, 9-layer `SUBLEQ`, 13-layer linear algebra/SGD), de Luca & Fountoulakis (2024, 19-layer/3-head $O(1)$-width graph algorithms), Yang et al. (2023/ICLR 2024, 1-layer looped ICL matching 12-layer with $1/12$ parameters), Fan et al. (2024/ICLR 2025, $n$-RASP-L length generalization), Saunshi et al. (Feb 2025, $T$ loops simulate $T$ CoT steps), Merrill & Sabharwal (2022 $\mathsf{TC}^0$ barrier; Mar 2025 $\Theta(\log n)$ depth escape), Hsu (May 2026 one-layer parity lower bound), Zhang (May 2026 compressed-loop vs. full-sequence memory-budget separation), and RLT (Zhang et al. Oct 2026, 256-bit parity & $8\times$ swap-$S_5$).
   - *Equilibrium & DeepThinking:* DEQ (Bai et al. 2019), DeepThinking / recall / overthinking (Schwarzschild et al. 2021; Bansal et al. 2022), and path independence (Anil et al. 2022).
   - *Small recursive puzzle models & debates:* HRM (Sapient, June 2025, 27M), ARC Prize Foundation's August 2025 audit (showing deep supervision doubled ARC accuracy 19%→39% while the $H/L$ hierarchy added only 35.7%→39.0%), TRM (Jolicoeur-Martineau, Oct 2025, single 2-layer 5M/7M net jumping to 87.4% Sudoku-Extreme and 45%/8% ARC-AGI-1/2), PTRM (May 2026, stochastic noise pushing Sudoku-Extreme to 98.75%), and FPRM (June 2026, flat 7M model halting on fixed-point convergence).
   - *LoopLMs at scale & halting (2025–2026):* Ouro-1.4B/2.6B (ByteDance Seed, Oct 2025, 7.7T tokens, 4 loops, entropy-regularized exit gate), Mixture-of-Recursions (Bae et al. July 2025), Nanbeige4.2-3B (July 2026, 22 layers × 2 loops, 28T tokens), iso-depth scaling law ($\varphi = 0.46$ equivalent unique parameters per recurrence, Schwethelm et al. Apr 2026, NeurIPS 2026 Spotlight), Parcae (Apr 2026), Loop-$\mu$P ($c = a N^{-1} L^{-1/2}$, June/Oct 2026), Closing the Loop (Sept 2026), and Looped MoE / LOOM (Sept–Oct 2026).

2. **What the project owner most probably heard about:**
   - **Sept 2026:** *The Information*'s report linking OpenAI's GPT-6 "Astra" to looped transformers / recurrent depth and "hidden reasoning," amplified by Sebastian Raschka's 9 Sept 2026 explainer and HN/safety threads.
   - **July–Oct 2026:** Nanbeige4.2-3B and the autumn 2026 LoopLM/NeurIPS wave.
   - **June–Oct 2025 (plus May 2026 PTRM):** HRM (27M) & TRM (5M/7M) on Sudoku/ARC-AGI, ByteDance Ouro (Oct 2025), and Huginn-3.5B (Feb 2025).

3. **Smallest published instances & ~1,000-parameter gap:**
   - The smallest trained looped transformer found is Kavuncu & Hafez's **20,704-parameter** ($d=32$) LSTM-UT on Rule 30 cellular automata (Sept 2026; extrapolating from training depth 12 to depth 60–1,000), followed by 5M–7M TRM/FPRM and $\sim 0.8\text{M}$ ($d=256$) 1-layer synthetic models (Yang, Fan, Saunshi). A ~1,000-parameter inspectable looped transformer is genuinely novel territory.

4. **Best fit for a ~1,000-parameter specimen:**
   - **Prefix parity** (with Hsu 2026 / $\mathsf{TC}^0$ lower bounds + linear vs. $\log_2$-depth front) paired with **autonomous cellular automata (Rule 60)** or **$p$-hop pointer chasing** are the strongest candidates; **input injection vs. autonomous iteration**, **fixed-point convergence / overthinking**, **randomized training depth**, and **$1/N$ residual scaling** (Loop-$\mu$P) are the cleanest mechanisms for physics students. Sudoku/ARC belong in an epilogue rather than the core specimen.

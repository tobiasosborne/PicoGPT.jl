# Brief: a sourced survey of looped transformers (to November 2026)

You are helping plan a third small book in the PicoGPT project (a GPT written from scratch in Julia,
and a book about it for BSc physics and maths students; the first book's model has 1007
parameters and is trained on the line `to be or not to be`). The third book would be about **the
tiniest looped transformer**: one transformer block, with the same parameters, applied again and
again to its own output. The project owner "has heard much about looped transformers recently".
Today is 21 November 2026, so your web search must reach 2025 and 2026; do not rely on memory alone.

## What to find out (use web search; read the papers' abstracts and key sections)

1. The families of ideas, with the founding and the most-cited recent papers of each:
   - weight-tied / recurrent-depth transformers (Universal Transformer, Dehghani et al. 2018;
     ALBERT-style sharing; recurrent-depth reasoning models such as Geiping et al. 2025 "Huginn");
   - looped transformers as programmable computers (Giannou et al. 2023) and for learning
     algorithms in context (Yang et al. 2023 "Looped transformers are better at learning learning
     algorithms"), length generalisation by looping (Fan et al. 2024), reasoning with latent thoughts
     (Saunshi et al. 2025), expressivity results (looping and circuit complexity, e.g. parity, state
     tracking, TC⁰ vs. more);
   - equilibrium models (DEQ, Bai, Kolter, Koltun 2019) and implicit differentiation;
   - small recursive reasoning models: the Hierarchical Reasoning Model (Sapient, 2025) and the Tiny
     Recursive Model (Jolicoeur-Martineau, Samsung, Oct 2025), their Sudoku/maze/ARC results and the
     debates about them;
   - looped / recurrent LLMs at scale in 2025–26 (e.g. ByteDance "Ouro" looped language models, and
     anything newer), adaptive computation and halting (ACT, Graves 2016; PonderNet; how recent
     models decide how many loops to run);
   - anything else that appeared in 2025–26 and was widely discussed (blog posts, talks, viral
     threads count as evidence of what the owner "heard about"; say where they came from).
2. **Which ideas are central**, which are fashionable, and **which the owner most probably heard
   about** (be concrete: names, dates, venues, what made noise).
3. **The smallest published instances**: the smallest looped/recurrent-depth transformers anyone has
   trained or hand-built, with parameter counts, widths, depths, loop counts and tasks (parity,
   addition, pointer chasing, cellular automata, Sudoku, graph search, linear regression in context,
   …). Hand-constructed weight solutions (e.g. RASP-style compilations, Giannou's constructions) count.
   Toy tasks used to show that a loop is *necessary* (with the control experiment or the theorem).
4. Mechanisms seen when people look inside looped models: fixed points, convergence of the state
   across passes, overthinking (accuracy falling with too many passes), stability tricks (input
   injection, normalisation, "path independence" of Anil et al. 2022), extrapolation to more passes
   than trained.

## Rules

- Every claim carries a source: authors, year, title, venue or arXiv number, and a URL you actually
  opened. If you could not verify something, say "unverified" rather than guess. Do not invent
  arXiv numbers or parameter counts.
- Be compact and exact; no hype. The readers are physicists.

## Deliverable

Write `loop/notes/literature.md` in this repository (create the folder): sections 1–4 above, then a
short final section "What this means for a tiny specimen" (which published toy tasks and mechanisms
are best suited to a model of ~1000 parameters that a reader can see whole). Then copy your final
summary message to `loop/briefs/literature-report.md`. Touch no other file.

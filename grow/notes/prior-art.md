# Prior art for growing a transformer by renormalisation

24 Nov 2026. Every item below was looked up in this session by web search. **Verified** means the
bibliographic record and the abstract-level claim were seen on the page named (arXiv, proceedings
or publisher); no full text was read, so finer claims are marked. Nothing is cited from memory.

## Lifts that change nothing at birth (width, depth, heads)

- **Net2Net.** Chen, Goodfellow, Shlens, ICLR 2016. [arXiv:1511.05641](https://arxiv.org/abs/1511.05641).
  Net2WiderNet and Net2DeeperNet: the larger network starts out computing the parent's function.
  Width, depth. The origin of our lift. Verified (arXiv).
- **Network Morphism.** Wei, Wang, Rui, Chen, ICML 2016.
  [PMLR 48:564](https://proceedings.mlr.press/v48/wei16.html). Function-preserving changes of
  depth, width, kernel size, subnet. Width, depth. Verified (PMLR).
- **Progressive stacking.** Gong, He, Li, Qin, Wang, Liu, ICML 2019.
  [PMLR 97:2337](https://proceedings.mlr.press/v97/gong19a.html). Trains a shallow BERT and stacks
  copies of its layers; about 25 % faster. Depth; not function preserving. Verified (PMLR).
- **bert2BERT.** Chen, Yin, Shang, Jiang et al., ACL 2022.
  [2022.acl-long.151](https://aclanthology.org/2022.acl-long.151/). Initialises a large LM from one
  about half its size; saves about 45 % of BERT-base pre-training. Width, depth. Verified (ACL).
- **Staged training.** Shen, Walsh, Keutzer, Dodge, Peters, Beltagy, ICML 2022.
  [arXiv:2203.06211](https://arxiv.org/abs/2203.06211). Growth operators that preserve the loss
  *and the training dynamics* (optimiser state carried over); up to 22 % compute saved against a
  strong scratch baseline. Width, depth. Verified (PMLR 162:19893).
- **GradMax.** Evci et al., ICLR 2022. [arXiv:2201.05125](https://arxiv.org/abs/2201.05125). New
  neurons leave the function alone and are initialised by an SVD to maximise their gradient norm.
  Width; an answer to "which directions to add". Verified (arXiv).
- **LiGO.** Wang, Panda et al., ICLR 2023. [arXiv:2303.00980](https://arxiv.org/abs/2303.00980).
  Learns a Kronecker-factored linear map from small to large weights. Width, depth; a learned
  lift, not an exact one. Verified (ICLR page).
- **Composable function-preserving expansions.** Gesmundo, Maile, 2023.
  [arXiv:2308.06103](https://arxiv.org/abs/2308.06103). Six expansions of a transformer (MLP
  width, number of heads, head output size, attention input size, stream width, number of layers),
  each with a proof of exact preservation. Width, heads, depth. **This is our lift column, minus
  vocabulary and context.** Verified (arXiv abstract; the list of six from a secondary summary).
- **LEMON.** Wang et al., ICLR 2024. [arXiv:2310.07999](https://arxiv.org/abs/2310.07999).
  Lossless expansion including LayerNorm; reports that bert2BERT is lossy because of LayerNorm,
  and that plain duplication is lossless only when the new width is a multiple of the old. Width,
  depth. **Bears directly on our unchecked width row.** Verified (ICLR proceedings).
- **Progressive growing of GANs.** Karras, Aila, Laine, Lehtinen, ICLR 2018.
  [arXiv:1710.10196](https://arxiv.org/abs/1710.10196). Model and image resolution grow together.
  Depth and environment: the nearest case of *world and model refined in step*. Verified (arXiv).

## Vocabulary

- **Brown et al.**, Class-based n-gram models, Comput. Linguistics 18(4), 1992.
  [J92-4003](https://aclanthology.org/J92-4003). Classes from co-occurrence statistics. Verified.
- **Goodman**, Classes for fast maximum entropy training, ICASSP 2001.
  [arXiv:cs/0108006](https://arxiv.org/abs/cs/0108006). One model predicts the class, a second the
  word given the class: our p(v|h)·q(b|v,h). Verified (arXiv).
- **Morin, Bengio**, Hierarchical probabilistic NNLM, AISTATS 2005.
  [PMLR R5:246](https://proceedings.mlr.press/r5/morin05a.html). The same, as a tree. Verified.

All three use the factorisation for speed, with both factors trained together. Enlarging the
vocabulary of a trained model exists as embedding initialisation (mean of old rows; search hit:
Hewitt's note, not read). Two searches found no use of the class factorisation as an exact lift.

## Worlds: growing, and coarse-graining exactly

- **Curriculum learning.** Bengio, Louradour, Collobert, Weston, ICML 2009.
  [doi](https://doi.org/10.1145/1553374.1553380). Easy-to-hard ordering; one experiment is a
  language model with a gradually expanding vocabulary. Environment, vocabulary. Verified (DBLP).
- **POET** (Wang, Lehman, Clune, Stanley, [arXiv:1901.01753](https://arxiv.org/abs/1901.01753));
  **PAIRED** (Dennis et al., NeurIPS 2020, [arXiv:2012.02096](https://arxiv.org/abs/2012.02096));
  **XLand** (DeepMind, [arXiv:2107.12808](https://arxiv.org/abs/2107.12808)). Environments are
  generated to track the agent. Environment. Model size fixed; no map between worlds. Verified.
- **Lumpability.** Kemeny, Snell, *Finite Markov Chains*, 1960
  ([summary](https://en.wikipedia.org/wiki/Lumpability)). A partition is lumpable when the
  probability of entering each block is constant on each block. Our exact-closure condition.
  Verified through secondary sources only; the book was not seen.
- **Bisimulation.** Givan, Dean, Greig, Artificial Intelligence 147:163-223, 2003. The same idea
  with actions and rewards: the reduced model's optimal policy lifts. Verified (citations).
- **MDP homomorphisms.** Ravindran, Barto, 2002.
  [doi](https://doi.org/10.1007/3-540-45622-8_15). Adds symmetry, so actions are mapped too.
  Verified (Springer). **Li, Walsh, Littman**, Towards a unified theory of state abstraction,
  ISAIM 2006, pp. 531-539: five exact abstraction types, ordered. Verified (bibliography only).

## Renormalisation and deep learning

- **Mehta, Schwab** 2014, [arXiv:1410.3831](https://arxiv.org/abs/1410.3831): variational RG mapped
  to stacked RBMs. **Koch-Janusz, Ringel**, Nature Physics 14:578, 2018,
  [doi](https://doi.org/10.1038/s41567-018-0081-4): choose the coarse variables by maximising
  mutual information with the distant environment. This is White's criterion in our reading:
  the rest of the system decides what to keep. Truncation. Both verified.
- **Roberts, Yaida, Hanin**, *Principles of Deep Learning Theory*, CUP 2022,
  [arXiv:2106.10165](https://arxiv.org/abs/2106.10165). "RG flow" there means *representation*
  group flow over depth at initialisation. Not about growth. Verified (arXiv).
- **Haggi-Mani, Rish** 2026, [arXiv:2607.15449](https://arxiv.org/abs/2607.15449): Wilsonian
  analysis of attention as a perturbation of an MLP fixed point. Analysis, not growth. Abstract
  fragments only.

## Tensor networks

- **Stoudenmire, Schwab**, NIPS 2016, [arXiv:1605.05775](https://arxiv.org/abs/1605.05775); **Han,
  Wang, Fan, Wang, Zhang**, PRX 8:031012, 2018,
  [doi](https://doi.org/10.1103/PhysRevX.8.031012). MPS models trained DMRG-style; bond dimensions
  grow and shrink during training. Width and truncation, with the canonical form transformers
  lack. Verified (abstracts; the sweep details were not checked).

## Attention under blocking, passes, context

- **H-Transformer-1D** (Zhu, Soricut, ACL 2021, [arXiv:2107.11906](https://arxiv.org/abs/2107.11906)),
  hierarchical-matrix attention; **FMMformer** (Nguyen et al., NeurIPS 2021,
  [arXiv:2108.02347](https://arxiv.org/abs/2108.02347)), banded near field plus low-rank far
  field; **Performer** (Choromanski et al., ICLR 2021,
  [arXiv:2009.14794](https://arxiv.org/abs/2009.14794)), random-feature estimate of the softmax
  kernel. Blocked positions. All approximate; none states the exact merged pair. Verified.
- **Universal Transformer** (Dehghani et al., ICLR 2019,
  [arXiv:1807.03819](https://arxiv.org/abs/1807.03819)), one block applied repeatedly with
  per-position halting; **Giannou et al.**, ICML 2023,
  [arXiv:2301.13196](https://arxiv.org/abs/2301.13196); **Yang, Lee, Nowak, Papailiopoulos**,
  ICLR 2024, [arXiv:2311.12424](https://arxiv.org/abs/2311.12424). Passes. None adds a pass to a
  trained model behind a gate. Verified.
- **Position Interpolation** (Chen et al. 2023, [arXiv:2306.15595](https://arxiv.org/abs/2306.15595));
  **YaRN** (Peng et al. 2023, [arXiv:2309.00071](https://arxiv.org/abs/2309.00071)). Extend a
  trained model's window by rescaling rotary positions, then fine-tune briefly. Context;
  approximate. Verified.
- **Matryoshka representations** (Kusupati et al., NeurIPS 2022,
  [arXiv:2205.13147](https://arxiv.org/abs/2205.13147)): leading coordinates usable alone.
  Truncation by construction. Verified.

## Conclusion

**What exists, and under what names.** The lift on width, heads, MLP and depth is
*function-preserving* or *lossless expansion* (Net2Net; for transformers Gesmundo–Maile and LEMON).
The vocabulary factorisation is the *class-based* or *hierarchical softmax* language model. Exact
coarse-graining of a world is *lumpability*, *bisimulation* or an *MDP homomorphism*. Worlds that
grow with the learner are *curriculum learning*, *open-endedness* and *unsupervised environment
design*. Growing a bond dimension while sweeping is ordinary MPS learning. Choosing coarse
variables by what the environment needs is Koch-Janusz and Ringel.

**What was not found.** (i) The coarse-graining *back*, T2 → T1, stated as a map of which the
lift is a section; the growth literature only goes up. (ii) A tower of worlds joined by lumpable
maps, with the model lifted along the same maps and the loss splitting exactly. (iii) The class
factorisation used as a growth step. (iv) The sweep back, on transformers. This rests on two
targeted searches and the items above, so it is "not found", not "does not exist".

**A caution.** The measured savings of growth are modest (22 to 45 %) and come from large
models. Our open claim that growing is cheaper has this as its realistic ceiling.

**Read first.** Gesmundo and Maile (the lifts, proved); Shen et al. (loss *and* dynamics, honest
baselines); Givan, Dean and Greig (exact abstraction of worlds). Then LEMON for the width row.

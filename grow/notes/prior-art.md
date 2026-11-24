# Prior art for growing a transformer by renormalisation

24 Nov 2026. Every item was looked up in this session by web search (arXiv, publisher or
proceedings pages in the results). "Verified" means the bibliographic record and the abstract-level
claim were seen; no full text was read unless stated. FIRST VERSION, being extended.

## Function-preserving growth (width, depth, heads)

- **Net2Net.** Chen, Goodfellow, Shlens, ICLR 2016. [arXiv:1511.05641](https://arxiv.org/abs/1511.05641).
  Net2WiderNet and Net2DeeperNet: the larger network starts computing the same function as the
  smaller. Axis: width, depth. This is our "lift that changes nothing at birth". Verified (arXiv).
- **Network Morphism.** Wei, Wang, Rui, Chen, ICML 2016, PMLR 48:564-572.
  [arXiv:1603.01670](https://arxiv.org/abs/1603.01670). Morphs a trained network with the function
  completely preserved: depth, width, kernel size, subnet. Axis: width, depth. Verified (PMLR).
- **bert2BERT.** Chen, Yin, Shang, Jiang et al., ACL 2022.
  [aclanthology 2022.acl-long.151](https://aclanthology.org/2022.acl-long.151/). Initialises a
  large pretrained LM from a small one; saves about 45 % of pre-training cost for BERT-base from a
  half-size model. Axis: width, depth. Verified (ACL Anthology).
- **LiGO.** Wang, Panda et al., ICLR 2023. [arXiv:2303.00980](https://arxiv.org/abs/2303.00980).
  Learns a linear (Kronecker-factored) map from small to large parameters, width and depth
  operators. Not function preserving; the lift is learned. Axis: width, depth. Verified (ICLR page).
- **Composable function-preserving expansions for transformers.** Gesmundo, Maile, 2023.
  [arXiv:2308.06103](https://arxiv.org/abs/2308.06103). Six expansions of a transformer, each with a
  proof of exact function preservation under constraints on the new parameters' initialisation.
  Axis: width, heads, depth. The closest thing to our table's lift column. Verified (arXiv).
- **GradMax.** Evci, van Merriënboer, Unterthiner, Vladymyrov, Pedregosa, ICLR 2022.
  [arXiv:2201.05125](https://arxiv.org/abs/2201.05125). New neurons leave the function unchanged
  and are initialised, by an SVD, to maximise their gradient norm. Axis: width. Relevant to "which
  directions to add". Verified (arXiv).
- **Staged training for transformer LMs.** Shen, Walsh, Keutzer, Dodge, Peters, Beltagy, ICML 2022,
  PMLR 162:19893-19908. [arXiv:2203.06211](https://arxiv.org/abs/2203.06211). Growth operators
  that preserve the loss *and the training dynamics* (the optimiser state is carried over); up to
  22 % compute saved against a strong scratch baseline. Axis: width, depth. Verified (PMLR).
- **Progressive growing of GANs.** Karras, Aila, Laine, Lehtinen, ICLR 2018.
  [arXiv:1710.10196](https://arxiv.org/abs/1710.10196). Generator and discriminator grow together
  from low to high resolution; layers are added for finer detail. Axis: depth and environment (the
  data are coarse-grained images). Verified (arXiv).

## Vocabulary

- **Class-based n-grams.** Brown, Della Pietra, deSouza, Lai, Mercer, Computational Linguistics
  18(4):467-480, 1992. [J92-4003](https://aclanthology.org/J92-4003). Word classes found from
  co-occurrence statistics; predict class, then word. Verified (ACL Anthology).
- **Classes for fast maximum entropy training.** Goodman, ICASSP 2001.
  [arXiv:cs/0108006](https://arxiv.org/abs/cs/0108006). Two models: one predicts the class, the
  second the word given the class. Exactly our p(v|h)·q(b|v,h). Verified (arXiv).
- **Hierarchical probabilistic neural network language model.** Morin, Bengio, AISTATS 2005,
  PMLR R5:246-252. [link](https://proceedings.mlr.press/r5/morin05a.html). The same factorisation
  made into a binary tree. Verified (PMLR).

## Environments that grow

- **POET.** Wang, Lehman, Clune, Stanley, 2019. [arXiv:1901.01753](https://arxiv.org/abs/1901.01753).
  Co-evolves environments and agents, with transfer between them. Verified (arXiv).
- **PAIRED.** Dennis, Jaques, Vinitsky, Bayen, Russell, Critch, Levine, NeurIPS 2020.
  [arXiv:2012.02096](https://arxiv.org/abs/2012.02096). An adversary designs environments to
  maximise regret, which yields a curriculum. Verified (arXiv).
- **XLand.** Open-Ended Learning Team (DeepMind), 2021.
  [arXiv:2107.12808](https://arxiv.org/abs/2107.12808). Task distributions regenerated as the agent
  learns; each generation distils from the previous. Verified (arXiv).
- **Curriculum learning.** Bengio, Louradour, Collobert, Weston, ICML 2009, pp. 41-48.
  [doi:10.1145/1553374.1553380](https://doi.org/10.1145/1553374.1553380). Ordering examples easy to
  hard; one experiment trains a language model with a gradually expanding vocabulary. Verified.

In all four the model stays the same size while the world grows. None has a coarse-graining map
between worlds.

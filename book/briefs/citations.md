# Citation verification — integration pass, 4 November 2026

Checked every named scholarly citation in `book/chapters/*.md` by web search and the linked
paper, publisher, author or library record. Compact citations in the chapters now link to
sources. Years refer to the cited version (conference years for Adam/AdamW/Mixed Precision;
preprint years where explicitly stated). Long author lists remain “et al.” in the prose.

| Citation | Verified title, authors and publication | Source / identifier |
|---|---|---|
| Berkson 1944 | Joseph Berkson, *Application of the Logistic Function to Bio-Assay*, Journal of the American Statistical Association 39(227), 357–365. | [Publisher DOI](https://doi.org/10.1080/01621459.1944.10500699) |
| Shannon 1948 | Claude E. Shannon, *A Mathematical Theory of Communication*, Bell System Technical Journal 27, 379–423 and 623–656. | [Paper](https://people.tamu.edu/~jml/entropy.pdf) |
| Tribus 1961 | Myron Tribus, *Thermostatics and Thermodynamics: An Introduction to Energy, Information and States of Matter, with Engineering Applications*, D. Van Nostrand. Author and publication year verified; see limitation below. | [Author’s retrospective](https://elsmar.com/pdf_files/Tribus/sys_entropy.pdf), [book record](https://archive.org/details/thermostaticsthe00trib) |
| He et al. 2016 | Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, *Deep Residual Learning for Image Recognition*, CVPR 2016 (preprint 2015). | [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) |
| E 2017 | Weinan E, *A Proposal on Machine Learning via Dynamical Systems*, Communications in Mathematics and Statistics 5, 1–11. | [Publisher DOI](https://doi.org/10.1007/s40304-017-0103-z), [author bibliography](https://web.math.princeton.edu/~weinan/index.html) |
| Haber & Ruthotto 2017 | Eldad Haber, Lars Ruthotto, *Stable Architectures for Deep Neural Networks*, 2017 preprint; Inverse Problems 34(1), 014004 (2018 issue; online publication 2017). | [arXiv:1705.03341](https://arxiv.org/abs/1705.03341) |
| Chen et al. 2018 | Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud, *Neural Ordinary Differential Equations*, NeurIPS 2018. | [arXiv:1806.07366](https://arxiv.org/abs/1806.07366) |
| Ba et al. 2016 | Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton, *Layer Normalization*, arXiv preprint. | [arXiv:1607.06450](https://arxiv.org/abs/1607.06450) |
| Vaswani et al. 2017 | Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin, *Attention Is All You Need*, NeurIPS 2017. | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| Radford et al. 2019 | Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, *Language Models are Unsupervised Multitask Learners*, OpenAI technical report (no arXiv ID). | [Original report](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) |
| Rosenblatt 1958 | Frank Rosenblatt, *The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain*, Psychological Review 65(6), 386–408. | [Publisher DOI](https://doi.org/10.1037/h0042519), [paper scan](https://web.stanford.edu/class/psych209a/ReadingsByDate/01_30/Rosenblatt58Perceptron.pdf) |
| Linnainmaa 1970 | Seppo Linnainmaa, *Algoritmin kumulatiivinen pyöristysvirhe yksittäisten pyöristysvirheiden Taylor-kehitelmänä* (English rendering: *The Representation of the Cumulative Rounding Error of an Algorithm as a Taylor Expansion of the Local Rounding Errors*), master’s thesis, University of Helsinki. His 1976 BIT paper cites this thesis. | [University library record](https://www.finna.fi/Record/helka.9933382303506253), [author’s 1976 paper](https://doi.org/10.1007/BF01931367) |
| Werbos 1974 | Paul John Werbos, *Beyond Regression: New Tools for Prediction and Analysis in the Behavioral Sciences*, PhD thesis, Harvard University (title page: August 1974). | [Thesis scan](https://gwern.net/doc/ai/nn/1974-werbos.pdf) |
| Rumelhart et al. 1986 | David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams, *Learning representations by back-propagating errors*, Nature 323, 533–536. | [Publisher DOI](https://doi.org/10.1038/323533a0), [paper scan](https://gwern.net/doc/ai/nn/1986-rumelhart-2.pdf) |
| Cohen et al. 2021 | Jeremy M. Cohen, Simran Kaur, Yuanzhi Li, J. Zico Kolter, Ameet Talwalkar, *Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability*, ICLR 2021. | [arXiv:2103.00065](https://arxiv.org/abs/2103.00065), [conference paper](https://openreview.net/pdf/49962b19307f5018fe57b60a80c6cecb9a29eb02.pdf) |
| Kingma & Ba 2015 | Diederik P. Kingma, Jimmy Ba, *Adam: A Method for Stochastic Optimization*, ICLR 2015 (preprint 2014). | [arXiv:1412.6980](https://arxiv.org/abs/1412.6980) |
| Loshchilov & Hutter 2019 | Ilya Loshchilov, Frank Hutter, *Decoupled Weight Decay Regularization*, ICLR 2019 (preprint 2017, initially titled *Fixing Weight Decay Regularization in Adam*). | [arXiv:1711.05101](https://arxiv.org/abs/1711.05101), [conference paper](https://openreview.net/pdf?id=Bkg6RiCqY7) |
| Thilak et al. 2022 | Vimal Thilak, Etai Littwin, Shuangfei Zhai, Omid Saremi, Roni Paiss, Joshua Susskind, *The Slingshot Mechanism: An Empirical Study of Adaptive Optimizers and the Grokking Phenomenon*, arXiv preprint. | [arXiv:2206.04817](https://arxiv.org/abs/2206.04817) |
| LeCun et al. 1998 | Yann LeCun, Léon Bottou, Genevieve B. Orr, Klaus-Robert Müller, *Efficient BackProp*, in *Neural Networks: Tricks of the Trade*, LNCS 1524, Springer. Previously an undated attribution in ch. 8. | [Author’s publication record and paper](https://bottou.org/papers/lecun-98x) |
| Glorot & Bengio 2010 | Xavier Glorot, Yoshua Bengio, *Understanding the difficulty of training deep feedforward neural networks*, AISTATS, PMLR 9, 249–256. | [Conference record](https://proceedings.mlr.press/v9/glorot10a.html) |
| He et al. 2015 | Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification*, ICCV 2015. | [arXiv:1502.01852](https://arxiv.org/abs/1502.01852) |
| Olsson et al. 2022 | Catherine Olsson et al. (26 authors), *In-context Learning and Induction Heads*, Transformer Circuits / arXiv preprint. | [arXiv:2209.11895](https://arxiv.org/abs/2209.11895) |
| Gage 1994 | Philip Gage, *A New Algorithm for Data Compression*, C Users Journal 12(2), 23–38. | [Original article archive](https://jacobfilipp.com/DrDobbs/articles/CUJ/1994/9402/gage/gage.htm) |
| Sennrich et al. 2016 | Rico Sennrich, Barry Haddow, Alexandra Birch, *Neural Machine Translation of Rare Words with Subword Units*, ACL 2016, 1715–1725 (preprint 2015). | [ACL record](https://aclanthology.org/P16-1162/), [arXiv:1508.07909](https://arxiv.org/abs/1508.07909) |
| Brown et al. 2020 | Tom B. Brown et al. (31 authors), *Language Models are Few-Shot Learners*, NeurIPS 2020. | [arXiv:2005.14165](https://arxiv.org/abs/2005.14165) |
| Kaplan et al. 2020 | Jared Kaplan et al. (10 authors), *Scaling Laws for Neural Language Models*, arXiv preprint. | [arXiv:2001.08361](https://arxiv.org/abs/2001.08361) |
| Hoffmann et al. 2022 | Jordan Hoffmann et al. (22 authors), *Training Compute-Optimal Large Language Models*, arXiv preprint; conference title *An empirical analysis of compute-optimal large language model training*, NeurIPS 2022. | [arXiv:2203.15556](https://arxiv.org/abs/2203.15556), [conference record](https://papers.nips.cc/paper_files/paper/2022/hash/c1e2faff6f588870935f114ebe04a3e5-Abstract-Conference.html) |
| Dao et al. 2022 | Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré, *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS 2022. | [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) |
| Micikevicius et al. 2018 | Paulius Micikevicius et al. (11 authors), *Mixed Precision Training*, ICLR 2018 (preprint 2017). | [arXiv:1710.03740](https://arxiv.org/abs/1710.03740) |
| Xiong et al. 2020 | Ruibin Xiong et al. (10 authors), *On Layer Normalization in the Transformer Architecture*, ICML 2020, PMLR 119, 10524–10533. | [Conference record](https://proceedings.mlr.press/v119/xiong20b.html), [arXiv:2002.04745](https://arxiv.org/abs/2002.04745) |
| Su et al. 2021 | Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu, *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021 arXiv version (later revised and published). | [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) |
| Shazeer 2020 | Noam Shazeer, *GLU Variants Improve Transformer*, arXiv preprint. | [arXiv:2002.05202](https://arxiv.org/abs/2002.05202) |
| Gu & Dao 2023 | Albert Gu, Tri Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, 2023 arXiv preprint (conference publication followed in 2024). | [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) |
| Ouyang et al. 2022 | Long Ouyang et al. (20 authors), *Training language models to follow instructions with human feedback*, NeurIPS 2022. | [arXiv:2203.02155](https://arxiv.org/abs/2203.02155) |

Corrections: expanded the truncated ResNet title; named Weinan E unambiguously; supplied the
missing date and coauthors for the LeCun attribution; supplied full Slingshot title; distinguished
preprint years/titles from conference publications for Hoffmann, Su and Gu/Dao. Other cited
authors and years were consistent with the cited versions. No invented arXiv IDs were added.

Remaining verification limitation: the claim that Tribus **coined** “surprisal” could not be
established from an accessible original 1961 page. The prose now says he calls it surprisal,
without claiming priority. The book’s author/title/year/publisher are verified. Other works
were identifiable; this is the only unresolved historical priority claim. Undated references
to Pontryagin and weather forecasting are historical analogies, not bibliographic citations.

# The mean-field step in the trained specimen

Equal weight for each of the 80 readers in the ten overlapping training windows. Each variant uses the saved trained parameters and the same attention rows, values and residual input.

At reader $i$, independently draw $j_1,j_2\le i$ with weight $w=A^{(1)}_{ij_1}A^{(2)}_{ij_2}$. Let $q$ be the next-character distribution after substituting those two value vectors into column $i$. The usual prediction $p$ decodes the two mean value vectors. The arithmetic mixture $p^{\rm mix}=\sum wq$ averages the per-pair probabilities. The geometric mixture $p^{\rm geo}=\operatorname{softmax}(\sum w\log q)$ averages their logarithms and renormalises. A single-head mixture draws one source and keeps the other head at its usual mean output.

## Distribution gaps

KL and target cross-entropy are in nats. TV is half the sum of absolute probability differences.

| Distance | Mean | Median | Max |
| --- | ---: | ---: | ---: |
| KL(p‖mix) | 0.137724 | 0.0525889 | 0.698828 |
| KL(mix‖p) | 0.782668 | 0.266463 | 3.92433 |
| KL(p‖geo) | 0.00295389 | 0.000144351 | 0.055186 |
| TV(p, mix) | 0.11878 | 0.0522109 | 0.503729 |
| TV(p, geo) | 0.00339336 | 0.000445193 | 0.0548657 |
| TV(mix, geo) | 0.115488 | 0.0519361 | 0.470254 |

| Prediction | Mean target cross-entropy |
| --- | ---: |
| Usual | 0.058935139 |
| Arithmetic mixture | 0.198007055 |
| Geometric mixture | 0.062365737 |

Top character agreement was 80/80 for p and mix, 80/80 for p and geo, and 80/80 for mix and geo. All three agreed at 80/80 readers.

| Head sampled | Mean KL(p‖mix) | Mean KL(mix‖p) | Mean TV | Top character agreement with p | Mean target cross-entropy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0647274 | 0.347863 | 0.055802 | 79/80 | 0.124483 |
| 2 | 0.0699329 | 0.391833 | 0.0609032 | 79/80 | 0.12972 |

| Head | Mean row entropy (nats) | Mean weighted squared value spread | Mean maximum weight |
| ---: | ---: | ---: | ---: |
| 1 | 0.630021 | 0.873016 | 0.757751 |
| 2 | 0.758843 | 1.30102 | 0.703163 |

## Largest gaps

The three highest values for each distance, with the full window and its 1-based reader position.

| Distance | Largest | Second | Third |
| --- | --- | --- | --- |
| KL(p‖mix) | `"or not t"` · 6 · 0.698828 | `"not to b"` · 3 · 0.644705 | `"o be or "` · 6 · 0.519876 |
| KL(mix‖p) | `"not to b"` · 3 · 3.92433 | `"or not t"` · 6 · 3.56832 | `" not to "` · 4 · 2.64042 |
| KL(p‖geo) | `"or not t"` · 6 · 0.055186 | `"o be or "` · 6 · 0.0245014 | `"be or no"` · 6 · 0.0179142 |
| TV(p, mix) | `"or not t"` · 6 · 0.503729 | `"not to b"` · 3 · 0.4755 | `"o be or "` · 6 · 0.406901 |
| TV(p, geo) | `"or not t"` · 6 · 0.0548657 | `"o be or "` · 6 · 0.025727 | `"be or no"` · 6 · 0.0196356 |
| TV(mix, geo) | `"not to b"` · 3 · 0.470254 | `"or not t"` · 6 · 0.452061 | `"be or no"` · 6 · 0.385662 |

## Figure readers

Each spread example maximises the selected head's single-head TV among readers with at least 0.15 weight on each of two distinct characters and no individual source above 0.75. The examples are selected in head order with distinct readers. The near-copy example maximises the smaller of the two row maxima among readers with more than one allowed source.

- **Window 7 `"or not t"`, position 6.** Head 1 gives substantial weight to 4 (`'n'`, 0.4740), 5 (`'o'`, 0.1524), 6 (`'t'`, 0.2368). Sampling that head gives TV 0.655297 from p.
- **Window 2 `"o be or "`, position 6.** Head 2 gives substantial weight to 3 (`'b'`, 0.2065), 5 (`' '`, 0.2411), 6 (`'o'`, 0.3295). Sampling that head gives TV 0.332762 from p.
- **Window 4 `"be or no"`, position 3.** The two heads put 0.991839 and 0.998841 on their favourite sources at positions 2 (`'e'`) and 1 (`'b'`). TV between p and the joint arithmetic mixture is 0.00793822.

At window 7 `"or not t"`, position 6, the probability of the target `" "` is 0.9997 under p, 0.4960 under the arithmetic mixture and 0.9449 under the geometric mixture. The single-head mixtures choose `"o"` and `"o"`, respectively.

The figure records in [mixture.json](../artifacts/mixture.json) carry both attention rows, per-source predictions for each head with the other held fixed, and all three joint predictions. Numeric arrays use column-major storage.

## Checks

- The ordinary mean loss agrees with the stored final checkpoint loss (0.058935138812 nats) and the package loss to within 1.39e-17 nats. The brief gives approximately 0.0588 nats.
- Replaying the unedited batch from the attention output reproduced the package logits to 0. Across all 2760 interventions, untouched columns changed by at most 0 in logits and 0 in probabilities. The edited-column logits agreed with the isolated-column calculation to 2.31e-14.
- With fixed affine logits, averaging log probabilities subtracts a common scalar from the expected logits. Softmax cancels that scalar, so decoding the mean and taking the normalised geometric mixture agree. On the specimen's actual rows and values, the largest probability error was 3.33e-16, below 1e-12. For the real nonlinear readout, mean TV between p and geo was 0.00339336 and the maximum was 0.0548657.
- At position 1, both heads have a single source in all 10 windows. The variants coincide algebraically. The raw calculation agreed within 5.55e-16 before recording the identical distributions.

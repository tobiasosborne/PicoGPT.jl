# The loop on Tiny Shakespeare

## Setup

Book 1's four independently trained blocks reached a best validation loss of **1.5313 nats per character** at step 5,000 with 810,049 parameters. The recorded update time was 36.2 minutes. The trial uses the same Tiny Shakespeare file via a symlink into the main checkout. The ordered split is 1,003,855 training characters and 111,539 held-out characters. Validation uses Book 1's 256 windows of 256 predictions each, drawn with replacement from the held-out tail with `Xoshiro(1236)`. Training-set evaluation uses the same procedure with `Xoshiro(1235)` on the training split. [Book 1 loss data](../../artifacts/shakespeare/loss.json)

Each trial model has `V=65, D=128, H=4, F=512, T=256, L=1`. The embedding and output matrices, final LayerNorm and output bias account for 16,961 scalars. The attention projections, MLP and two block LayerNorms account for 198,272, giving **215,233 parameters** per model. All models use PicoGPT's ordinary Float32 initialization with `Xoshiro(1234)`, sinusoidal positions, full causal attention with predecessor bias zero and a shared final LayerNorm and readout. Each training step draws 32 windows with `Xoshiro(1234 + step)`. AdamW uses β₁=0.9, β₂=0.95, ε=10⁻⁸, matrix-only decay 0.1 and global-norm clipping at 1. The learning rate ramps linearly to 0.001 over 200 steps, then follows a cosine to 0.0001 at step 5,000. The runs use 12 Julia threads and one BLAS thread.

| Model | Initial stream | First residual | Stream after the MLP residual | Training objective |
| --- | --- | --- | --- | --- |
| One block, one pass | Tokens plus positions | Current stream plus attention output | Ordinary residual sum | Pass 1 cross-entropy |
| Looped | First input column only, other columns zero | Original token-plus-position input plus attention output | Fixed LayerNorm, unit gain and zero offset | Mean cross-entropy over every pass of the sampled unroll |
| Stock tied | Tokens plus positions | Current stream plus attention output | Ordinary residual sum | Pass 4 cross-entropy |

The looped model retains the toy specimen's boundary-only initial stream. Its first column has the ordinary token embedding and sinusoidal position. All positions get their token embedding and position through the anchored residual starting at pass 1. Each batch takes one pass count uniformly from 3, 4 and 5 after drawing its window starts. The expected count of four matches Book 1's block applications. Extra final readouts and fixed stream LayerNorms add work to that count. Every pass scored during training uses the observed next character as its target.

Checkpoint selection uses the lowest held-out loss at the trained pass for each control and at pass 4 for the looped model. All rollout losses, samples and settling measurements use the selected checkpoint for that model. Passes above 1 in the one-pass control reuse its block after training. Passes above 4 for stock tied and above 5 for looped extend each trained recurrence.

The optional width-two Shakespeare model and n-gram comparison were skipped to prioritize the three full-width runs.

Run the checks, training jobs and export from the repository root.

```sh
set -e
julia --project -t 12 loop/scripts/shakespeare_gradcheck.jl
for model in one looped stock; do
  julia --project -t 12 loop/scripts/shakespeare_train.jl "$model"
done
julia --project -t 12 loop/scripts/shakespeare_export.jl
```

Checkpoints and raw training logs live under the git-ignored `artifacts/shakespeare/checkpoints/loop-trial/`. The data link is under the git-ignored `artifacts/data/`.

The trial ran on AC power with the laptop's Power Saver profile active. Update time covers the timed training steps. Wall time covers the training loop, including its fixed-window evaluations and checkpoint work. Book 1's 36.2 minutes is the historical update time, and its log does not record a power profile. The elapsed times here measure this run under Power Saver and do not establish a hardware-normalized speed ratio.

## Float64 checks before training

The check uses `V=5, D=4, H=2, F=7, T=4`, two sequences and three tied passes. It uses sinusoidal positions and the full causal mask for both stream maps. Central differences step every parameter by ±10⁻⁶.

| Stream map | Logits at all passes, arbitrary covectors | Mean loss at all passes |
| --- | ---: | ---: |
| Anchored, normalized | 2.48 × 10⁻¹⁰ | 1.30 × 10⁻⁹ |
| Stock | 2.97 × 10⁻¹⁰ | 1.43 × 10⁻⁹ |

The figures are relative gradient errors. The one-pass control also matches the stock one-block forward and pullback. The stock tied forward matches stock PicoGPT with three identical copies of the block, and its block gradient matches the sum over those copies. The 14 Shakespeare checks and the existing loop and specimen tests passed. Training used Julia 1.13.3.

## Held-out loss by pass

The table gives mean next-character cross-entropy in nats over Book 1's 256 fixed held-out windows. Its 810,049-parameter reference is **1.5313**.

| Pass | One pass | Looped | Stock tied | A · Full input | B · Window of two |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.698270 | 2.492265 | 3.539166 | 1.882829 | 2.054792 |
| 2 | 3.569208 | 1.883311 | 2.751502 | 1.740334 | 1.786840 |
| 3 | 4.239137 | 1.786466 | 1.990949 | 1.710249 | 1.669827 |
| 4 | 4.774442 | 1.784327 | 1.654106 | 1.714502 | 1.634295 |
| 5 | 5.187675 | 1.800816 | 1.917334 | 1.725308 | 1.625673 |
| 6 | 5.500590 | 1.820031 | 2.488327 | 1.735950 | 1.625555 |
| 7 | 5.755695 | 1.835376 | 3.090388 | 1.745752 | 1.627698 |
| 8 | 5.972504 | 1.848174 | 3.600285 | 1.753726 | 1.631797 |
| 9 | 6.154857 | 1.857968 | 4.005468 | 1.760097 | 1.633215 |
| 10 | 6.307955 | 1.865680 | 4.320483 | 1.765113 | 1.634078 |
| 11 | 6.438417 | 1.871944 | 4.565818 | 1.769131 | 1.635628 |
| 12 | 6.551881 | 1.877576 | 4.762143 | 1.772489 | 1.636252 |
| 13 | 6.652435 | 1.882012 | 4.925695 | 1.775383 | 1.637739 |
| 14 | 6.742736 | 1.885670 | 5.067607 | 1.777844 | 1.638098 |
| 15 | 6.824457 | 1.888624 | 5.195070 | 1.780019 | 1.639099 |
| 16 | 6.898866 | 1.891412 | 5.312422 | 1.782015 | 1.638717 |

| Model | Selected step | Selection pass | Lowest-loss pass | Gap to Book 1 at lowest-loss pass |
| --- | ---: | ---: | ---: | ---: |
| One pass | 5000 | 1 | 1 | +0.166970 |
| Looped | 5000 | 4 | 4 | +0.253027 |
| Stock tied | 5000 | 4 | 4 | +0.122806 |
| A · Full input | 5000 | 4 | 3 | +0.178949 |
| B · Window of two | 5000 | 8 | 6 | +0.094255 |

## Training curves, compute and stability

Each checkpoint row uses the fixed training and held-out windows. The loss on the 5,000 actual batches, the pre-clipping gradient norms and all checkpoint evaluations are in the JSON. [One-pass data](../artifacts/shakespeare-one.json) · [Looped data](../artifacts/shakespeare-looped.json) · [Stock tied data](../artifacts/shakespeare-stock.json)

| Step | One pass train | One pass held out | Looped train | Looped held out | Stock tied train | Stock tied held out |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 4.1871 | 4.1873 | 4.1839 | 4.1840 | 4.1844 | 4.1848 |
| 500 | 2.2434 | 2.2717 | 2.2658 | 2.2804 | 2.1934 | 2.2267 |
| 1000 | 1.9110 | 2.0243 | 2.0581 | 2.1127 | 1.8767 | 1.9911 |
| 2000 | 1.6455 | 1.8130 | 1.8200 | 1.9376 | 1.6322 | 1.7919 |
| 3000 | 1.5610 | 1.7455 | 1.7064 | 1.8431 | 1.5295 | 1.7070 |
| 4000 | 1.5212 | 1.7120 | 1.6522 | 1.8031 | 1.4753 | 1.6699 |
| 5000 | 1.5065 | 1.6983 | 1.6268 | 1.7843 | 1.4502 | 1.6541 |

| Model | Block applications / step | Total block applications | Update time (min) | Wall time (min) |
| --- | ---: | ---: | ---: | ---: |
| One pass | 1.0000 | 5,000 | 23.48 | 24.91 |
| Looped | 4.0022 | 20,011 | 80.14 | 83.07 |
| Stock tied | 4.0000 | 20,000 | 72.33 | 75.14 |

The looped model's unroll counts were 1,662 at pass 3, 1,665 at pass 4 and 1,673 at pass 5. It ran 163,930,112 token-block applications, compared with 163,840,000 for stock tied and Book 1's four-block recipe. The count excludes the extra readouts and stream normalization.

All runs completed with finite batch losses, gradient norms and fixed-window losses. For a common spike check, each eligible post-warmup batch is compared with the median of its previous 50 batches at the same unroll length. The table also counts increases between consecutive held-out evaluations.

| Model | Largest batch loss excess (nats) | Excess ≥0.25 | Peak gradient norm | Clipped updates | Held-out increases |
| --- | ---: | ---: | ---: | ---: | ---: |
| One pass | 0.1187 (step 4,660) | 0 | 3.83 | 376 | 0 |
| Looped | 0.0818 (step 1,060) | 0 | 6.98 | 1,951 | 0 |
| Stock tied | 0.1131 (step 2,544) | 0 | 2.47 | 4,411 | 0 |

## State and prediction settling

The input is the exact 83-character “Good morrow, neighbour” passage used by Book 1's attention figure, starting at character 3 of the held-out tail. Each row ends at pass *k* and measures the Frobenius norm of *Xₖ − Xₖ₋₁* over the whole stream. The centred version subtracts the mean within each character's column before taking the norm.

| Pass | One raw | One centred | Looped raw | Looped centred | Stock raw | Stock centred |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 119.04 | 118.4 | 103.2 | 103.08 | 98.319 | 98.258 |
| 2 | 102.6 | 102.28 | 68.714 | 68.714 | 53.841 | 53.696 |
| 3 | 106.79 | 106.46 | 34.402 | 34.402 | 55.928 | 55.526 |
| 4 | 106.47 | 106.2 | 19.745 | 19.745 | 51.948 | 51.545 |
| 5 | 111.64 | 111.34 | 12.343 | 12.343 | 47.067 | 46.648 |
| 6 | 116.72 | 116.39 | 7.6174 | 7.6174 | 43.523 | 43.152 |
| 7 | 120.48 | 120.12 | 4.7543 | 4.7543 | 41.915 | 41.577 |
| 8 | 123.33 | 122.96 | 3.4784 | 3.4784 | 41.811 | 41.418 |
| 9 | 125.66 | 125.29 | 2.954 | 2.954 | 40.189 | 39.759 |
| 10 | 127.57 | 127.18 | 2.3654 | 2.3654 | 38.472 | 38.005 |
| 11 | 129.05 | 128.65 | 1.8288 | 1.8288 | 37.391 | 36.889 |
| 12 | 130.2 | 129.79 | 1.5435 | 1.5435 | 36.443 | 35.91 |
| 13 | 131.11 | 130.7 | 1.2306 | 1.2306 | 35.764 | 35.217 |
| 14 | 131.86 | 131.44 | 1.0581 | 1.0581 | 35.272 | 34.72 |
| 15 | 132.51 | 132.08 | 0.88302 | 0.88302 | 34.906 | 34.346 |
| 16 | 133.07 | 132.64 | 0.8332 | 0.8332 | 34.594 | 34.018 |

A position settles at the first pass whose argmax agrees with every subsequent argmax through pass 16. A position whose last change arrives at pass 16 can still change again beyond the measured horizon.

| Model | Settled by pass 4 | Settled by pass 8 | Last changed at pass 16 |
| --- | ---: | ---: | ---: |
| One pass | 14 | 69 | 0 |
| Looped | 62 | 78 | 0 |
| Stock tied | 9 | 26 | 5 |

Late positions are listed below as one-based character offsets in the passage, grouped by the pass at which the final argmax change arrived. ␠ denotes a space and ↵ a newline. Each model's JSON includes the full per-position argmax trajectory and the next-character target.

- **One pass.** Pass 9 at 6 `O`, 14 `m`, 22 `n`, 38 `t`, 58 `m`, 69 `g`, 82 `.`. Pass 10 at 17 `r`, 67 `e`. Pass 11 at 20 `,`. Pass 12 at 27 `b`, 40 `.`, 55 `o`. Pass 13 at 39 `a`.
- **Looped.** Pass 9 at 35 `t`. Pass 10 at 56 `d`. Pass 11 at 58 `m`. Pass 15 at 39 `a`, 57 `␠`.
- **Stock tied.** Pass 9 at 17 `r`, 39 `a`, 56 `d`, 59 `o`. Pass 10 at 9 `G`, 13 `␠`, 14 `m`, 51 `:`, 58 `m`, 70 `h`. Pass 11 at 22 `n`, 23 `e`, 27 `b`, 29 `u`, 30 `r`, 32 `B`, 36 `i`, 79 `m`. Pass 12 at 15 `o`, 19 `w`, 37 `s`, 42 `↵`, 65 `␠`, 66 `n`, 68 `i`. Pass 13 at 1 `G`, 18 `o`, 21 `␠`, 33 `a`, 34 `p`, 47 `I`, 49 `T`, 77 `r`. Pass 14 at 6 `O`, 10 `o`, 12 `d`, 26 `h`, 43 `B`, 50 `A`, 55 `o`, 69 `g`, 74 `r`, 75 `␠`, 81 `o`. Pass 15 at 4 `M`, 8 `↵`, 24 `i`, 25 `g`, 72 `o`, 73 `u`, 76 `G`, 82 `.`. Pass 16 at 2 `R`, 16 `r`, 48 `S`, 60 `r`, 67 `e`.

## Samples at temperature 1

Each sample starts with a newline and adds 240 characters using `Xoshiro(2026)`. For each continuation, the model reads the latest context through the indicated number of passes at every generated character. The blocks show the continuation after the common prompt. The symbol ␠ marks a trailing space.

### One pass

Pass 1

~~~text
MENENIUS:
How ward, now, that is the proclosation,
Unvast: I,
If not think beloved--
Tybeeing him and of platice tall and causio.

HENRY BOLINA; may'd, gave, graves muet, undeed but return
To must friends?way and ristrengther's made.

NORFO
~~~

Pass 2

~~~text
RILLA:
ThuLnTA:
AlonemeUs;
ATO:
Rh,
Mysseld:
TheRl?

WARRA

Z
HeR:
Olune!
AMAO.

HiGR:s'eSTO:
Heh,
TheRRRRRRVE
IAeARuI: Isa thrha'ly osmel'emr twi
lY to Hoctense mysssoveymn,
DYa stvelihol! nomis mema
ssppy off,
CHiSto.

f:
Notcrmaon'swaiFu
~~~

Pass 4

~~~text
RARRGRII
RARoElIa''.
RoutISARARARUhAR
STAyiEAeAR
PHRRiARARI
sARI
HeROrUR
HRRIIIlUioARRy ROLhAe;R RARI
MAhieUaoAReoARlaAR sHARRhI

SRIn,
FOIeasa?R AsOhoi,
TEsHeiEnHwRTU ssthRT
CRHHiTLhRhAii;R STUrhhRo,
sTOxyEeR RRI
Rue:
CitR ROnlUL
WhAThLeRo
~~~

Pass 8

~~~text
RRARRRiUBR RRAieREi:
RURUR␠
RARiERRRO RRRieRE;RR RRU U sR:

s
HTMR:
RURE
HARCRRRMRURE-y RRARRA RARARLA,'RRRHieRU
RARLA R VA
WRUR␠
RARIRH O
GRU
WhRA RAR
HAR
sHeiARRhREURwRRU RHMRRREURRRU?Hi RR?
RE
GRU
'VRAYs
IL:
GRE RRRR RrReReReReARi.
RAR s
~~~

### Looped

Pass 1

~~~text
MISe foulokng w?
Anostoutin orenlorederr win tor onrby ur s y
Calengithifan Rerimes goy pe m, he idis o-oreatiar s
Of ar t bupus Cand, ham f w tor thaund s want tevear: prt ld mairsponorur ssinor.
Comawhewweaike armyot ary theshe Mye ng l w
~~~

Pass 2

~~~text
Mark inquies,
We fore: trul not's of thoour but Solldraw?
'Ay go, some, and ason's, fry rear hed,
And never; ten tank ancomes wom sine.

EDONTCUS:
What if I'll.
Bether your leas, my onget?
In my death witty in of thour boy slesee as,
Rome w
~~~

Pass 4

~~~text
PRINCE:
Is that death: would me to list:
You to and holy colvant muster.

GLOUCESTER:
yet did is latter the rian?

MON:
Ay, why, Come, ghed frayeid; as fashords sive,
Fring me of striglowed time enevery loyal come;
To cary these day disck w
~~~

Pass 8

~~~text
QUEN MERGINGIET:
Dother; that his meen thy injusenciougion.
Warwhick, that a too premyy of leed.

BUCKINGHAM:

Nay oth, I'll out and not happy to arm the
faullieneve shunkn no'er prosis? have my death withy
That and ureany somford to no may
~~~

### Stock tied

Pass 1

~~~text
QUniffourugur'spanourusutur! mmpsurinstugurinust mmysuest,S t murjurntur: mbllllinugesy me ncourAn,fullastinus mbu sun,Baugburyse sthst:
ABnev.urifurOurien,t, sturve:'suryspllllffusur;: tuburinur:
BunEurAsy. e,:
Anevoulefunge;Win:
Theff: Bu
~~~

Pass 2

~~~text
Inde mournarrOue mestirny; Inenourdantes,araint:
Heret tren warounst:
Pond:
Ande ourdsy stan,
An marranIt:
And mas stecont myest,
Binke, laint tre trest:
May my thy, mextys, ear.
Hist trun'st.
Sot my louryy.
ABENCHOTONCEYour:
Then trenor:
W
~~~

Pass 4

~~~text
LUCIO:
Very say deep, too keep to reposicle,
With look view years to the fearlow's,
How shall die and of none,
The preciase too, God more alleaving,
But I am it shrighty to me nears.

ROMEO:
Sir, dishitinsy in secrets!

Lue now! Nay, my lay
~~~

Pass 8

~~~text
Oiimaangenesyas
Ielllooutiebled soooosouewshrrun
Toshreryas
gaipendlnnecadd JulioguIfnw
Trelahdrieaksahasooknn oes
TugFiawaco'tibaihgorhed
Hat'sslaosouleaiir hudoiyhlnnuhgubdiah airly towt
Tulesuhefyemrhouyabah:
Cootiueepy skpreeeast noemaw
~~~

## Verdict

**What carries over.** The same tied cell runs on a character vocabulary of 65 and a context of 256. The fresh-input residual, fixed stream LayerNorm, boundary-only start, shared readout at every pass and mean loss on a sampled unroll all carry over from the specimen. On the held-out passage, the looped model's centred state step falls from 19.75 at pass 4 to 0.83 at pass 16. By pass 8, 78 of 83 top predictions have made their final change within the measured horizon. Its pass-eight sample still has speaker headings and dialogue.

**What changes.** The full causal mask gives queries global access to their preceding stream columns as early as pass 1. The specimen's one-position-per-pass information cone and exact parity loss floor have no counterpart in this setup. Tiny Shakespeare uses sinusoidal positions, zero predecessor bias and observed next characters at every supervised pass. The looped model improves from 2.4923 to 1.7843 nats between passes 1 and 4, then loses ground on every successive pass through 16. Its minimum is 0.0861 nats above the one-pass model and 0.1302 above stock tied. The state moves less as the held-out loss rises.

**What a chapter could claim.** One 215,233-parameter block can learn to generate Shakespeare-like character sequences, and the anchored, normalized loop preserves far more of its predictive quality under extra passes than either stock recurrence in this trial. Stock tied gives the lowest loss of the three new runs at its trained depth. Both tied models fall short of Book 1's 1.5313, and the specimen-style loop has the highest minimum loss of the three. The evidence supports a chapter about the different effects of iteration on state settling and prediction quality. Reliability across seeds, convergence beyond pass 16, the separate effect of each architectural change and the width-two language model remain open.

## Round 2 · Full-input start and the window of two

Both models start with the token embedding plus sinusoidal position at every column. The first residual adds the attention output to that same input at every pass, and a fixed LayerNorm follows the MLP residual. The shared readout scores the observed next character at every pass. Each update minimizes the mean of those losses. Both runs use zero predecessor bias and the same data, optimizer, parameter count, Float32 precision, batch size, training length and thread counts as Round 1. There is one seed per run, 1234.

| Run | Attention | Passes per training batch | Expected block applications per update | Checkpoint selection pass |
| --- | --- | --- | ---: | ---: |
| A · Full input | Full causal | Uniform over 3–5 | 4 | 4 |
| B · Window of two | Self and predecessor | Uniform over 1–8 | 4.5 | 8 |

The initial stream is the sole recipe change from the Round 1 loop to Run A. Run B narrows the attention and varies the training depth so that every supervised depth up to eight occurs in training. At pass 8 the window model can use up to nine characters. The schedule expects 22,500 block applications in 5,000 updates, or 184,320,000 token-block applications, 12.5% above a four-pass schedule. The two-key attention reduces the work within each block application relative to full causal attention. Checkpoints are selected by minimum loss at the stated pass on Book 1's fixed held-out windows.

Run from the repository root. The raw logs and checkpoints use the same git-ignored `loop-trial` directory as Round 1, with subdirectories `fullstart` and `window`.

```sh
set -e
julia --project -t 12 loop/scripts/shakespeare_gradcheck.jl
for model in fullstart window; do
  julia --project -t 12 loop/scripts/shakespeare_train.jl "$model"
done
julia --project -t 12 loop/scripts/shakespeare_export.jl fullstart window
julia --project loop/scripts/shakespeare_ngram.jl
```

Before training, the Round 1 Float64 finite differences were repeated with both new configurations. They checked all model parameters with arbitrary logit covectors and with the mean of the three per-pass cross-entropies, using a step of 10⁻⁶. The suite also checked width-two boundary weights, perturbed the left edge of the cone and its exterior, and checked isolation across packed sequences. All 29 checks passed.

| Run | Relative logit gradient error | Relative loss gradient error |
| --- | ---: | ---: |
| A · Full input | 2.33 × 10⁻¹⁰ | 1.11 × 10⁻⁹ |
| B · Window of two | 4.46 × 10⁻¹⁰ | 7.44 × 10⁻¹⁰ |

### Training curves, compute and stability

Losses use the fixed training and held-out windows at pass 4 for A and pass 8 for B. Both checkpoints were selected at step 5,000.

| Step | A train | A held out | B train | B held out |
| --- | ---: | ---: | ---: | ---: |
| 0 | 4.1821 | 4.1824 | 4.1868 | 4.1869 |
| 500 | 2.2491 | 2.2652 | 1.8470 | 1.9405 |
| 1000 | 2.0166 | 2.0727 | 1.6736 | 1.8231 |
| 2000 | 1.7287 | 1.8614 | 1.5537 | 1.7144 |
| 3000 | 1.6209 | 1.7725 | 1.5027 | 1.6646 |
| 4000 | 1.5647 | 1.7319 | 1.4723 | 1.6420 |
| 5000 | 1.5421 | 1.7145 | 1.4587 | 1.6318 |

| Run | Block applications / update | Total block applications | Token-block applications | Update time (min) | Training wall time (min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A · Full input | 4.0022 | 20,011 | 163,930,112 | 86.29 | 89.44 |
| B · Window of two | 4.4998 | 22,499 | 184,311,808 | 68.19 | 71.74 |

Run A drew 1,662, 1,665 and 1,673 batches at depths 3, 4 and 5, exactly as in the original loop. Run B drew 627, 645, 587, 631, 624, 630, 651 and 605 batches at depths 1–8. Each block application in both runs includes a stream normalization and a supervised readout. In B, all 5,000 updates supervise pass 1 and 605 supervise pass 8.

Power Saver was recorded at both ends of A and at the start of B. Performance was recorded at the end of B. Run A began on battery and continued on AC after a 4 h 54 min laptop suspend. It took 6 h 23 min from settings capture to the final checkpoint. The training wall timer uses the same awake-time measurement as Round 1. The two timings reflect the observed system conditions.

Both runs completed with finite losses and gradients. As in Round 1, the spike check compares each eligible post-warmup batch with the median of its 50 most recent predecessors at the same unroll length.

| Run | Largest batch loss excess (nats) | Excess ≥0.25 | Peak gradient norm | Clipped updates | Held-out increases |
| --- | ---: | ---: | ---: | ---: | ---: |
| A · Full input | 0.1022 (step 2,544) | 0 | 3.15 | 1,062 | 0 |
| B · Window of two | 0.0924 (step 3,637) | 0 | 1.99 | 70 | 1 |

### Run A · What the stabilized recipe costs

At pass 1, the full-input start reduces held-out loss from 2.4923 to 1.8828. At the matched pass 4, it reduces loss from 1.7843 to 1.7145, leaving A **0.0604 nats above stock tied** and **0.0162 above the one-pass control**. At A's lowest-loss pass, pass 3, the loss is **1.7102** and the gaps are **+0.0561** and **+0.0120**, respectively. These are the measured gaps for the combined recipe. The trial compares the effect of the starting stream within that recipe and leaves the individual effects of anchoring, normalization and supervision open. [Run A data](../artifacts/shakespeare-fullstart.json)

### Run B · Matching character context to n-gram order

In `initial_stream`, `start = :embedded` puts character `t` in the stream at its own column at pass zero. With `width = 2`, `window_mix` reads only the current and preceding previous-pass columns, with a singleton at the start of each sequence. The fresh input residual, LayerNorms, MLP and readout operate column by column. Induction therefore gives the character interval `max(1, t-k)…t` at pass `k`. The target at column `t` is character `t+1`. That yields a maximum of `k+1` context characters and order `k+2`, from a trigram at pass 1 through an 18-gram at pass 16. Within a window, the predictor has at most `min(t, k+1)` characters of context. [Window implementation](../src/window.jl) · [Input and target construction](../../src/train.jl)

The reference uses interpolated Kneser–Ney with one absolute discount shared across the discounted levels of each fitted model. The top order uses occurrence counts, lower orders use distinct left-extension counts, and the unigram base normalizes the number of distinct predecessors of each character. At each available order, a history with total count `C` and `U` distinct continuations assigns `max(c-D, 0)/C` to a target of count `c` and adds `D·U/C` times the probability from the shorter suffix. Unseen and unavailable histories drop to the shorter suffix. The text is counted as one character stream. Evaluation uses context within the window only, so the first prediction starts with exactly the first input character and no padding. [Chen and Goodman, §4.1.6](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-10-98.pdf#page=23) · [Estimator and window scoring](../scripts/shakespeare_ngram.jl)

For each order, the discount is chosen from 0.01 to 0.99 in steps of 0.01. Counts for tuning use characters 1–953,662 of the original training split. The remaining 50,193 training characters supply 256 tuning windows of length 256, drawn with replacement using `Xoshiro(1237)`. The smallest loss wins, with ties going to the smaller discount. The counts are then refit on all 1,003,855 training characters. Only this final model is scored on Book 1's 65,536 fixed held-out predictions using `Xoshiro(1236)`.

The ladder is a fitted reference line. A lower bound or Bayes risk would require a separate argument about the true conditional distribution. The transformer also receives absolute sinusoidal positions within each window, while this estimator uses the character suffix alone.

### Run B · Loss against the n-gram ladder

**Run B lags the reference at every pass.** The gap is 0.0187 nats at pass 1, 0.0650 at pass 2 and 0.0905–0.1022 across passes 3–16. Both curves improve sharply as the first few characters enter the context and flatten around passes 5–6. B reaches its minimum of **1.6256** at pass 6. The ladder reaches **1.5303** at order 7, which matches pass 5. At higher orders it mostly relies on shorter suffixes. At order 18 the longest context has a counted continuation at 0.68% of eligible predictions. [Run B data](../artifacts/shakespeare-window.json) · [N-gram data](../artifacts/shakespeare-ngram.json)

| Pass | Max context | Order | Discount | N-gram loss | B − n-gram | Comparison | Full context seen |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 1 | 2 | 3 | 0.96 | 2.036140 | +0.018652 | Lags | 99.84% |
| 2 | 3 | 4 | 0.85 | 1.721853 | +0.064987 | Lags | 98.72% |
| 3 | 4 | 5 | 0.87 | 1.567611 | +0.102216 | Lags | 96.17% |
| 4 | 5 | 6 | 0.91 | 1.534396 | +0.099900 | Lags | 90.44% |
| 5 | 6 | 7 | 0.93 | 1.530271 | +0.095402 | Lags | 79.75% |
| 6 | 7 | 8 | 0.95 | 1.533361 | +0.092194 | Lags | 65.56% |
| 7 | 8 | 9 | 0.96 | 1.537238 | +0.090460 | Lags | 50.40% |
| 8 | 9 | 10 | 0.97 | 1.538525 | +0.093272 | Lags | 36.02% |
| 9 | 10 | 11 | 0.97 | 1.539119 | +0.094096 | Lags | 24.48% |
| 10 | 11 | 12 | 0.97 | 1.539144 | +0.094934 | Lags | 16.06% |
| 11 | 12 | 13 | 0.97 | 1.539645 | +0.095982 | Lags | 10.15% |
| 12 | 13 | 14 | 0.97 | 1.539800 | +0.096452 | Lags | 6.27% |
| 13 | 14 | 15 | 0.97 | 1.539969 | +0.097770 | Lags | 3.88% |
| 14 | 15 | 16 | 0.98 | 1.540348 | +0.097750 | Lags | 2.25% |
| 15 | 16 | 17 | 0.98 | 1.540375 | +0.098724 | Lags | 1.22% |
| 16 | 17 | 18 | 0.98 | 1.540413 | +0.098304 | Lags | 0.68% |

“Full context seen” is the fraction of positions with room for the full context whose longest history has at least one counted continuation in the refit training text.

### Run B · Perturbing the trained cone

The probe uses Book 1 held-out windows 1, 86 and 256 with query positions 33, 128 and 231, respectively. It replaces one input character at a time with a space, or with `e` if the input is already a space. At each of passes 1–16, changes at positions `t−k−1` and `t−k−2` produced exactly zero logit difference in all **96** comparisons. Changes at `t−k`, the left edge inside the cone, shifted logits in all **48** comparisons. The table gives the largest absolute change across vocabulary logits at selected passes. [All perturbations](../artifacts/shakespeare-window.json)

| Pass | Window 1, query 33 | Window 86, query 128 | Window 256, query 231 |
| --- | ---: | ---: | ---: |
| 1 | 7.320 | 1.297 | 9.104 |
| 4 | 1.826 | 0.1197 | 0.6499 |
| 8 | 0.7368 | 0.03683 | 0.03012 |
| 16 | 0.002283 | 0.0004239 | 0.0006306 |

### Settling on “Good morrow, neighbour”

Each row gives the Frobenius norm of the state step ending at that pass over the same 83-character held-out passage as in Round 1. The centred norm removes each column's mean before taking the norm.

| Pass | A raw | A centred | B raw | B centred |
| --- | ---: | ---: | ---: | ---: |
| 1 | 105.56 | 98.236 | 109.55 | 102.5 |
| 2 | 39.551 | 39.551 | 57.911 | 57.911 |
| 3 | 21.662 | 21.662 | 38.088 | 38.088 |
| 4 | 12.674 | 12.674 | 26.911 | 26.911 |
| 5 | 8.704 | 8.704 | 19.483 | 19.483 |
| 6 | 5.8963 | 5.8963 | 15.087 | 15.087 |
| 7 | 4.0551 | 4.0551 | 11.586 | 11.586 |
| 8 | 2.7503 | 2.7503 | 10.341 | 10.341 |
| 9 | 1.8609 | 1.8609 | 9.0182 | 9.0182 |
| 10 | 1.2681 | 1.2681 | 7.5718 | 7.5718 |
| 11 | 0.84479 | 0.84479 | 6.1057 | 6.1057 |
| 12 | 0.60476 | 0.60476 | 5.0226 | 5.0226 |
| 13 | 0.46488 | 0.46488 | 3.8426 | 3.8426 |
| 14 | 0.37907 | 0.37907 | 3.2227 | 3.2227 |
| 15 | 0.32778 | 0.32778 | 2.3886 | 2.3886 |
| 16 | 0.28866 | 0.28866 | 1.9059 | 1.9059 |

The counts use the last change of each argmax through pass 16.

| Run | Settled by pass 4 | Settled by pass 8 | Last changed at pass 16 |
| --- | ---: | ---: | ---: |
| A · Full input | 68 | 81 | 0 |
| B · Window of two | 63 | 74 | 0 |

One-based offsets and input characters for positions whose final change arrived after pass 8.

- **Run A.** Pass 9 at 34 `p`. Pass 11 at 63 `w`.
- **Run B.** Pass 9 at 23 `e`, 26 `h`, 75 `␠`. Pass 10 at 37 `s`. Pass 11 at 27 `b`, 70 `h`. Pass 12 at 71 `b`, 78 `e`. Pass 13 at 47 `I`.

A's centred steps fall below 1 at pass 11. B's are still 1.91 at pass 16, three passes after its final measured argmax change. Both models keep moving in state space as their top predictions settle on this passage.

### Samples at temperature 1

Each sample adds 240 characters to the same newline prompt with `Xoshiro(2026)`. The code blocks show the continuation.

#### A · Full input

Pass 1

~~~text
LISA:
Onding I'll not took and strone.
A will, marry way comp'd dispitle he but gor gry plance senatile the stater
To blaught wom recons cancoly.

Second say hopedured your king and sit.
FRIAR VI:
Hoves! ywas and sprove own.

Secomion? Do s
~~~

Pass 2

~~~text
MONTAGUS:
How we good, uside me; then, thus brote,
Why will warrily our be cats' our tyou come,
And the to his off succh then won sinder fairs
You upon ill weet thy is your king!
My loulth orngs, boy, whest is night; of nurthes?

AUTOLY:
Ay
~~~

Pass 4

~~~text
Last Your extand disposty.

LEONTES:
Romay:
By, Help yet company loss: the's to foul tymile.

GLOUCESTER:
My parer
To blaw by won the piggaget to Stenly Juck holots,
Shuld be did proves will; his her, whety in me ristle
Twere upon the raids
~~~

Pass 8

~~~text
LY CAPUS:
How we founds, that his not stay;
And I hedves compalle, provice a tongue
Isy shall he ight of mimpt sep to an t contreated
That give; law, when I'll do you, your leave out his.

PRINCE EDWARD:
Yrind made use any therefolly moincl
~~~

#### B · Window of two

Pass 1

~~~text
LUMILLOUCK:
SIO:
Foreptry, I kingre-forray;
As nand;
YORK:
Ay him.

My se.

But good ty pear, go, I'll but isf, Tuart--a thavest God mus eam. Ause thave an weet thy loourst none coure, my anionst her,
Youty in of thour boy, moung ass no may
~~~

Pass 2

~~~text
LEONTESBY:
What deepersorsabelinghame: thy have Ment, I'll away.

NORTHUMBERLE:
Her hoy sight is impon but is I be
burdenumbless I hat in affolluncle unfascreat thy is your king and retits stoo?

GLOUCESTYwawd nears? no, youngroughter:
And,
~~~

Pass 4

~~~text
LUCIO:
Why to undershipsion of strike stay: as man, drawd thy foot, other hagot;
My hoy piece is if the son, take Born screatest I have dead wation:
Come say my quoth guispason,
And sistinent:
Villane: muty maid and will your time
Will Waly
~~~

Pass 8

~~~text
LUCIO:
Why to understisure and rish'd to words; I hear up thy foot, other had; many hoy of me, of Nover the roar take an of thy care of be her trogs' time,
Your pity shut! I have prouds, sentented it eyesty in peace; peacter, which she an u
~~~

### Verdict after Round 2

The window-of-two run has the lowest held-out loss of the 215,233-parameter neural models in this trial. It improves on stock tied by 0.0286 nats at pass 6 and still leads by 0.0223 at the pass-8 checkpoint criterion. The minimum is 0.0943 nats above Book 1's larger untied model. The width-two mask and the longer sampled unroll changed together between A and B, so their separate contributions remain open.

A Shakespeare chapter can show the toy model's information cone in trained language-model logits and compare the resulting predictors against a conventional model at every depth. It can also show the distinction between state settling and prediction quality. At pass 16, A's loss is 0.0718 nats above its minimum, and B's is 0.0132 above its minimum, even as both state steps continue to shrink on the passage. The comparison uses one seed per run and measures settling through pass 16.

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

| Pass | One pass | Looped | Stock tied |
| --- | ---: | ---: | ---: |
| 1 | 1.698270 | 2.492265 | 3.539166 |
| 2 | 3.569208 | 1.883311 | 2.751502 |
| 3 | 4.239137 | 1.786466 | 1.990949 |
| 4 | 4.774442 | 1.784327 | 1.654106 |
| 5 | 5.187675 | 1.800816 | 1.917334 |
| 6 | 5.500590 | 1.820031 | 2.488327 |
| 7 | 5.755695 | 1.835376 | 3.090388 |
| 8 | 5.972504 | 1.848174 | 3.600285 |
| 9 | 6.154857 | 1.857968 | 4.005468 |
| 10 | 6.307955 | 1.865680 | 4.320483 |
| 11 | 6.438417 | 1.871944 | 4.565818 |
| 12 | 6.551881 | 1.877576 | 4.762143 |
| 13 | 6.652435 | 1.882012 | 4.925695 |
| 14 | 6.742736 | 1.885670 | 5.067607 |
| 15 | 6.824457 | 1.888624 | 5.195070 |
| 16 | 6.898866 | 1.891412 | 5.312422 |

| Model | Selected step | Selection pass | Lowest-loss pass | Gap to Book 1 at lowest-loss pass |
| --- | ---: | ---: | ---: | ---: |
| One pass | 5000 | 1 | 1 | +0.166970 |
| Looped | 5000 | 4 | 4 | +0.253027 |
| Stock tied | 5000 | 4 | 4 | +0.122806 |

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


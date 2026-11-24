# Reading the tower of worlds off from the text (`grow/lumpable/`)

**Verdict: needs more.** The bookkeeping is exact and cheap, and it gives a usable measure of
"new structure per level". But no non-trivial coarse world is closed, neither on the specimen nor
on Shakespeare, and minimising the defect alone builds a poor tower.

No network is trained here: counting and exact entropies only, in nats. Julia 1.x, standard
library only, 2 threads.

    julia grow/lumpable/specimen.jl       # about 2 s   -> specimen.out, specimen.tsv
    julia -t2 grow/lumpable/tower.jl      # about 25 s  -> tower.out, tower_*.tsv, levels.tsv, tower.svg

## The quantities

π is a partition of the characters, Y the next character, H the fine history.

| | | |
|---|---|---|
| class floor | F = H(πY \| πH) | the best loss of a model that lives in the coarse world |
| defect | D = I(πY ; H \| πH) = H(πY \| πH) − H(πY \| H) | zero exactly when the coarse world is closed |
| split entropy | S = H(Y \| πY, H) | what the fine level adds |

Chain rule: H(Y | H) = F − D + S. Between two levels of a tower, coarse c and fine f, the same
identity reads **F_f = F_c − (relative defect) + (relative split)**, with relative defect
I(π_c Y ; π_f H | π_c H) and relative split H(π_f Y | π_c Y, π_f H). This is the form that matters
for growing: the child's floor is the parent's floor, minus what the parent could not see, plus
the new term.

## 1. The specimen, `to be or not to be`

Samples as in the repository: 10 windows of length 8, each position predicting the next character
from the window prefix, N = 80 pairs, histories of length 1 to 8. All 877 partitions of the seven
characters were enumerated.

- Fine floor 0.058526640339053, equal to (2 ln 2 + 3 ln 3)/80 to the last digit.
- Chain rule: largest error over all 877 partitions 6.9e-18.
- **No partition into 2, 3 or 4 classes is closed** (nor into 5 or 6). Only the trivial ones are.

| classes | partitions | smallest defect | its class floor | its split | best partition (one of several) |
|---|---|---|---|---|---|
| 2 | 63 | 0.1494 | 0.1733 | 0.0347 | `␣benrt\|o` |
| 3 | 301 | 0.0932 | 0.1517 | 0 | `␣b\|enr\|ot` (six tie) |
| 4 | 350 | 0.0585 | 0.1171 | 0 | `␣\|ber\|nt\|o` (several tie) |
| 5 | 140 | 0.0347 | 0.0932 | 0 | `␣\|be\|nr\|o\|t` |
| 6 | 21 | 0.0173 | 0.0759 | 0 | `␣\|b\|er\|n\|o\|t` |

- Nothing is even nearly closed. The smallest defect at 2 to 4 classes is one to 2.6 times the
  whole fine floor, and it is 50 to 86 % of the class floor. The text is almost deterministic
  given the fine history, so nearly everything a coarse model cannot predict is defect.
- **Vowel | consonant | space ranks 50th of 301** three-class partitions by defect (0.1105; 32
  partitions are strictly better, the best has 0.0932, the median 0.1287, the worst 0.2079). Its
  class floor, 0.1517, equals that of the best ones; it loses on split entropy (0.0173 against 0).
  It is unremarkable for this text.

## 2. Tiny Shakespeare (1,115,394 characters, 65 characters)

The corpus is not in the repository; `scripts/common.jl` downloads it to `artifacts/data/`, and
the copy already on this laptop was used. Greedy from 65 classes down: merge the two classes
whose merger leaves the smallest defect, ties broken by the smallest loss of I(πY ; πH).

**History = two previous characters (trigram counts, 11,556 distinct trigrams).** Fine floor 1.9076.

| classes | class floor | defect | split | defect / class floor |
|---|---|---|---|---|
| 65 | 1.9076 | 0 | 0 | |
| 32 | 1.9248 | 0.0418 | 0.0246 | 2 % |
| 16 | 1.8798 | 0.2226 | 0.2504 | 12 % |
| 8 | 1.2939 | 0.2771 | 0.8908 | 21 % |
| 4 | 0.4789 | 0.1309 | 1.5595 | 27 % |
| 2 | 0.1780 | 0.0432 | 1.7728 | 24 % |

The classes (`␣` space, `\n` newline):

- 32: `!$&3QSZq` `.?JPWX` `-RVY` `Ujpz` `;KO` `Ccx` `EFf` `'G` `,B` `:H` `Av` `Db` `Lg` `Mm` `Nk` `Tw`, and alone `\n ␣ I a d e h i l n o r s t u y`
- 16: `!$&.3?JPQSTWXZqw` `,:;BDHKOb` `EFMUfjmpz` `\n-CRVYcx` `'GLg` `AIv` `Nhk` `dsy` `iu` `lr`, and alone `␣ a e n o t`
- 8: `'EFGLMNUdefghjklmnprstyz` `!$&.3?JPQSTWXZqw` `,:;BDHKOb` `\n-CRVYcx` `AIv` `aiu` `␣` `o`
- 4: `\n␣'-CEFGLMNRUVYacdefghijklmnoprstuxyz` `!$&.3?JPQSTWXZqw` `,:;BDHKOb` `AIv`
- 2: everything else against `!$&.3?JPQSTWXZqw`

**History = one previous character (bigram counts).** Fine floor 2.4526.

| classes | class floor | defect | split | defect / class floor |
|---|---|---|---|---|
| 32 | 2.3539 | 0.0264 | 0.1250 | 1 % |
| 16 | 1.9340 | 0.0723 | 0.5909 | 4 % |
| 8 | 1.2536 | 0.0696 | 1.2686 | 6 % |
| 4 | 0.3762 | 0.0604 | 2.1368 | 16 % |
| 2 | 0.1710 | 0.0246 | 2.3062 | 14 % |

Classes at 4: capitals with `&jq` | space, lower case, `$'-` | `!,.3:;?` | newline. At 2:
`!,.3:;?` against the rest. The full lists for both runs are in `tower.out`.

The first five merges of the trigram tower cost nothing (defect below 1e-9): they join rare
characters (`$ & 3 Z`, `! Q`, `. J`). The defect is weighted by frequency, so the criterion merges
rare characters first, and the trigram classes are hard to read (`Tw`, `Mm`, `Nk` pair a capital
with its lower case, but `!$&3QSZq` is a bin of rare things).

### How this differs from Brown clustering

Brown merges the two classes whose merger loses the least class bigram information I(πY ; πH),
the likelihood of a class bigram model. It asks that the classes stay *informative*. The defect
asks that the coarse world be *closed*: the fine history must add nothing about the next class.
The fine history enters the criterion here and never enters Brown's. The two pull apart: a world
with one class is perfectly closed and says nothing. Both towers were built with one character of
history, for comparison:

| classes | defect tower: floor, defect | Brown tower: floor, defect |
|---|---|---|
| 32 | 2.354, 0.026 | 2.344, 0.029 |
| 16 | 1.934, 0.072 | 2.006, 0.099 |
| 8 | 1.254, 0.070 | 1.375, 0.149 |
| 4 | 0.376, 0.060 | 0.841, 0.106 |
| 2 | 0.171, 0.025 | 0.300, 0.038 |

The defect tower has about half the defect, but its coarse worlds also contain less. As a share
of the class floor, Brown is no worse (13 % against 16 % at four classes), and its classes are
the readable ones: at 4, capitals and newline | lower-case consonants | space, vowels, `'-` |
punctuation; at 8 the vowels, `h`, space and newline stand apart.

## 3. Is the new structure per level bounded?

Per level, coarse → fine, with F_fine = F_coarse − relative defect + relative split (checked to
4e-14 at every level):

| level | defect tower, trigram: rel. defect, rel. split | defect tower, bigram | Brown tower, bigram |
|---|---|---|---|
| 1 → 2 | 0, 0.178 | 0, 0.171 | 0, 0.300 |
| 2 → 4 | 0.006, 0.307 | 0.002, 0.207 | 0.002, 0.543 |
| 4 → 8 | 0.046, 0.861 | 0.032, 0.909 | 0.035, 0.568 |
| 8 → 16 | 0.138, 0.724 | 0.037, 0.718 | 0.086, 0.717 |
| 16 → 32 | 0.186, 0.231 | 0.053, 0.473 | 0.076, 0.415 |
| 32 → 65 | 0.042, 0.025 | 0.026, 0.125 | 0.029, 0.137 |

- **With the defect criterion the new structure is concentrated.** In the trigram tower, the two
  levels 4 → 8 and 8 → 16 carry 1.585 of the 2.326 nats of relative split (68 %); the top level
  32 → 65 carries 0.025 (1 %). The largest level is 35 times the smallest.
- **With Brown's criterion it is roughly even** over the first five doublings: 0.30 to 0.72 nats
  each, a factor 2.4, each close to or below ln 2 = 0.693. Only the last level is small.
- So boundedness is a property of the tower one chooses, and the least-defect tower is not the
  one that has it. A tower in which each class splits in two has relative split at most ln 2 per
  level by construction; the Brown tower is close to that without being forced.
- The relative defect is small against the relative split at these history lengths (at most 0.19
  nats per level, 0.42 in total against a fine floor of 1.91), and it grows towards the fine end
  of the trigram tower. It is not zero at any level above the first.

`tower.svg` plots class floor, defect and split against the number of classes for both defect
towers. In the trigram tower the defect peaks at 0.29 nats, at 10 classes.

## What this means for growing

- "Training then learns only what is new" fails by exactly the relative defect. A parent trained
  to its floor in the coarse world predicts the class from the coarse history; the child sees the
  fine history and can predict the same class better, by I(π_c Y ; π_f H | π_c H). The parent's
  part must be revised, not frozen. On Shakespeare with two characters of history this is up to
  0.19 nats per level.
- The specimen is the warning case. With the full history its defect is most of the class floor,
  because merging characters destroys what the history uses to locate itself.

## Verified, and how

- Specimen fine floor equals (2 ln 2 + 3 ln 3)/80 to 1e-15.
- Chain rule F − D + S = fine floor: 7e-18 over all 877 specimen partitions; below 3e-14 at all
  65 levels of each of the three Shakespeare towers.
- The fast evaluation used in the greedy search agrees with an independent dictionary
  implementation to 2e-13 at every level. This check caught a thread race in a first run (two
  loop variables shared between threads); the numbers above are from the corrected run.
- Level identity F_f = F_c − rel. defect + rel. split: below 4e-14.

## Not reached, and open

- Histories longer than two characters on Shakespeare. All counts are in-sample; with longer
  histories a held-out estimate is needed, since the plug-in defect is biased upwards.
- The towers are greedy. No search for the partition of least defect at fixed size, and no
  criterion that trades closure against information (for example defect per nat of class floor).
- The specimen with a fixed short history, where a closed partition might exist.
- No model was grown along either tower, so whether the even Brown tower is cheaper to train
  along than the concentrated one is untested.

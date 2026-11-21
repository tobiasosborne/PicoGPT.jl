# Audit of the round 1–2 loop prototypes

## 1. The positioned light cone

**Correct.** In `windowgpt`, the initial stream is the local embedding. Each iteration calls `advance` before `readout`. At a nonboundary query, `window_mix` combines the values at columns `n` and `n-1` from the previous stream. The residual, input injection, MLP, LayerNorm and readout all act within a token column. Induction gives the visible input interval `max(1,t−k):t`. The existing perturbation test actually changes the first-pass output at site 2 by flipping site 1 and checks equality outside the cone. [Implementation](../src/window.jl) · [Perturbation and pullback tests](../test/window.jl)

For independent fair bits, the prefix label is determined when `t ≤ k+1`. Otherwise one can pair every completion with its opposite label by flipping one invisible bit. Each such site contributes conditional entropy `ln 2`. Hence the pass floor is `max(T−k−1,0) ln(2)/T` and the deep floor is its mean over `k=1,…,K`. At `T=K=8` the successive counts of ambiguous sites are 6, 5, 4, 3, 2, 1, 0, 0. This gives **21 ln(2)/64 = 0.227438918621232 nats**, with a one-pass floor of 0.519860385419959 nats and a final-pass floor of zero. Counting completions of every prefix at every site for lengths 4–12 independently reproduced the deterministic and fair groups. [Audit calculation](../scratch/audit/check.jl)

The floor is the optimal risk of a predictor given position and visible bits. For deterministic labels, it is an infimum over finite real logits. Tied and untied local blocks obey the same information bound. Attainment by a particular width of stock block requires a separate construction or experiment.

## 2. Mixed deep supervision

**Correct.** `fit_round2.batch_at` draws one length uniformly from 4–12, one of its `T+2` horizons uniformly from `T−1:2T`, and 12 independent words. `crossentropy` divides each pass's loss by `12T`. `supervised_window` divides the sum over passes by `K`. Thus each sampled batch has equal weight. Equivalently, draw a batch, then a uniform word, position and supervised pass within it. Pooling all token-pass observations across batches would overrepresent long words and large horizons. `validation_prefix` weights chunk means by word count at fixed length, then takes the same nested horizon and length means. Averaging the two labels outside the cone preserves the expected hard-label loss there. [Batch sampler and validation](../scripts/window_round2.jl) · [Deep objective](../src/window.jl) · [Per-column loss](../../src/softmax.jl)

For every allowed horizon `K ≥ T−1`, summing the ambiguous-site counts gives `(T−2)(T−1)/2`. The coefficient of `ln 2` is therefore

```text
(1/9) Σ[T=4…12] (1/(T+2)) Σ[K=T−1…2T] (T−2)(T−1)/(2TK)
= 384975233676617 / 1653814216454400.
```

The mixed floor is **0.16135095172929345 nats**. I also rescored all five prefix gain pairs on every word and every relevant pass with ordinary binary logistic loss and the nested weights. Their losses agreed with the saved results to within `2e−13`, ranging from 0.161459555 to 0.161646040 nats. A 24-block untied gain stack can approach the same floor by using the same large positive neutral and large negative negation gain in each block. It has 48 parameters. [Recalculation](../scratch/audit/check.jl) · [Fitted gains](../artifacts/window-round2-register.json) · [Untied control](../artifacts/window-round2-controls.json)

## 3. The one-pass control

**Correct.** The control uses the same length distribution and a single pass. Two positions are determined and the remaining `T−2` labels are conditionally fair. The length-averaged entropy is `(1/9) Σ (T−2)/T × ln 2 = (89539/124740) ln 2 = 0.497544535835794 nats`. The expected token accuracy at fixed `T ≥ 2` is at most `[2+(T−2)/2]/T = 1/2+1/T`. This gives 62.5% at length 8, **53.125% at 32** and **51.5625% at 64**. All five register controls scored 52.770996% and 51.062012% on the two fixed 256-word samples, respectively. Individual samples can fall on either side of a population ceiling. [Control records](../artifacts/window-round2-controls.json) · [Evaluation words and tie handling](../scripts/window_parity.jl)

## 4. The NoPE boundary obstruction

**Correct with caveats.** Without a position signal or boundary marker, site 1's singleton returns its own value. Two copies of an identical state at an interior query have identical scores and their weighted average returns that same value. Extend the input indefinitely to the left with copies of its first bit. Inductively every copy has site 1's state at every pass. A real site's state is therefore a function of the last `k+1` bits of this padded input and the pass index. Local input injection preserves the equivalence. For example, prefixes `10` and `110` end in the same padded context at every depth and have opposite parities. This argument uses the real-arithmetic model of attention. At ordinary normal magnitudes the compact mixer implements the duplicate mean exactly. For pathological subnormals, even half of the smallest positive Float64 rounds to zero, so bitwise equality for arbitrary machine parameters would require additional assumptions. [Boundary and compact mixer](../src/window.jl)

At a full receptive field (`k ≥ T−1`), prefixes starting with zero have known parity after padding. For prefixes starting with one, fix the suffix after the initial run of ones, allowing an empty suffix. The unknown positive run length `r` determines the parity ambiguity. Each possible even `r` pairs with `r−1` for the same suffix and the shorter prefix has twice its probability mass under uniform query position and fair bits. Thus the even runs form the minority in each class. Summed over query positions, the all-ones prefix ending at `t=r` contributes `2^(−r)`. Each of `T−r` later positions contributes `2^(−r−1)` for exactly `r` leading ones followed by zero. Subtracting this minority mass from one gives

```text
1 − (1/T) Σ[r even, 2≤r≤T] 2^(−r) (1 + (T−r)/2).
```

The ceilings are **84.716796875%, 83.6805555555% and 83.5069444444%** at lengths 8, 32 and 64. Small-length counts match the expression exactly in rational arithmetic. It bounds models that share the readout across positions and have no independent boundary cue.

The reported deep bound **0.432345179880789 nats** equals the weighted conditional entropy after grouping by `(T,k,padded context)`. The model also lacks access to the total length. Grouping instead by `(k,padded context)` gives **0.449949952304051 nats**. To reproduce this, give a `t`-bit prefix mass `2^(−t)/T` within length and multiply by `Σ[K=max(k,T−1)…2T] 1/[9(T+2)K]`. Pool the two label masses first, then compute each group's entropy. In this cross-length calculation the key must contain all `k+1` bits, including repeated left padding even when `k+1>T`. Both numbers are lower bounds on model loss. The larger number is the Bayes risk for the length-hidden padded-context observation. [Independent enumeration](../scratch/audit/check.jl) · [Original per-length counts](../artifacts/window-round2-floor.json)

## 5. The two-gain prefix register

**Correct for the forward claims and the recorded probe. Indefinite nonzero Float64 covector transport is incorrect.** The register encodes bit 0 as −1 and bit 1 as +1. At pass zero site 1 has its input sign, later sites have state zero, and the virtual predecessor has sign −1 at every pass. A positive neutral gain preserves the predecessor's parity sign and a negative negation gain flips it. The real map `tanh` preserves sign and zero. Induction gives the right sign for `t ≤ k+1` and exactly zero log odds elsewhere. The exterior includes signed floating-point zeros, all decoding to probability 1/2. At known sites the probability has finite margin. [Forward recurrence and pullback](../src/window.jl)

For all five mixed-prefix checkpoints, `min |gain| = 7.557568045010668` across the fits. Starting from magnitude 1, the invariant lower bound 0.99 holds because `tanh(0.99 × 7.557568045010668) = 0.999999365759658`. This margin protects the forward induction in Float64 for arbitrary word length. Signs alone for arbitrary gains suffice over the reals but very small gains can underflow in Float64.

With a tied deterministic transition, site 1 is constant for `k ≥ 1`. If its predecessor is constant from pass `t−1`, site `t` is constant from pass `t`. Thus `s[T] = s[T+1] = …`, including in Float64 for the fitted parameters. The first pass with zero **increment** is guaranteed by `T+1`. It can come earlier. At length 1, for example, the first step can change magnitude from 1 to `tanh(|gain|)`. On the published length-64 probe the front reaches site 64 on pass 63 and the increment is exactly zero on passes 64–128. [Recorded states and front](../artifacts/window-round2-probe.json)

The saved covector is with respect to the **pre-tanh** scalar `u`. From site `t` at pass `k`, the sole possible predecessor at the previous pass is site `t−1`. Its multiplier is `gain[x_t] (1−s[k−1,t−1]^2)`. The path ends at the fixed left boundary. With finite nonzero gains in real arithmetic every factor is nonzero. The implemented Float64 pullback can vanish when `tanh` rounds to ±1 or when the accumulated product underflows. The seed-1 16-site replay had nonzero magnitudes 1, `1.0451346191e−6`, `4.3328450555e−13` and finally `1.8019930411e−93` at pass/site 1. Pass 0 was all zero. With the **same gains**, 64 neutral bits and the same terminal logit contrast at pass/site 64, the entire covector first became zero at pass 13, before reaching the boundary. Book wording can say that the architecture permits exactly one path and that the recorded 16-site path survives. [Float64 counterexample and replay](../scratch/audit/check.jl)

## 6. Rule 60 and the Lucas check

**Correct targets and test matches. The finite tests alone establish agreement only on the evaluated words.** Write `S` for the one-site right shift with zero entering the left boundary. Over bits, one step is `I+S`. In characteristic two, `(I+S)^(2^i) = I+S^(2^i)`. Multiplying those factors for the set bits of `k` contributes `S^j` exactly when `(j & k) == j`. This is the mod-2 binomial case of Lucas' theorem. The output at site `t` includes `x[t−j]` for these lags less than `t`. Both scripts shift in the correct direction. I checked the bit test against integer binomial coefficients through 96, then replayed all five saved gain pairs against direct iteration and the Lucas targets at every pass on the original 256 length-64 words and the whole length-97 impulse. Every decoded bit agreed. The zero information floor follows from the deterministic accessible target. [Window script](../scripts/window_round2.jl) · [Prototype script](../scripts/common.jl) · [Replay](../scratch/audit/check.jl)

A nonlinear model may deviate on unseen inputs even if it reproduces the impulse response. The saved gain pairs allow a separate proof on **all words and all passes**. Let the positive and negative gains be `a` and `−b`, with `a,b>0`. For current state `q` and predecessor `p`, the new logit is `p × [(1−q)a/2 − (1+q)b/2]`. Assume both input magnitudes are in `[0.99,1]`. The weight on the gain selected by the current sign is at least 0.995 and the opposing weight is at most 0.005. The multiplier has the required copy/flip sign whenever this lower bound is positive, and

```text
|u| ≥ 0.99 × [0.995 min(a,b) − 0.005 max(a,b)].
```

The smallest bound among the five checkpoints is **7.217529638519129**. Its tanh is **0.999998923626066**, restoring the invariant with ample Float64 margin. The initial states and the virtual predecessor all have magnitude 1. Induction proves the XOR sign, even though the analog feedback magnitudes vary. [Five fitted Rule 60 gain pairs](../artifacts/window-round2-rule60.json)

## 7. Other book-quotable exact claims

**Correct with the following scope and precision qualifications.**

- **Clipped ReLU certificates.** In the prototype register all initial bits and the boundary are binary, and the feedback is `clamp(u,−1,1)`. Recomputing the four XOR margins from each saved weight vector gives minima of 1.6013791436 (prefix) and 1.5911123134 (Rule 60) across the 17-parameter seeds. Every correct output therefore clips to the exact binary value. Induction works for arbitrary horizons and lengths. For prefix parity the intermediate result is the trailing `min(k+1,t)`-bit parity, hence the full prefix is known for `t ≤ k+1`. The corresponding 33-parameter window/deep runs have minima of 3.0985066618 and 1.9019063369. In the 13-parameter prefix runs the smallest passing margin is 1.8277725371. Quote “approximately 1.83” or “above 1.82” if using a strict bound. These clipped prototypes initialize every site to its bit. The window gain register initializes only site 1 for prefix parity. [Clipped register](../src/register.jl) · [Independent gate checks](../scratch/audit/check_artifacts.py)
- **13-parameter Rule 60 seed 2.** Its four-pair minimum 0.6063011421 fails the unit-margin test, as reported. The saved weights nevertheless admit an exact decoded-rule proof. Write `A = −0.6063011420952008` (the output bias), `B = −1` and `P = +1`. Both A and B encode bit 0. The clipped cell returns P for every mixed binary-sign pair and B for `(P,P)`. For equal zeros it returns A from `(A,A)` and `(B,B)`. It also returns A for current A with the virtual left B. These A results have three inactive ReLUs, so the value is exactly the bias. Interior adjacent zeros must have the same register value. Initially both are B. Thereafter two adjacent zero outputs came from overlapping equal-bit pairs, hence from three identical preceding bits. Two 00 pairs both write A and two 11 pairs both write B. This preserves the invariant, including at the first real pair, and excludes the problematic case of current B with left A. Its actual logit is +0.3400. Thus finite rollouts are corroboration for this seed too. [Weights](../artifacts/round2-rule60-three-tied.json) · [Transition checks](../scratch/audit/check_artifacts.py)
- **Parameter and seed counts.** A scalar two-input ReLU cell has `2F + F + F + 1 = 4F+1` trainables. The stock formula is `2VD+V+2D + L(4D²+2DF+9D+F)`. It gives 50 shared parameters and 532 per block for `V=2,D=8,F=12`, hence 582 tied and 4,306 eight-block untied. For `D=4,F=16,L=1` it gives 270. Recounting the saved runs gives one 1%-floor hit among 15 round-1 width-screen runs. The tuned seed's deep loss was 0.227447351214711. The 270- and 582-parameter normalized stock recipes each met both criteria in one of five seeds. Their 100% long scores describe the fixed test sample. “Smallest” refers to the tested configurations. [Parameter formula](../../src/parameters.jl) · [Screen checks](../scratch/audit/check_artifacts.py)
- **Stationarity and gradient diagnostics.** The predecessor-only prefix fixed-point argument also applies to the unclipped tied MLP with finite states. At length 48, seed 4's last update into pass 48 has RMS 2.8760. Its first necessarily unchanged update is into pass 49. A tiny measured stock increment is a numerical diagnostic, and a stock attention gradient has possible support throughout the backward cone. A nonzero gradient on its left edge is checkpoint-specific. Finite-difference errors also depend on the run. On Julia 1.13.3 with `-O1` the five round-2 relative-loop checks and four register checks had maximum relative loss-gradient error `7.76082e−10`. The separate adjacent-window check gave `1.46571e−9`. [Unclipped traces](../artifacts/round2-prefix-unbounded.json) · [Test definitions](../test/runtests.jl)

## Reproduction

Run these checks from the repository root with Python 3 available to read the saved JSON.

```sh
julia --project -O1 loop/scratch/audit/check.jl
python3 loop/scratch/audit/check_artifacts.py
julia --project -O1 loop/test/window.jl
julia --project -O1 loop/test/runtests.jl
```

Both audit checks completed and the existing suites passed 143 and 78 assertions, respectively, on Julia 1.13.3. The scope is rounds 1–2. Checkpoint evaluation was independent of training. Timing figures were outside this verification.

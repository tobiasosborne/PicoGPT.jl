# Independent review of written versus looped negation

The comparison, 270-parameter construction, gate pullbacks, lookup bounds and Shakespeare accounting hold under the assumptions in the note. Make three wording changes before quoting it.

1. Say that one blank pass has **minimum population end-only CE** $\ln 2$ for fair words at $T>2$. An uncalibrated predictor incurs greater loss.
2. In the Ising analogy, the intended update is the preferred sign only for $\epsilon<1/2$. The flip-probability formula also holds above one half.
3. Bound the normalized coefficients **in the witness basis** by one. Individual LayerNorm feature values can exceed one.

The independent window forward reproduced all four reported margin extrema on every word through length eight. At $T=8$, seven calls use 14 columns and six scratch tokens in the written protocol, or 56 columns and zero scratch tokens in the dense loop. The closed scalar gate had a nonzero derivative on a three-word example. A model with a uniform terminal forecast at length four had zero population gradient despite nonzero individual derivatives. The proposed Shakespeare sandwich has **810,177 parameters and eight block applications per token**. Its tiny closed-gate implementation matched Book 1, and its open-gate pullback passed finite differences. Training reliability with the proposed gate and the new sandwich’s held-out loss remain unmeasured. [Full verdicts and reproduction](../notes/check-wvl.md) · [Independent Julia checks](../scratch/check-wvl/check.jl)

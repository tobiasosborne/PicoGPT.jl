# Brief: audit of the exact claims of the looped-transformer prototypes

You are the reviewer. Other agents (Sol, a coder) ran experiments for a possible third PicoGPT book,
on the tiniest looped transformer. A book will rest on a handful of exact numbers and proofs; each
must be checked by someone other than its author before the user sees it. Check them; do not trust
the authors' prose. Read `BRIEF.md` §1–3 for notation, then `loop/notes/window.md`,
`loop/briefs/window-2-report.md`, `loop/src/window.jl` and the relevant scripts in `loop/scripts/`
(`window_*.jl`), and `loop/notes/prototypes.md` (round 2 section).

## Claims to audit (each: correct / incorrect (with the correction) / correct with caveats)

1. **The light-cone floor.** Prefix parity on uniform bits, window-2 attention, input injected,
   readout after each pass k = 1…K, deep supervision averaged over passes: the Bayes CE floor
   ℓ* = (ln 2/(T K)) Σ_k max(T − k − 1, 0); at T = K = 8 it is 21 ln 2/64 = 0.227438919 nats. In
   particular check the offset (pass 1 already sees x_{t−1}) against the actual code path.
2. **The mixed floor**: T uniform on 4…12, K uniform on T−1…2T, claimed exact value
   (384975233676617/1653814216454400) ln 2 = 0.161350951729293 nats. Check the weighting convention
   (per batch, per word, per token) matches the training loss actually computed.
3. **The one-pass control floor** (89539/124740) ln 2 = 0.497544536 nats on the same mixed
   distribution, and the information ceilings 53.125 % / 51.563 % at T = 32 / 64.
4. **The NoPE boundary obstruction**: with no positional signal and singleton attention at
   position 1, a ceiling of 84.72 / 83.68 / 83.51 % at T = 8 / 32 / 64 and a deep-loss lower bound
   0.432345180 nats. Check the argument (padding by the first bit) and the numbers.
5. **The two-gain register** (`loop/notes/window.md`, "The two-gain register"): the induction
   proof that learned gains with the right signs give the exact sign inside the cone, exactly zero
   log odds outside it, and a state that is exactly stationary from pass T on; that the terminal
   covector travels exactly one site per pass. Check whether "exactly" is literally true in
   Float64 (tanh of a large argument rounds to 1, products of tiny numbers, etc.).
6. **Rule 60**: Lucas' theorem as used in the scripts, and the claim that matching the impulse row
   plus 256 random words is evidence of the full rule (a nonlinear model need not superpose).
7. Anything else in these notes that is stated as exact and that a book would quote.

## Rules

You may write checking code in `loop/scratch/audit/` (use `julia --project`; `using PicoGPT`).
Edit nothing else. Do not commit. Do not start Muse or Gemini agents. Deliverable:
`loop/notes/audit.md` (one section per claim, verdict first, then the check), and your summary
copied to `loop/briefs/audit-report.md`.

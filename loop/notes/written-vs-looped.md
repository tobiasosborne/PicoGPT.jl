# Negation with and without a written answer

**Proved.** For a sentence of $T\ge2$ independent fair words, score only its final parity. A single pass of the two-word block has cross-entropy $\ln 2$ for $T>2$. To reach a zero infimum, the same block can either write $T-2$ intermediate one-bit answers or make $T-2$ additional passes over the sentence. Both schedules have $T-1$ dependent calls. Writing is cheaper in evaluated token columns for $T>2$. Looping keeps the transcript at its original length.

Here **proved** means a deduction for the stated architecture, distribution or computational model. **Checked** identifies either a finite calculation or a result reported in a named source. **Conjectured** marks a proposed experiment or extrapolation. All losses below use natural logarithms.

## 1. Where the intermediate answer lives

**Proved.** Write $x_t=0$ for `be`, $x_t=1$ for `not`, and $y_t=x_1\oplus\cdots\oplus x_t$. An answer uses the same two token IDs, with 0 meaning “affirms” and 1 meaning “denies.” The caller supplies the words in sentence order. It asks for $y_T$ after the last word. The block reads the answer from the logits at the last input word, with no extra query token. The final answer costs one emitted token in all three rows. Training on prefix labels at every position supplies supervision. A label becomes a scratchpad only when its predicted token actually enters a later input.

| Schedule | Observation at the final decision | Optimal end-only CE | Dependent block calls | Token columns evaluated by ordinary forwards | Intermediate tokens emitted |
|---|---|---:|---:|---:|---:|
| One pass, blank scratchpad | At most the last two words and boundary information | $[T>2]\ln 2$ | 1 | $T$ | 0 |
| One pass per decision, written feedback | Previous written parity and next word | 0 | $T-1$ | $2(T-1)$ | $T-2$ |
| $K\ge1$ tied passes, blank scratchpad | At most the last $K+1$ words and boundary information | $[T>K+1]\ln 2$ | $K$ | $KT$ | 0 |

**Proved.** The written protocol starts with the two-word context $(x_1,x_2)$ and emits $y_2$. For each $j=3,\ldots,T$ it passes $(\hat y_{j-1},x_j)$ through the block once and emits $\hat y_j$. Each pair is a fresh length-two context, including the specimen's ordinary first-column initialization. For `not not be not` the calls are `not not → be`, `be be → be`, `be not → not`. Only the first two outputs are scratch. In teacher forcing the first argument is gold. In free decoding it is the model's own argmax. The four possible two-word inputs suffice to prove correct free decoding by induction. This streaming protocol requires the caller to supply the next word between answers. Appending a scratchpad *after* an already supplied prompt with a strict two-token mask would give the decoder different information.

**Proved.** With window two, each block can move information at most one position to the right. The fresh embedding, stream LayerNorm and readout operate within a column. The boundary start puts $x_1$ into the initial stream, so it can reach column $K+1$ on pass $K$. If $T>K+1$, pair each sentence with the one obtained by flipping $x_1$. The terminal observations coincide and the labels differ. The optimal forecast is exactly $(1/2,1/2)$, attained by zero logits. Once the first word arrives, the conditional label is deterministic. Zero CE denotes an infimum under increasing correct logit margins. For $T=1$ a single call and a single output suffice for all three protocols. For $T=2$ the three minimum-depth schedules coincide.

**Proved.** For a terminal query, the first call folds two input bits, and each further call consumes one more input bit along the dependency path. One intermediate answer therefore trades for one extra whole-sentence pass *in this protocol*. At $T=8$ both exact schedules have seven dependent calls. Writing evaluates 14 token columns and emits six scratch tokens. The dense loop evaluates 56 token columns and emits none. Counts exclude embedding, decoding, padding and I/O. They describe straightforward execution. A specially scheduled register can compute only the path to a single requested answer in linear work. A transformer processing a full sequence has parallel columns, and token generation introduces synchronization. These counts alone do not predict latency. When every prefix is wanted in the output anyway, writing its answer incurs no additional intermediate-output charge.

**Proved.** Scoring all prefix positions instead restores the original information floor $\max(T-k-1,0)\ln(2)/T$ at pass $k$. Averaging that floor over positions, lengths and supervised passes produced the two numbers 0.497545 and 0.161351 in the concept. They measure different objectives from the terminal rows above. [Window analysis](window.md) · [Audit §§1–3](audit.md)

### Weights for the 270-parameter block

**Proved.** The three rows are achievable with the specimen's $D=4,H=1,F=16,V=2$ block, first-word start, anchored first residual, fixed stream LayerNorm and predecessor bias $+4$. The following construction also gives a single family approaching the all-prefix deep floor. The executable arrays are in the check script. The two-gain register in the earlier notes gives an independent shorter digital solution. [Implementation of the witness](../scratch/wvl/check.jl) · [Two-gain register](window.md#the-two-gain-register)

Use the three orthogonal, centered feature vectors
$$
a=(1,1,-1,-1)^\mathsf T,\quad b=(1,-1,-1,1)^\mathsf T,\quad c=(1,-1,1,-1)^\mathsf T.
$$
Each has squared norm four, so the coordinate of $X$ along $a$ is $a^\mathsf TX/4$, and similarly for $b,c$. Use the sign $s=(-1)^{x_t}$, embedding $sa$, and the stock marker $c$ at the first word. Set every LN gain to one and offset to zero. Set $Q=K=0$ and let the value/output projections read the $a$ coordinate of the normalized incoming stream and write it into $b$. The positional bias gives predecessor weight $p=1/(1+e^{-4})$ at interior sites. Write $r$ for the resulting scalar mixture. After the anchored attention residual the column is
$$
Y=sa+rb+\delta c,\qquad \delta=[t=1],\qquad \sigma=\sqrt{1+r^2+\delta+10^{-5}}.
$$

Define the clipped ramp $C_{l,h}(v)=([v-l]_+-[v-h]_+)/(h-l)$. One ramp takes two ReLUs. The MLP writes $M(c+d a)$ with $M\ge100$, where
$$
d=\sum_{u,w\in\{-1,1\}}uw\, C_{1.12,1.25}\!\left((us+wr-4\delta)/\sigma\right)
 +\sum_{u\in\{-1,1\}}u\, C_{1.6,1.8}\!\left((us+3\delta)/\sigma\right).
$$
These arguments are linear forms on $\mathrm{LN}_2(Y)$. The constant $Mc$ is the output bias. This uses twelve of sixteen hidden units. The stream is $\mathrm{LN}(Y+M(c+da))$. The logit difference $Z_{\rm be}-Z_{\rm not}$ reads $\sqrt M$ times the $a$ coordinate after the final LN.

**Proved by induction.** A known predecessor carries its parity sign in the $a$ coordinate after the next input normalization, with magnitude in $[0.68,1]$. An unresolved predecessor has magnitude at most $1/M$. Initially the first word has magnitude $1/\sqrt{2+10^{-5}}$ and the other columns are zero. All normalized coordinates have absolute value at most one. Thus a known predecessor yields $|r|\ge0.68p-(1-p)>0.64$ with its sign preserved. An unresolved predecessor gives $|r|\le p/M+1-p<0.03$. At an interior site with known predecessor, $(1+|r|)/\sigma>1.25$ and $(1-|r|)/\sigma<1.12$. Exactly one of the four interior ramps fires fully, with sign $s\,\mathrm{sign}(r)$. At an unresolved site all four arguments are below 1.12. At the boundary the $-4\delta$ term suppresses these ramps. There the matching boundary argument is at least $4/\sqrt{3+10^{-5}}>1.8$, and the other is at most $2/\sqrt{2+10^{-5}}<1.6$. Interior boundary ramps stay off. Hence $d$ is exactly the known prefix sign or zero.

To close the induction, before the stream LN the three coordinates are $(s+Md,r,M+\delta)$. Put $q=(s+Md)^2+r^2+(M+\delta)^2$. Two successive unit LNs give the first coordinate $(s+Md)/\sqrt{(1+\varepsilon)q+\varepsilon^2}$, $\varepsilon=10^{-5}$. For $|d|=1$ its sign is $d$ and its magnitude exceeds 0.68 even at $M=100$ (use $|s+Md|\ge M-1$ and $q\le2(M+1)^2+1$). For $d=0$ its magnitude is at most $1/M$. This applies to both LN₁ on the next pass and the final LN before readout. Correct known margins exceed $0.68\sqrt M$ while unresolved logit gaps vanish uniformly as $1/\sqrt M$. The cone therefore has an attainable CE infimum with a fixed width and a shared set of weights at each $M$. No training assumption enters the construction.

**Checked.** The stock window forward ran on every word of each length 1–8, every site through pass $T+1$, and every free-written decision on those words. At $M=100$ the smallest signed margin inside the cone was 7.03532 and the largest absolute gap outside was 0.099995. At $M=10{,}000$ they were 70.70679 and 0.00999995. Both scales decoded every written answer correctly. These are hand-set arrays. The measured fresh-seed rate for *trained* specimen weights is a separate result. [Check script](../scratch/wvl/check.jl) · [Fresh-seed study](window.md#fresh-seeds)

## 2. A wrong intermediate answer

**Proved.** Number the noisy transitions from an exact seed. At step $i$ let $e_i$ be an error relative to the *correct update of the state actually received*. Then $\hat y_i\oplus y_i=(\hat y_{i-1}\oplus y_{i-1})\oplus e_i$. Two flips cancel. For independent Bernoulli errors with rates $\epsilon_i$,
$$
\Pr(\hat y_t\ne y_t)=\tfrac12\left(1-\prod_{i=1}^t(1-2\epsilon_i)\right).
$$
Indeed the sign of the final error is the product of the $(-1)^{e_i}$, whose expectations multiply. Equal rates give $(1-(1-2\epsilon)^t)/2$. The written protocol has $t=T-1$ noisy decisions including the final answer if $x_1$ is exact. A convention that emits a possibly wrong $y_1$ too has $t=T$. A teacher-forced error rate by itself supplies neither independence nor the error law on mispredicted prefixes.

**Proved under a noise model.** In the ideal predecessor-only loop, the terminal bit follows one backward diagonal. Independent flips of the decoded transition on that diagonal give precisely the same formula, with $t$ the number of noisy edges traversed from the prescribed boundary or seed. In particular the first arriving answer follows $T-1$ updates from $x_1$. Later answers can use a newly computed boundary sign. Strip off the intended copy/flip sign on each edge. The resulting binary symmetric channel has transition probabilities $1-\epsilon_i,\epsilon_i$ and sign eigenvalue $1-2\epsilon_i$. In the zero-field open Ising chain with a fixed starting spin the physical edge correlation is $\tanh(\beta J_i)$; it is $(-1)^{x_i}(1-2\epsilon_i)$ if its preferred sign is the input-controlled copy or flip. Endpoint correlations multiply. The trained gain cell instead evaluates $\tanh(g(x)s)$ deterministically on a real state. Its sign is exact under the gain conditions in the audit. A learned soft-attention block can also combine several paths and share errors across passes. Applying independent-channel rates to either model requires measurements. [Register audit §5](audit.md)

**Checked.** Exact rational enumeration of every flip mask through eight transitions reproduced the expression at rates 0, 0.1, 0.5, 0.8 and 1, including the zero-step boundary. [Check script](../scratch/wvl/check.jl)

## 3. How far a lookup goes in one pass

**Proved.** In width-$w$ causal attention, one layer reads at most $w-1$ columns to the left. A stack of $L$ such attention layers per pass, iterated $K$ times, reads back at most $KL(w-1)$ columns beyond its starting input column, assuming pointwise embeddings, residual injections and readout. Induct by composing the intervals. Reaching the edge requires a sensitivity or construction check. For negation, the line is a chain of maps on a single bit, $f_t(y)=y\oplus x_t$. The window puts each next map one edge away. It converts the number of dependent transfers into a visible distance.

**Proved with a specified access model.** If a query carries only its current address and may read one arbitrary-table entry at that address per adaptive round, $h$ serial reads in a pass can advance $h$ hops. Following $n$ independent arbitrary tables needs $\lceil n/h\rceil$ passes in the worst case under this interface: change the entry on any still-unqueried edge to change the answer, choosing subsequent tables to preserve the difference. This assumes a domain of size at least two, fresh tables, and no preprocessing, speculative table-wide reads or composed entries. Heads whose queries are all fixed before the reads cannot use one another's returned address *within* that attention operation. Full attention can fetch a distant entry in its first layer, and soft mixtures, wide summaries and many table positions working in parallel exceed the serial-oracle interface. There is no universal bound of one hop per transformer pass. Finite-precision one-layer composition lower bounds have explicit size assumptions. [Derivations §4](derivations.md#4-following-a-permutation)

**Proved for pairwise composition.** If every position may read one *old summary* in parallel and compose it with its own, initial single-edge summaries can represent at most $2^r$ edges after $r$ rounds of binary concatenation. Pointer doubling meets the bound. Store an address $a_t=t-1$ and bit $s_t=x_t$, with a neutral sentinel, and simultaneously update $(s_t,a_t)\leftarrow(s_t\oplus s_{a_t},a_{a_t})$. After $r$ rounds the span is $\min(t,2^r)$; the address is $\max(0,t-2^r)$. Ordered function composition replaces XOR for general summaries when representable. The remote read needs global routing. The earlier stock block uses one additional preparation pass and grows its tables with the maximum supported length. [Construction and finite check](derivations.md#shared-weight-pointer-doubling)

**Checked against the release.** The fixed shipped model executes 12 core layers per pass. The separately trained variable-loop research model improved on N-Hop Lookup through ten passes, while closed-book recall changed little on the reported tasks. The benchmark measures accuracy by chain length. It establishes neither a literal hop count per pass nor a general law about stored knowledge. The source also reports shorter written reasoning with extra passes in a selected subset of questions. The toy's exact token exchange is specific to its protocol. [Thinking Machines, “Inkling 2 Mini”](https://thinkingmachines.ai/news/inkling-2-mini/)

## 4. What a closed gate stabilizes

**Checked against the architecture sources.** Each of Inkling's later core passes can read the embeddings and prelude again. Each Q, K, V and MoE input has its own message, a learned mixture of the previous pass's two normalized block sums at the same token. Goodfire's appendix gives $z=P[\mathrm{RMSNorm}(x);\mathrm{RMSNorm}(m)]$ with 64 bottleneck coordinates and
$$
x'=\kappa\odot x+\gamma\odot m,\qquad
\kappa=1-\tanh(Fz),\qquad \gamma=\tanh(Gz).
$$
The announcement describes an initially blocked message. Zero output matrices $F,G$ would realize the identity mix; that is a sufficient initialization, not a recovered training checkpoint. The gate below retains only the essential restart and adjustable message path. [Goodfire appendix](https://www.goodfire.com/research/interpreting-inkling-2-mini/) · [Release](https://thinkingmachines.ai/news/inkling-2-mini/) · [Source note](inkling.md)

**Proposed design, proved identities.** Write $U$ for the marked immutable embeddings, $S=S(U)$ for the first-column-only initial stream, and $A_p(Y,U)$ for the existing anchored window block with optional final stream normalization. Fix this normalization choice across comparisons. Add one learned scalar $g$, initialized to zero, shared across positions and passes. Take $X_0=S$ and
$$
m_k=X_{k-1}-S,\qquad Y_k=S+g m_k,\qquad X_k=A_p(Y_k,U).
$$
The subtraction makes $g=1$ exactly the original recurrence. At $g=0$ every call receives $S$ and returns $A_p(S,U)$ for every input, horizon and parameter vector. Thus the final logits, loss and backbone gradient equal those of a single call with the same stream normalization. The same identity holds for a mean of identical pass losses with identical targets. $K=1$ is independent of $g$. Merely gating a correction on an autonomously carried residual would leave the old residual path open and would fail this proof.

**Proved pullback.** Seed the last readout, or add each supervised pass's readout cotangent at its own $X_k$. Apply the saved block VJP to $\bar X_k$, obtaining $V_k=\bar Y_k$, an anchor cotangent $U_k^A$ and a block-parameter cotangent $p_k^A$. Reverse $k=K,\ldots,1$ with
$$
\bar g\mathrel{+}=\langle V_k,X_{k-1}-S\rangle_F,\quad
\bar X_{k-1}\mathrel{+}=g V_k,\quad
\bar S\mathrel{+}=(1-g)V_k,\quad
\bar U\mathrel{+}=U_k^A,\quad \bar p\mathrel{+}=p_k^A .
$$
Add $\bar X_0$ to $\bar S$ and send $S_U^\mathsf T\bar S$ to $U$ before the embedding scatter. A channel gain $g_i$ replaces the Frobenius product with the sum over token columns for row $i$. Parameterizing a gain as $\tanh b$ multiplies its cotangent by $1-\tanh^2 b$; a direct scalar allows $g=1$ exactly.

**Proved.** With final loss and $K\ge2$, at closure $\partial_g\mathcal L=\langle A_Y^\mathsf T\bar X_K,A_p(S,U)-S\rangle_F$. All earlier recurrent adjoints vanish. This inner product can be nonzero when both vectors have aligned components. Orthogonality, zero messages or cancellation across examples can give zero. In particular, with a calibrated uniform terminal readout at closure and $T>3$, the *population* gate gradient of end-only CE vanishes on fair words. The first-order logit perturbation can traverse at most two local calls, hence depends on at most the last three words (and fixed boundary information). Conditioned on them, the gold terminal parity remains fair. Short lengths or prefix supervision may therefore matter for learning. For the Goodfire formula, if $F=G=0$ then $\bar F=-(\bar x'\odot x)z^\mathsf T$ and $\bar G=(\bar x'\odot m)z^\mathsf T$ per column, while this route gives $\bar z=0$. Assuming $P$ is zero as well would remove this opening signal. [Gate formula](https://www.goodfire.com/research/interpreting-inkling-2-mini/)

**Checked.** On one seeded three-word example with prefix CE at the final pass, three closed passes gave the same loss and backbone gradient as one. The gain derivative was 1.81722078 at closure and 0.00234017 at $g=0.2$. Centered differences of the scalar gain and one direction through the whole backbone gave maximum absolute discrepancies $1.29\times10^{-7}$ and $3.81\times10^{-11}$, respectively. This is an example of an opening signal, with a different objective from the long end-only population above. [Check script](../scratch/wvl/check.jl)

**Proved scope of stability.** Holding $U$ fixed gives the state Jacobian $J=g A_Y$ (channelwise, $A_Y\operatorname{diag}(g)$ at each column). It vanishes at closure. If $A$ is $L$-Lipschitz on the relevant inputs, $|g|L<1$ gives a contraction on the state domain. For a fixed-point convergence conclusion on a restricted domain also require an invariant complete domain. Smallness at one point gives only local control. Stream normalization bounds each output column by $\sqrt D$ in Euclidean norm and can still have a large derivative. Even without that final normalization, the toy's *anchored* block at fixed $U$ has bounded branch outputs because the pre-LNs bound their inputs. Removing the final normalization changes the model and needs its own control.

For the ordinary pre-LN residual $B$, translations $v=\mathbf1 c^\mathsf T$ pass unchanged through its derivative. In a restarted scalar loop $B(U+gX)$ their eigenvalue is exactly $g$. In the toy's anchored $A$, LN₁ removes the translation before attention, and the residual is anchored to $U$, so $A_Yv=0$ even with the output normalization removed. A general channel gain can take a constant-feature vector out of that subspace. Input-dependent keep and gain also contribute terms involving their own derivatives. Small observed gain in Inkling is insufficient to infer a contractive Jacobian. **Conjectured:** the learned scalar could replace final stream normalization without sacrificing training reliability. Only exact closure and the local gain bound currently justify calling it a stabilizer. [Stock symmetry, derivations §5](derivations.md#5-linearization-and-the-light-cone)

## 5. Where the fixed point belongs

**Checked against the described methods.** Teach finite unrolling and shared-parameter VJPs in the main account of the three-pass sandwich. The shipped model has fixed depth and a final decode. Its reported training-stability argument is the initially closed cross-loop channel. The toy's many-pass extrapolation and passwise loss remain experiments in recurrent refinement. Huginn varies the unroll and trains with truncated backpropagation; HRM/TRM use outer deep supervision, and TRM removes HRM's one-step equilibrium approximation. An actual equilibrium solve with an implicit adjoint belongs to the DEQ case. The book can group these questions in a short aside while keeping the methods distinct. [Research notes and primary references](literature.md) · [Release source note](inkling.md)

**Proved.** For ordinary finite depth, backpropagate through each realized pass. If instead a differentiable fixed point solves $X^*=G(X^*,a)$ and $I-J_*$ is invertible, the implicit adjoint solves $(I-J_*^\mathsf T)\lambda=\nabla_X\ell$. A convergent Neumann expansion additionally needs $\rho(J_*)<1$. Stable decoded signs or a small observed state step establish neither condition. Loss at every pass is an objective choice and alone proves no convergence. [Derivation, hypotheses and counterexamples](derivations.md#1-a-tied-block-and-its-pullback)

## 6. Four Shakespeare blocks, eight applications

**Proposed design, proved accounting.** Keep Book 1's dense pre-LN blocks, $D=128,H=4,F=512,T=256,V=65$, full causal sequence attention and fixed sinusoidal positions. Use a single trainable gain *per channel*, shared by both restarts, and use the previous core output as the message. With $B_i$ distinct stored blocks, set
$$
U=B_1(E[:,x]+P),\quad M_0=0,\qquad
Y_k=U+g\odot M_{k-1},\quad M_k=B_3(B_2(Y_k)),\quad k=1,2,3.
$$
Compute $Z=W_{\rm out}\mathrm{LN}_f(B_4(M_3))+b_{\rm out}$ once and score Book 1's next characters. With $g=0$ this is exactly the four-block checkpoint's predictor for arbitrary existing weights. The design borrows Inkling's schedule and gated restart. The Book 1 block continues to use its ordinary residuals, and the sole message enters the first core block. Production Inkling additionally mixes both previous block sums at multiple inputs in every core layer. [Book 1 semantics](../../BRIEF.md#3-the-model-exact-semantics-the-julia-js-and-prose-all-implement-this) · [Architecture source note](inkling.md)

| Stored component | Count |
|---|---:|
| Embedding, final LN, output matrix and bias, $2VD+2D+V$ | 16,961 |
| One block, $4D^2+2DF+9D+F$ | 198,272 |
| Book 1's four blocks plus embedding/readout | 810,049 |
| Shared 128-channel gate | 128 |
| Total | **810,177** |

**Proved; parameter total checked with repository code.** Depth is $1+2\times3+1=8$ block applications per token, versus four for Book 1. This is 2,048 token-block applications per length-256 window, exclusive of embedding, gates and readout. Equal-depth eight *untied* blocks would store 1,603,137 parameters. At inference each causal attention application uses its own past K/V activations, including distinct pass-local caches for each repeated core block. Weight sharing does not imply that those activations agree once the gate opens. [Parameter layout](../../src/parameters.jl) · [Check script](../scratch/wvl/check.jl)

**Proved pullback.** Retain or recompute the prelude and coda tapes, the six individual core block tapes and the preceding messages. Pull the single loss through the readout and $B_4$, then for $k=3,2,1$ pull $\bar M_k$ through $B_3,B_2$ to obtain $V_k=\bar Y_k$. Sum each block's parameter gradients across its three calls. Accumulate $\bar U\mathrel{+}=V_k$, $\bar g_i\mathrel{+}=\sum_n V_{k,in}M_{k-1,in}$, and $\bar M_{k-1}\mathrel{+}=g\odot V_k$. Discard the cotangent of the fixed zero $M_0$. Call the prelude VJP once with the summed $\bar U$ and scatter to the embedding. Store a single parameter vector and one optimizer state per shared parameter. No equilibrium inverse enters this computation.

**Checked historical context.** Book 1 reached 1.5313 held-out nats/character. The existing four-pass stock single-block model reached 1.654 and the small windowed recurrent model reached 1.626 at six passes, each in its own recorded run. Neither estimates the new sandwich's loss. It stores four blocks and uses eight applications; both parameter count and execution depth differ from the 215,233-parameter trials. [Prior Shakespeare runs](shakespeare.md)

## 7. Tests to schedule on mains power

**Conjectured timings.** These budgets use earlier update-loop measurements of roughly ten seconds per toy seed and 75 minutes wall time for a four-application Shakespeare run. Compilation, validation and hardware power profile affect elapsed time. [Toy timings](window.md#fresh-seeds) · [Shakespeare timings](shakespeare.md#training-curves-compute-and-stability)

1. **Written and latent parity** (allow 5–15 minutes). Fit the three end-only schedules on matched words and initial seeds. Give the written model gold pairs during training and its own argmax during evaluation. Report end-only CE, final accuracy, intermediate-token count and evaluated columns separately. Compare free rollouts with teacher forcing through lengths 32 and 64; inject independent bit flips as a positive control for the error law. Retain the untrained witness as an exact reference.
2. **Does the gate replace stream normalization?** (allow 20–60 minutes with exhaustive validation). Cross gate enabled/bypassed with final stream LN present/absent, using identical anchored blocks, seeds, pass schedule and objective. Bypassing here means fixed $g=1$. Start with five seeds per cell; budget 30 fresh seeds per cell for a reliability claim. Log gain, closed-gate gradient on short and long words, reached front, centered state steps and estimates of the Jacobian norm. Compare an end-only run against prefix supervision if the initial gate stalls.
3. **Adaptive hops and doubling** (allow 10–30 minutes for a small synthetic sweep). Probe an immutable-table query and a parallel writable-summary scan separately. Sweep hop count and passes, permute addresses, and inspect decoded intermediate addresses. A chain-length accuracy curve alone cannot diagnose doubling.
4. **Shakespeare sandwich** (allow 2–3 hours per 5,000-update eight-application run, plus controls and hardware-dependent overhead). First check exact closed-gate logits against the four-block weights and finite-difference the core, gate and prelude on a tiny configuration. Pair a scratch-trained sandwich with Book 1 at equal optimizer updates; compare about 2,500 sandwich updates with 5,000 four-block updates for a block-count-matched view. Record tokens seen under each budget, final-only validation CE on the same windows and wall time. Extending the architecture to other pass counts is a diagnostic. The shipped setting motivating this design is three.

The short checks in this note can be rerun from the repository root with one Julia thread:

~~~sh
OPENBLAS_NUM_THREADS=1 julia -t1 --project -O0 loop/scratch/wvl/check.jl
~~~

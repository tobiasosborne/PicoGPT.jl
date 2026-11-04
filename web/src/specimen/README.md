# Browser specimen engine

Pure ES modules, Float64, no runtime dependencies or build step. Import
[`index.mjs`](./index.mjs) directly in the browser. Kernels and their hand-written
pullbacks live in `ops.mjs`; `tensor.mjs` supplies shaped column-major views.

```js
import {
  createModel, createBatch, forward, backward, step, trainStep,
  SPECIMEN_CONFIG, seededRandom, sample,
} from './specimen/index.mjs';

const model = createModel(SPECIMEN_CONFIG); // or createModel(config, exportedTheta0)
const batch = createBatch(
  [[7, 5, 1, 2, 3, 1, 5, 6]], // B sequences, T tokens each
  [[5, 1, 2, 3, 1, 5, 6, 1]], // next-token targets
);
const values = forward(model, batch);
const bars = backward(model, values);
values.blocks[0].Q.column(2);           // Float64Array view, third token column
values.blocks[0].A.attention(0, 0);     // cached T×T Tensor view, head 0 / sequence 0
values.blocks[0].A.attention(0).row(2); // copy of query 2's attention weights
bars.blocks[0].Xmid.column(2);         // corresponding covector
bars.theta;                           // flat ∇θ; bars.params has named views
step(model, { optimizer: 'adamw', lr: 1e-3, clipNorm: 1 });

const opt = { optimizer: 'adamw', lr: 1e-3, weightDecay: 0.01, clipNorm: 1 };
trainStep(model, batch, opt);          // forward + backward + step
const next = sample(forward(model, batch).Z.column(7), {
  temperature: 0.8, rng: seededRandom(42),
});
```

## Shapes, tokens, and ownership

`config` contains `V,D,H,F,L,T,eps`. `F` defaults to `4D`, `eps` to `1e-5`;
`D` must be divisible by `H`. A batch may use any length up to `config.T`.
`createBatch(x,y,{T,tokenBase})` accepts nested sequences or flat arrays;
for flat arrays, specify `T` to pack multiple sequences. Targets are optional for
inference. **Public token IDs default to 1-based**, matching Julia JSON. Use
`tokenBase: 0` explicitly for zero-based IDs. Prepared batches store zero-based
`Int32Array`s; reuse them in training. All tensor accessor indices are zero-based.
Token column `n = t + T*b` belongs to position `t` in sequence `b`.

Every tensor has `{shape, data: Float64Array}`, `get(...indices)`, `column(n)`,
`row(i,out?)`, `clone()`, and JSON serialization to `{shape,data}`. `column` shares
storage; rows need a copy because storage is column-major. `attention(h,b)` on a
four-dimensional tensor returns a shared matrix view with **queries in rows and
keys in columns**.

`forward` returns `X0`, `blocks[]`, `Xf`, `Z`, `probs`, `loss`. Each block has
`ln1,Q,K,V,S,A,O,Xmid,ln2,U,R,Y,Xout`, exactly as in BRIEF §5. Residual/normalized
arrays have shape `[D,N]`, `U,R` have `[F,N]`, `Z,probs` have `[V,N]`, and `S,A`
have `[T,T,H,B]`. `S` includes future scores; only `A` is causally masked.
`Y = W2*R+b2`; `Xf` is the final LayerNorm output. Loss is mean log-sum-exp
cross-entropy. Inference without targets sets `loss` to `NaN`.

`backward` returns the same trace structure, containing every intermediate's
adjoint, plus `theta` and `params`. Its default seed is `loss=1`. Since loss is
computed directly from logits, `probs` is an auxiliary output and its loss adjoint
is zero; `Z̄=(probs−onehot(y))/N`. Above-mask `S̄` is zero; above-mask `Ā` can be
nonzero because `O=V*Aᵀ` is defined on the entire matrix.

Each model owns one contiguous `theta`. `model.params.E.data`,
`model.params.blocks[0].Wq.data`, etc. are **subarrays of that same buffer**.
ASCII `gamma1,beta1,gamma2,beta2,gammaf,betaf` name γ/β. `model.layout` gives
`{name,shape,offset,size}` in the precise BRIEF §3 order. Supplied θ is copied;
the model's `theta` can be exported directly to Julia. Parameter count is computed
from the layout (1007 for the specimen).

Traces, covectors, optimizer results, and workspaces are reused and mutable.
Forward overwrites the trace for that `(T,B)`; backward overwrites all gradients;
clipping in `step` changes `model.grad`. Copy values with `tensor.clone()` or
`theta.slice()` when retaining a snapshot. Run backward before changing θ or
running another forward on that workspace. `trainStep` has no explicit allocations
after warming up its shape and optimizer. Interactive accessors/callbacks may
allocate; keep them outside the inner training loop. `workspace(model,T,B)` can
preallocate a shape explicitly.

## Initialization, optimization, and sampling

`init(config,seed,options?)` returns θ. `createModel(config)` uses seed 0.
`DEFAULT_INIT` documents the defaults: Gaussian matrices with std 0.02;
`Wo,W2` scaled by `1/√(2L)`; γ=1; β and biases=0. Override `matrixStd`,
`residualScaling`, `gamma`, `beta`, or `bias` in the third argument to adapt an
exporter's initialization specification. The deterministic RNG is Mulberry32
with Box–Muller Gaussian samples; Julia conformance uses exported `theta0`
rather than assuming identical random streams. `model.P` contains paired
sin/cos frequencies with zero-based positions exactly as in BRIEF §3.

`step` applies the current gradient; `trainStep` also runs forward and backward.
Both return a reused `{loss,gradNorm,clipScale,lr,step}` record; loss is before
the update. Options and defaults:

| Option | Default | Meaning |
|---|---:|---|
| `optimizer` | `'adamw'` | `'adamw'` or `'gd'` (`'sgd'` is an alias) |
| `lr` | `0.001` | learning rate |
| `beta1,beta2` | `0.9,0.999` | Adam moment coefficients |
| `eps` | `1e-8` | Adam denominator ε, outside `sqrt(v̂)` |
| `weightDecay` | `0` | decoupled decay of every parameter |
| `clipNorm` | `Infinity` | global-norm threshold, before moments/updates |
| `schedule` | absent | `{warmupSteps,totalSteps,minLR}` |

Adam uses bias correction. For update number `t` starting at 1, warmup uses
`lr*t/warmupSteps`; cosine then reaches `minLR` at `totalSteps` and stays there.
`learningRate(t,options)` exposes the same schedule. `resetOptimizer(model)`
clears moments and the update count. `globalNorm` and `clipGlobalNorm` are also
exported. The aliases `learning_rate`, `weight_decay`, `clip_norm`, `clip`, and
`epsilon` support JSON hyperparameters.

`sample(logitColumn,{temperature,rng,tokenBase})` returns one ID. Temperature
zero chooses the first maximum; positive temperature uses stable Gibbs weights.
`rng` may be a function or a `seededRandom(seed)` object.
`generate(model,prompt,count,options)` appends sampled IDs, truncating context
to the last `T` tokens and resetting positions for each sliding context.

## Interventions and general pullbacks

For direct manipulation or gradient checks,
`forward(model,batch,null,{after(name,tensor){...}})` invokes a callback after
each named node, before its consumers run. Names are `X0`, `blocks.0.S`, etc.
Mutate the tensor's data to intervene. ReLU uses derivative zero at zero.

`backward(model,trace,seeds)` pulls back an arbitrary scalar objective. Providing
seeds changes the default loss seed to zero. For example,
`{probs: probabilityCovector, blocks: [{S: scoreCovector}]}` pulls back the sum
of those two linear objectives. Each covector can be a tensor or flat array;
use the matching shape. Include `loss:1` to add the loss gradient. This also works
without target tokens.

## Verification and benchmark

From the repository root:

```sh
node --test web/src/specimen/
node web/src/specimen/benchmark.mjs        # optional argument: steps per round
node --test web/src/specimen/conformance.test.mjs
```

The local `package.json` only makes Node's explicit-directory test invocation
resolve to `run-tests.mjs`; it adds no dependencies. Browser imports use
`index.mjs`.

Tests perturb **every parameter**, including the full 11-sequence benchmark
batch, and **every entry of every named intermediate** in a two-block model.
They also check arbitrary output covectors, causality, independent sequences,
attention normalization, stable cross-entropy, initialization, optimizers,
scheduling, and sampling. If `artifacts/specimen.json` exists, conformance checks
layout offsets/shapes, P and every forward array (absolute tolerance `1e-12`),
∇θ (`1e-12`), every pre-update training loss (`1e-9`), and final θ (`1e-9`).
If absent, that test reports an explicit skip.

The corpus has 18 characters, so there are **10** length-8 inputs with complete
next-token targets. The benchmark uses the requested **11 sequences** by repeating
the first complete window once; it never invents a final target. Conformance
always uses the artifact's batch verbatim. The benchmark reports warmed median,
minimum, and maximum µs per full forward + backward + clipped AdamW step over
five rounds, with 1500 warmup steps.

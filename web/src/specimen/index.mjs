import { Tensor } from './tensor.mjs';
import { affine, affinePullback, layerNorm, layerNormPullback, attentionScores,
  attentionProbabilities, attentionValues, attentionPullback, add, accumulate } from './ops.mjs';
export { Tensor } from './tensor.mjs';

/** @typedef {{V:number,D:number,H:number,F?:number,L:number,T:number,eps?:number}} Config */
/** @typedef {{x:Int32Array,y:Int32Array|null,T:number,B:number,N:number,tokenBase:0,prepared:true}} Batch */
/** @typedef {{name:string,shape:number[],offset:number,size:number}} LayoutEntry */
/** @typedef {{ln1:Tensor,Q:Tensor,K:Tensor,V:Tensor,S:Tensor,A:Tensor,O:Tensor,Xmid:Tensor,ln2:Tensor,U:Tensor,R:Tensor,Y:Tensor,Xout:Tensor}} BlockTrace */
/** @typedef {{X0:Tensor,blocks:BlockTrace[],Xf:Tensor,Z:Tensor,probs:Tensor,loss:number}} Trace */
/** @typedef {{matrixStd?:number,gamma?:number,beta?:number,bias?:number,residualScaling?:boolean}} InitOptions */
/** @typedef {{lr?:number,warmupSteps?:number,totalSteps?:number,minLR?:number}} Schedule */
/** @typedef {{optimizer?:'adamw'|'gd'|'sgd',lr?:number,beta1?:number,beta2?:number,eps?:number,weightDecay?:number,clipNorm?:number,schedule?:Schedule}} OptimizerOptions */
/** @typedef {{E:Tensor,blocks:Object<string,Tensor>[],gammaf:Tensor,betaf:Tensor,Wout:Tensor,bout:Tensor}} Parameters */
/** @typedef {{config:Config,theta:Float64Array,layout:LayoutEntry[],params:Parameters,grad:Float64Array,gradParams:Parameters,P:Tensor,workspaces:Map<number,any>,trace:Trace|null,optimizer:any}} Model */
export const SPECIMEN_CONFIG = Object.freeze({ V: 7, D: 8, H: 2, F: 32, L: 1, T: 8, eps: 1e-5 });
export const DEFAULT_INIT = Object.freeze({ matrixStd: 0.02, gamma: 1, beta: 0, bias: 0, residualScaling: true });

function configuration(config) {
  const c = { ...config, F: config.F ?? 4 * config.D, eps: config.eps ?? 1e-5 };
  for (const k of ['V', 'D', 'H', 'F', 'T']) {
    if (!Number.isSafeInteger(c[k]) || c[k] < 1) throw new RangeError(`${k} must be a positive integer`);
  }
  if (!Number.isSafeInteger(c.L) || c.L < 0 || c.D % c.H !== 0) throw new RangeError('Require L ≥ 0 and D divisible by H');
  if (!(c.eps > 0) || !Number.isFinite(c.eps)) throw new RangeError('eps must be finite and positive');
  return Object.freeze(c);
}

/** θ order is exactly BRIEF §3; offsets and all array indices are zero-based.
 * @param {Config} config @returns {LayoutEntry[]} */
export function parameterLayout(config) {
  const { V, D, F, L } = configuration(config);
  const layout = [];
  let offset = 0;
  const put = (name, shape) => {
    const size = shape.reduce((a, b) => a * b, 1);
    layout.push({ name, shape, offset, size }); offset += size;
  };
  put('E', [D, V]);
  for (let l = 0; l < L; l++) {
    const p = `blocks.${l}.`;
    for (const name of ['gamma1', 'beta1']) put(p + name, [D]);
    for (const name of ['q', 'k', 'v', 'o']) { put(p + 'W' + name, [D, D]); put(p + 'b' + name, [D]); }
    for (const name of ['gamma2', 'beta2']) put(p + name, [D]);
    put(p + 'W1', [F, D]); put(p + 'b1', [F]);
    put(p + 'W2', [D, F]); put(p + 'b2', [D]);
  }
  put('gammaf', [D]); put('betaf', [D]); put('Wout', [V, D]); put('bout', [V]);
  return layout;
}

function parameterViews(theta, layout, L) {
  const params = { blocks: Array.from({ length: L }, () => ({})) };
  for (const { name, shape, offset, size } of layout) {
    const tensor = new Tensor(shape, theta.subarray(offset, offset + size));
    const path = name.split('.');
    if (path.length === 1) params[name] = tensor;
    else params.blocks[+path[1]][path[2]] = tensor;
  }
  return params;
}

/** Mulberry32 + Box–Muller. Deterministic in JS; Julia conformance uses exported θ₀.
 * @param {number} [seed] @returns {{uniform:()=>number,normal:()=>number}} */
export function seededRandom(seed = 0) {
  if (!Number.isInteger(seed)) throw new TypeError('seed must be an integer');
  let state = seed >>> 0, spare = null;
  function uniform() {
    state = (state + 0x6D2B79F5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  function normal() {
    if (spare !== null) { const z = spare; spare = null; return z; }
    const r = Math.sqrt(-2 * Math.log(1 - uniform())), a = 2 * Math.PI * uniform();
    spare = r * Math.sin(a);
    return r * Math.cos(a);
  }
  return { uniform, normal };
}

/** Return an initialized flat θ. Override distributions here or via the third argument.
 * @param {Config} config @param {number} [seed] @param {InitOptions} [options] @returns {Float64Array} */
export function init(config, seed = 0, options = {}) {
  const c = configuration(config), layout = parameterLayout(c), last = layout.at(-1);
  const theta = new Float64Array(last.offset + last.size);
  const rng = seededRandom(seed), spec = { ...DEFAULT_INIT, ...options };
  for (const { name, shape, offset, size } of layout) {
    const key = name.split('.').at(-1);
    if (shape.length === 2) {
      const scale = spec.residualScaling && (key === 'Wo' || key === 'W2') ? 1 / Math.sqrt(2 * c.L) : 1;
      for (let i = 0; i < size; i++) theta[offset + i] = spec.matrixStd * scale * rng.normal();
    } else theta.fill(key.startsWith('gamma') ? spec.gamma : key.startsWith('beta') ? spec.beta : spec.bias, offset, offset + size);
  }
  return theta;
}

/** Fixed P[:,pos], pos=0,…,T−1, paired sine/cosine frequencies (also for odd D).
 * @param {number} D @param {number} T @returns {Tensor} */
export function positionalEncoding(D, T) {
  const P = new Tensor([D, T]);
  for (let t = 0; t < T; t++) for (let r = 0; r < D; r++) {
    const phase = t / 10000 ** (2 * Math.floor(r / 2) / D);
    P.data[r + D * t] = r % 2 === 0 ? Math.sin(phase) : Math.cos(phase);
  }
  return P;
}

/** A model owns one θ, named Tensor views, optimizer state, and reusable workspaces.
 * @param {Config} [config] @param {ArrayLike<number>} [theta] @returns {Model} */
export function createModel(config = SPECIMEN_CONFIG, theta) {
  const c = configuration(config), layout = parameterLayout(c), last = layout.at(-1);
  const values = theta === undefined ? init(c) : Float64Array.from(theta);
  if (values.length !== last.offset + last.size) throw new RangeError('Wrong θ length');
  const grad = new Float64Array(values.length);
  return { config: c, theta: values, layout, params: parameterViews(values, layout, c.L),
    grad, gradParams: parameterViews(grad, layout, c.L), P: positionalEncoding(c.D, c.T),
    workspaces: new Map(), trace: null, optimizer: null };
}

/** Prepare tokens once, outside a live training loop. Nested arrays mean B sequences.
 * @param {ArrayLike<number>|ArrayLike<number>[]} x
 * @param {ArrayLike<number>|ArrayLike<number>[]|null} [y]
 * @param {{T?:number,tokenBase?:0|1}} [options] @returns {Batch} */
export function createBatch(x, y = null, { T, tokenBase = 1 } = {}) {
  if (tokenBase !== 0 && tokenBase !== 1) throw new RangeError('tokenBase must be 0 or 1');
  const nested = Array.isArray(x[0]) || ArrayBuffer.isView(x[0]);
  T ??= nested ? x[0].length : x.length;
  if (!Number.isInteger(T) || T < 1) throw new RangeError('Invalid sequence length');
  const B = nested ? x.length : x.length / T;
  if (!Number.isInteger(B) || B < 1) throw new RangeError('Batch length must be a positive multiple of T');
  const N = T * B, tx = new Int32Array(N), ty = y === null ? null : new Int32Array(N);
  if (nested && x.some(row => row.length !== T)) throw new RangeError('Ragged x batch');
  if (y !== null && (nested ? y.length !== B || y.some(row => row.length !== T) : y.length !== N)) throw new RangeError('x/y shape mismatch');
  for (let n = 0; n < N; n++) {
    const a = nested ? x[Math.floor(n / T)][n % T] : x[n];
    const b = y === null ? 0 : nested ? y[Math.floor(n / T)][n % T] : y[n];
    if (!Number.isSafeInteger(a) || (y !== null && !Number.isSafeInteger(b))) throw new TypeError('Token IDs must be integers');
    if (a < tokenBase || a - tokenBase > 0x7fffffff || (y !== null && (b < tokenBase || b - tokenBase > 0x7fffffff))) throw new RangeError('Token ID outside Int32 range');
    tx[n] = a - tokenBase;
    if (ty) ty[n] = b - tokenBase;
  }
  return { x: tx, y: ty, T, B, N, tokenBase: 0, prepared: true };
}

function normCache(D, N) { return { normalized: new Float64Array(D * N), invstd: new Float64Array(N) }; }
function traceArrays(c, T, B) {
  const { D, H, F, V, L } = c, N = T * B;
  const t = (m) => new Tensor([m, N]);
  return { X0: t(D), blocks: Array.from({ length: L }, () => ({
    ln1: t(D), Q: t(D), K: t(D), V: t(D), S: new Tensor([T, T, H, B]), A: new Tensor([T, T, H, B]),
    O: t(D), Xmid: t(D), ln2: t(D), U: t(F), R: t(F), Y: t(D), Xout: t(D),
  })), Xf: t(D), Z: t(V), probs: t(V), loss: NaN };
}

/** Allocate once for each (T,B). Call explicitly to warm up a figure before animating.
 * @param {Model} model @param {number} T @param {number} B */
export function workspace(model, T, B) {
  const key = (T + B) * (T + B + 1) / 2 + B;
  if (model.workspaces.has(key)) return model.workspaces.get(key);
  if (!Number.isInteger(T) || T < 1 || T > model.config.T || !Number.isInteger(B) || B < 1) throw new RangeError('Invalid workspace dimensions');
  const { D, L } = model.config, N = T * B;
  const trace = traceArrays(model.config, T, B), bars = traceArrays(model.config, T, B);
  const allBars = [bars.X0, bars.Xf, bars.Z, bars.probs];
  for (const block of bars.blocks) allBars.push(...Object.values(block));
  const w = { model, T, B, N, trace, bars, allBars, batch: null,
    norms: Array.from({ length: 2 * L + 1 }, () => normCache(D, N)),
    projected: new Float64Array(D * N) };
  Object.defineProperty(trace, '_workspace', { value: w });
  Object.defineProperty(bars, 'theta', { value: model.grad, enumerable: false });
  Object.defineProperty(bars, 'params', { value: model.gradParams, enumerable: false });
  model.workspaces.set(key, w);
  return w;
}

function validateBatch(model, batch) {
  const { V } = model.config;
  if (batch.x.length !== batch.N || batch.N !== batch.T * batch.B || (batch.y && batch.y.length !== batch.N)) throw new RangeError('Invalid prepared batch shape');
  for (let i = 0; i < batch.N; i++) {
    if (batch.x[i] < 0 || batch.x[i] >= V || (batch.y && (batch.y[i] < 0 || batch.y[i] >= V))) throw new RangeError('Token ID outside vocabulary');
  }
}

/**
 * Forward values, reused on the next call of this shape. Batch is createBatch() or {x,y}.
 * options.after(name,tensor) can intervene on a node before downstream computations;
 * paths are X0, blocks.0.ln1, …, Xf, Z, probs. Omit it for the fast training path.
 * @param {Model} model @param {Batch|{x:any,y?:any,T?:number,tokenBase?:0|1}|any[]} input
 * @param {any} [y] @param {{after?:(name:string,tensor:Tensor)=>void}|null} [options] @returns {Trace}
 */
export function forward(model, input, y = null, options = null) {
  let batch;
  if (input.prepared) batch = input;
  else if (input.x !== undefined) batch = createBatch(input.x, input.y ?? null, { T: input.T, tokenBase: input.tokenBase ?? 1 });
  else batch = createBatch(input, y);
  validateBatch(model, batch);
  const w = workspace(model, batch.T, batch.B), f = w.trace;
  const { D, H, V, L, F, eps } = model.config, { T, B, N } = w, p = model.params;
  const after = options?.after;
  w.batch = batch;
  for (let n = 0; n < N; n++) for (let i = 0; i < D; i++) f.X0.data[n * D + i] = p.E.data[batch.x[n] * D + i] + model.P.data[(n % T) * D + i];
  if (after) after('X0', f.X0);
  let X = f.X0.data;
  for (let l = 0; l < L; l++) {
    const b = f.blocks[l], q = p.blocks[l];
    const prefix = after ? `blocks.${l}.` : '';
    layerNorm(X, q.gamma1.data, q.beta1.data, b.ln1.data, w.norms[2 * l], D, N, eps);
    if (after) after(prefix + 'ln1', b.ln1);
    affine(q.Wq.data, q.bq.data, b.ln1.data, b.Q.data, D, D, N);
    if (after) after(prefix + 'Q', b.Q);
    affine(q.Wk.data, q.bk.data, b.ln1.data, b.K.data, D, D, N);
    if (after) after(prefix + 'K', b.K);
    affine(q.Wv.data, q.bv.data, b.ln1.data, b.V.data, D, D, N);
    if (after) after(prefix + 'V', b.V);
    attentionScores(b.Q.data, b.K.data, b.S.data, D, H, T, B);
    if (after) after(prefix + 'S', b.S);
    attentionProbabilities(b.S.data, b.A.data, T, H, B);
    if (after) after(prefix + 'A', b.A);
    attentionValues(b.V.data, b.A.data, b.O.data, D, H, T, B);
    if (after) after(prefix + 'O', b.O);
    affine(q.Wo.data, q.bo.data, b.O.data, w.projected, D, D, N);
    add(X, w.projected, b.Xmid.data);
    if (after) after(prefix + 'Xmid', b.Xmid);
    layerNorm(b.Xmid.data, q.gamma2.data, q.beta2.data, b.ln2.data, w.norms[2 * l + 1], D, N, eps);
    if (after) after(prefix + 'ln2', b.ln2);
    affine(q.W1.data, q.b1.data, b.ln2.data, b.U.data, F, D, N);
    if (after) after(prefix + 'U', b.U);
    for (let i = 0; i < b.U.data.length; i++) b.R.data[i] = Math.max(0, b.U.data[i]);
    if (after) after(prefix + 'R', b.R);
    affine(q.W2.data, q.b2.data, b.R.data, b.Y.data, D, F, N);
    if (after) after(prefix + 'Y', b.Y);
    add(b.Xmid.data, b.Y.data, b.Xout.data);
    if (after) after(prefix + 'Xout', b.Xout);
    X = b.Xout.data;
  }
  layerNorm(X, p.gammaf.data, p.betaf.data, f.Xf.data, w.norms[2 * L], D, N, eps);
  if (after) after('Xf', f.Xf);
  affine(p.Wout.data, p.bout.data, f.Xf.data, f.Z.data, V, D, N);
  if (after) after('Z', f.Z);
  let loss = 0;
  for (let n = 0; n < N; n++) {
    const off = n * V;
    let max = -Infinity, sum = 0;
    for (let a = 0; a < V; a++) max = Math.max(max, f.Z.data[off + a]);
    for (let a = 0; a < V; a++) { const e = Math.exp(f.Z.data[off + a] - max); f.probs.data[off + a] = e; sum += e; }
    for (let a = 0; a < V; a++) f.probs.data[off + a] /= sum;
    if (batch.y) loss += max + Math.log(sum) - f.Z.data[off + batch.y[n]];
  }
  if (after) after('probs', f.probs);
  f.loss = batch.y ? loss / N : NaN;
  model.trace = f;
  return f;
}

/**
 * Pull back loss (default) or arbitrary node covectors. Result mirrors the forward trace,
 * plus .theta and .params. seeds={loss:0,Z:Tensor,blocks:[{A:Tensor}]} is a VJP.
 * probs is an auxiliary output: its adjoint for the log-sum-exp loss is zero.
 * @param {Model} model @param {Trace} [trace] @param {any} [seeds]
 * @returns {Trace & {theta:Float64Array,params:Parameters}}
 */
export function backward(model, trace = model.trace, seeds = null) {
  if (!trace || trace._workspace?.model !== model) throw new TypeError('Trace belongs to another model or no forward has run');
  const w = trace._workspace, f = trace, g = w.bars;
  const { D, H, F, V, L } = model.config, { T, B, N, batch } = w, p = model.params, dp = model.gradParams;
  model.grad.fill(0);
  for (const t of w.allBars) t.data.fill(0);
  const lossSeed = seeds?.loss ?? (seeds ? 0 : 1);
  g.loss = lossSeed;
  if (seeds) seedTrace(g, seeds);
  if (lossSeed !== 0 && !batch.y) throw new TypeError('Loss pullback requires target tokens');
  for (let n = 0; n < N; n++) {
    const off = n * V;
    // Recompute softmax(Z) for the loss; probs may have been independently intervened on.
    let max = -Infinity, sum = 0, projection = 0;
    for (let a = 0; a < V; a++) max = Math.max(max, f.Z.data[off + a]);
    for (let a = 0; a < V; a++) sum += Math.exp(f.Z.data[off + a] - max);
    for (let a = 0; a < V; a++) projection += g.probs.data[off + a] * f.probs.data[off + a];
    for (let a = 0; a < V; a++) {
      g.Z.data[off + a] += lossSeed * (Math.exp(f.Z.data[off + a] - max) / sum - (batch.y && a === batch.y[n] ? 1 : 0)) / N;
      g.Z.data[off + a] += f.probs.data[off + a] * (g.probs.data[off + a] - projection);
    }
  }
  affinePullback(p.Wout.data, f.Xf.data, g.Z.data, dp.Wout.data, dp.bout.data, g.Xf.data, V, D, N);
  let dX = L ? g.blocks[L - 1].Xout.data : g.X0.data;
  layerNormPullback(p.gammaf.data, g.Xf.data, dp.gammaf.data, dp.betaf.data, dX, w.norms[2 * L], D, N);
  for (let l = L - 1; l >= 0; l--) {
    const b = f.blocks[l], db = g.blocks[l], q = p.blocks[l], dq = dp.blocks[l];
    dX = l ? g.blocks[l - 1].Xout.data : g.X0.data;
    accumulate(db.Xmid.data, db.Xout.data); accumulate(db.Y.data, db.Xout.data);
    affinePullback(q.W2.data, b.R.data, db.Y.data, dq.W2.data, dq.b2.data, db.R.data, D, F, N);
    for (let i = 0; i < b.U.data.length; i++) db.U.data[i] += b.U.data[i] > 0 ? db.R.data[i] : 0;
    affinePullback(q.W1.data, b.ln2.data, db.U.data, dq.W1.data, dq.b1.data, db.ln2.data, F, D, N);
    layerNormPullback(q.gamma2.data, db.ln2.data, dq.gamma2.data, dq.beta2.data, db.Xmid.data, w.norms[2 * l + 1], D, N);
    accumulate(dX, db.Xmid.data);
    affinePullback(q.Wo.data, b.O.data, db.Xmid.data, dq.Wo.data, dq.bo.data, db.O.data, D, D, N);
    attentionPullback(b.Q.data, b.K.data, b.V.data, b.A.data, db.O.data, db.Q.data, db.K.data, db.V.data, db.S.data, db.A.data, D, H, T, B);
    affinePullback(q.Wq.data, b.ln1.data, db.Q.data, dq.Wq.data, dq.bq.data, db.ln1.data, D, D, N);
    affinePullback(q.Wk.data, b.ln1.data, db.K.data, dq.Wk.data, dq.bk.data, db.ln1.data, D, D, N);
    affinePullback(q.Wv.data, b.ln1.data, db.V.data, dq.Wv.data, dq.bv.data, db.ln1.data, D, D, N);
    layerNormPullback(q.gamma1.data, db.ln1.data, dq.gamma1.data, dq.beta1.data, dX, w.norms[2 * l], D, N);
  }
  for (let n = 0; n < N; n++) for (let i = 0; i < D; i++) dp.E.data[batch.x[n] * D + i] += g.X0.data[n * D + i];
  return g;
}

function seedTrace(bars, seeds) {
  for (const name of ['X0', 'Xf', 'Z', 'probs']) if (seeds[name]) seedCovector(bars[name], seeds[name]);
  if (seeds.blocks) for (let l = 0; l < seeds.blocks.length; l++) {
    if (!seeds.blocks[l]) continue;
    for (const [name, value] of Object.entries(seeds.blocks[l])) seedCovector(bars.blocks[l]?.[name], value);
  }
}

function seedCovector(target, value) {
  const data = value.data ?? value;
  if (!target || target.data.length !== data.length) throw new RangeError('Covector shape mismatch');
  accumulate(target.data, data);
}

/** Stable Euclidean norm, avoiding overflow even for large finite gradients.
 * @param {ArrayLike<number>} values @returns {number} */
export function globalNorm(values) {
  let scale = 0, ssq = 1;
  for (let i = 0; i < values.length; i++) {
    const a = Math.abs(values[i]);
    if (a === 0) continue;
    if (!Number.isFinite(a)) return a;
    if (scale < a) { ssq = 1 + ssq * (scale / a) ** 2; scale = a; }
    else ssq += (a / scale) ** 2;
  }
  return scale * Math.sqrt(ssq);
}

/** Clip in place, returning the original norm. A zero threshold really gives zero.
 * @param {Float64Array} grad @param {number} [maxNorm] @returns {number} */
export function clipGlobalNorm(grad, maxNorm = Infinity) {
  if (!(maxNorm >= 0)) throw new RangeError('clipNorm must be nonnegative');
  const norm = globalNorm(grad);
  if (!Number.isFinite(norm)) throw new RangeError('Nonfinite gradient');
  if (norm > maxNorm) {
    const scale = maxNorm / norm;
    for (let i = 0; i < grad.length; i++) grad[i] *= scale;
  }
  return norm;
}

/** Update numbers start at 1: linear warmup, then cosine to minLR at totalSteps.
 * @param {number} t @param {Schedule} [options] @returns {number} */
export function learningRate(t, { lr = 1e-3, warmupSteps = 0, totalSteps = 0, minLR = 0 } = {}) {
  return scheduledRate(t, lr, warmupSteps, totalSteps, minLR);
}

function scheduledRate(t, lr, warmupSteps, totalSteps, minLR) {
  if (warmupSteps > 0 && t <= warmupSteps) return lr * t / warmupSteps;
  if (totalSteps <= warmupSteps) return lr;
  const progress = Math.min(1, Math.max(0, (t - warmupSteps) / (totalSteps - warmupSteps)));
  return minLR + 0.5 * (lr - minLR) * (1 + Math.cos(Math.PI * progress));
}

/** Reset moments/count explicitly when beginning an independent training run.
 * @param {Model} model */
export function resetOptimizer(model) { model.optimizer = null; }

/**
 * Apply model.grad. AdamW decays EVERY θ entry, after global-norm clipping, with
 * bias-corrected moments and eps outside sqrt(v̂). GD uses optional decoupled decay.
 * Options: optimizer, lr, beta1, beta2, eps, weightDecay, clipNorm, schedule.
 * @param {Model} model @param {OptimizerOptions} [opt]
 * @returns {{loss:number,gradNorm:number,clipScale:number,lr:number,step:number}}
 */
export function step(model, opt = DEFAULT_OPTIMIZER) {
  const kind = opt.optimizer ?? opt.kind ?? 'adamw';
  if (kind !== 'adamw' && kind !== 'gd' && kind !== 'sgd') throw new RangeError('Unknown optimizer');
  const lr = opt.lr ?? opt.learning_rate ?? 1e-3;
  const beta1 = opt.beta1 ?? 0.9, beta2 = opt.beta2 ?? 0.999;
  const eps = opt.eps ?? opt.epsilon ?? 1e-8;
  const decay = opt.weightDecay ?? opt.weight_decay ?? 0;
  const clip = opt.clipNorm ?? opt.clip_norm ?? opt.clip ?? Infinity;
  if (!(lr >= 0 && Number.isFinite(lr)) || !(decay >= 0 && Number.isFinite(decay)) || !(eps > 0 && Number.isFinite(eps)) || !(beta1 >= 0 && beta1 < 1 && beta2 >= 0 && beta2 < 1)) throw new RangeError('Invalid optimizer hyperparameters');
  let state = model.optimizer;
  if (!state || state.kind !== kind) {
    state = model.optimizer = { kind, t: 0, m: new Float64Array(model.theta.length), v: new Float64Array(model.theta.length),
      result: { loss: NaN, gradNorm: NaN, clipScale: 1, lr, step: 0 } };
  }
  const norm = clipGlobalNorm(model.grad, clip), t = ++state.t;
  let rate = lr;
  if (opt.schedule) {
    const s = opt.schedule;
    const warmup = s.warmupSteps ?? s.warmup_steps ?? 0, total = s.totalSteps ?? s.total_steps ?? 0, min = s.minLR ?? s.min_lr ?? 0;
    rate = scheduledRate(t, lr, warmup, total, min);
  }
  const { theta, grad } = model;
  if (kind === 'adamw') {
    const c1 = 1 - beta1 ** t, c2 = 1 - beta2 ** t, { m, v } = state;
    for (let i = 0; i < theta.length; i++) {
      const g = grad[i];
      m[i] = beta1 * m[i] + (1 - beta1) * g;
      v[i] = beta2 * v[i] + (1 - beta2) * g * g;
      theta[i] -= rate * (m[i] / c1 / (Math.sqrt(v[i] / c2) + eps) + decay * theta[i]);
    }
  } else for (let i = 0; i < theta.length; i++) theta[i] -= rate * (grad[i] + decay * theta[i]);
  const r = state.result;
  r.loss = model.trace?.loss ?? NaN; r.gradNorm = norm; r.clipScale = norm > clip ? clip / norm : 1; r.lr = rate; r.step = t;
  return r;
}
export const DEFAULT_OPTIMIZER = Object.freeze({ optimizer: 'adamw', lr: 1e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: 0, clipNorm: Infinity });

/** Allocation-free after warmup if batch comes from createBatch(). Loss is pre-update.
 * @param {Model} model @param {Batch} batch @param {OptimizerOptions} [opt] */
export function trainStep(model, batch, opt = DEFAULT_OPTIMIZER) {
  forward(model, batch); backward(model); return step(model, opt);
}

/** Sample logits using the Gibbs distribution. Temperature=0 chooses the first maximum.
 * @param {ArrayLike<number>|Tensor} logits
 * @param {{temperature?:number,rng?:(()=>number)|{uniform:()=>number},tokenBase?:0|1}} [options] @returns {number} */
export function sample(logits, { temperature = 1, rng = Math.random, tokenBase = 1 } = {}) {
  const z = logits.data ?? logits;
  if (z.length === 0 || !(temperature >= 0) || !Number.isFinite(temperature) || (tokenBase !== 0 && tokenBase !== 1)) throw new RangeError('Invalid sampling input');
  let max = -Infinity, argmax = 0;
  for (let a = 0; a < z.length; a++) {
    if (!Number.isFinite(z[a])) throw new RangeError('Logits must be finite');
    if (z[a] > max) { max = z[a]; argmax = a; }
  }
  if (temperature === 0) return argmax + tokenBase;
  let sum = 0;
  for (let a = 0; a < z.length; a++) sum += Math.exp((z[a] - max) / temperature);
  const u = typeof rng === 'function' ? rng() : rng.uniform();
  if (!(u >= 0 && u < 1)) throw new RangeError('rng must return a number in [0,1)');
  let threshold = u * sum;
  for (let a = 0; a < z.length; a++) {
    threshold -= Math.exp((z[a] - max) / temperature);
    if (threshold < 0) return a + tokenBase;
  }
  return argmax + tokenBase;
}

/** Append token IDs; keep only the last T tokens in context and reset pos to zero.
 * @param {Model} model @param {ArrayLike<number>} prompt @param {number} count
 * @param {{temperature?:number,rng?:(()=>number)|{uniform:()=>number},tokenBase?:0|1}} [options] @returns {number[]} */
export function generate(model, prompt, count, { temperature = 1, rng = Math.random, tokenBase = 1 } = {}) {
  if (!Number.isSafeInteger(count) || count < 0 || prompt.length === 0) throw new RangeError('Require nonempty prompt and nonnegative count');
  const tokens = Array.from(prompt), T = model.config.T;
  for (let i = 0; i < count; i++) {
    const batch = createBatch(tokens.slice(-T), null, { tokenBase });
    const f = forward(model, batch);
    tokens.push(sample(f.Z.column(batch.N - 1), { temperature, rng, tokenBase }));
  }
  return tokens;
}

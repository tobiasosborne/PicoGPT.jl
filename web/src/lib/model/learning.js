// learning.js — training, gradients and adjoints of the specimen, for chapters 6–8.
//
// All initialization, scheduling and optimizer updates belong to the engine
// (web/src/specimen). This module adapts its buffers and records to the figure API.
//
// Figures reach this only through the adapter (lib/model/specimen.js).

import { createModel, createBatch, forward, backward, globalNorm, seededRandom,
	drawTheta, matrixMask, learningRate, prepareOptimizer, trainStep, SPECIMEN_RECIPE } from '#specimen';

const KEEP_GRADIENT = Object.freeze({ preserveGradient: true });

/** Every length-T window of the corpus that has a complete next-character target. */
export function trainingBatch(corpus, vocab, T) {
	const ids = Array.from(corpus, (c) => vocab.indexOf(c) + 1);
	const x = [], y = [];
	for (let s = 0; s + T < ids.length; s++) { x.push(ids.slice(s, s + T)); y.push(ids.slice(s + 1, s + T + 1)); }
	return { x, y, batch: createBatch(x, y) };
}

/**
 * The entropy floor of the training set: the smallest loss any model can reach.
 * A prediction is conditioned on its window prefix; a prefix followed by different
 * characters in different windows costs the entropy of those followers.
 * Returns {floor, ambiguous: [{context, followers:{char:count}, cost}]}.
 */
export function entropyFloor(corpus, T) {
	const groups = new Map();
	let N = 0;
	for (let s = 0; s + T < corpus.length; s++)
		for (let t = 1; t <= T; t++) {
			const ctx = corpus.slice(s, s + t), next = corpus[s + t];
			if (!groups.has(ctx)) groups.set(ctx, new Map());
			const g = groups.get(ctx);
			g.set(next, (g.get(next) ?? 0) + 1);
			N++;
		}
	let total = 0;
	const ambiguous = [];
	for (const [context, g] of groups) {
		const n = [...g.values()].reduce((a, b) => a + b, 0);
		let cost = 0;
		for (const k of g.values()) cost -= k * Math.log(k / n);
		if (g.size > 1) ambiguous.push({ context, followers: Object.fromEntries(g), cost });
		total += cost;
	}
	return { floor: total / N, ambiguous, N };
}

/** Warmup + cosine, update numbers from 1 (train.conventions.schedule). */
export function scheduleRate(t, { eta, warmup = 0, total = 0, etaMin = 0 }) {
	return learningRate(t, { eta, warmup, steps: total, eta_min: etaMin });
}

/** The artifact's recipe as options for `Run` (see train.hyper in the artifact). */
export function recipeFrom(hyper, steps) {
	return {
		optimizer: 'adamw',
		eta: hyper?.eta ?? SPECIMEN_RECIPE.eta,
		schedule: { warmup: hyper?.warmup ?? SPECIMEN_RECIPE.warmup, total: steps ?? SPECIMEN_RECIPE.steps, etaMin: hyper?.eta_min ?? SPECIMEN_RECIPE.eta_min },
		beta1: hyper?.beta1 ?? SPECIMEN_RECIPE.beta1,
		beta2: hyper?.beta2 ?? SPECIMEN_RECIPE.beta2,
		eps: hyper?.eps ?? SPECIMEN_RECIPE.eps,
		weightDecay: hyper?.weight_decay ?? SPECIMEN_RECIPE.weight_decay,
		clip: hyper?.clip ?? SPECIMEN_RECIPE.clip
	};
}

/**
 * A training run. opts: {optimizer:'adamw'|'gd', eta, schedule:{warmup,total,etaMin}|null,
 * beta1, beta2, eps, weightDecay (matrices only), clip}. `record` keeps per-step numbers:
 * loss (before the update), gradNorm (before clipping), lr, updateNorm.
 */
export class Run {
	constructor(config, theta0, batch, opts) {
		this.model = createModel(config, theta0);
		this.batch = batch;
		this.opts = { optimizer: 'adamw', eta: 0.02, schedule: null, beta1: 0.9, beta2: 0.95, eps: 1e-8, weightDecay: 0, clip: Infinity, ...opts };
		const state = prepareOptimizer(this.model, this.opts.optimizer);
		this.mask = state.mask;
		this.m = state.m;
		this.v = state.v;
		this.delta = state.delta; // the last update, θ_{t} − θ_{t−1}
		this.t = 0;
		this.loss = [];
		this.gradNorm = [];
		this.lr = [];
		this.updateNorm = [];
		this.clipped = [];
	}
	get theta() { return this.model.theta; }
	get grad() { return this.model.grad; }
	rate(t) { return this.opts.schedule ? scheduleRate(t, { eta: this.opts.eta, ...this.opts.schedule }) : this.opts.eta; }
	/** Forward and backward at the current θ; returns the loss. Leaves ∇θ in this.grad. */
	gradient() {
		const f = forward(this.model, this.batch);
		backward(this.model, f);
		return f.loss;
	}
	/** Loss at the current θ, without touching the gradient. */
	currentLoss() { return forward(this.model, this.batch).loss; }
	/** One update. Returns the loss before it. */
	step() {
		const r = trainStep(this.model, this.batch, this.opts, KEEP_GRADIENT);
		const state = this.model.optimizer;
		this.t = r.step;
		this.m = state.m; this.v = state.v; this.mask = state.mask; this.delta = state.delta;
		this.loss.push(r.loss);
		this.gradNorm.push(r.gradNorm);
		this.lr.push(r.lr);
		this.updateNorm.push(r.updateNorm);
		this.clipped.push(r.clipScale < 1);
		return r.loss;
	}
	/** n updates; returns the loss before the last one. */
	steps(n) { let l = NaN; for (let k = 0; k < n; k++) l = this.step(); return l; }
}

/**
 * Forward + backward of the full-batch loss at θ, as plain arrays per window.
 * For window b (0-based) and column t: values X0, ln1, Q, K, V, O, attn (=Wo O + bo),
 * Xmid, ln2, U, R, Y, Xout, Xf, Z, probs; and the adjoint of each (bars.*).
 * Attention per head: A, S and their adjoints (Ā includes the future; S̄ does not).
 */
export function gradTrace(config, theta, batch) {
	const model = createModel(config, theta);
	const f = forward(model, batch);
	const g = backward(model, f);
	const { D, H, T } = config, d = D / H, B = batch.B;
	const cols = (tensor, b, from = 0, to = tensor.shape[0]) =>
		Array.from({ length: T }, (_, t) => Array.from(tensor.column(b * T + t).subarray(from, to)));
	const blk = f.blocks[0], gb = g.blocks[0];
	const windows = [];
	for (let b = 0; b < B; b++) {
		const val = {}, bar = {};
		for (const k of ['X0', 'Xf', 'Z', 'probs']) { val[k] = cols(f[k], b); bar[k] = cols(g[k], b); }
		for (const k of ['ln1', 'Q', 'K', 'V', 'O', 'Xmid', 'ln2', 'U', 'R', 'Y', 'Xout']) { val[k] = cols(blk[k], b); bar[k] = cols(gb[k], b); }
		// the attention branch's update, Wo O + bo, and its adjoint (= X̄mid)
		val.attn = val.Xmid.map((xm, t) => xm.map((x, r) => x - val.X0[t][r]));
		bar.attn = bar.Xmid;
		const heads = [];
		for (let h = 0; h < H; h++) {
			const rows = (M) => Array.from({ length: T }, (_, i) => Array.from({ length: T }, (_, j) => M.get(i, j)));
			heads.push({
				A: rows(blk.A.attention(h, b)), S: rows(blk.S.attention(h, b)),
				Abar: rows(gb.A.attention(h, b)), Sbar: rows(gb.S.attention(h, b))
			});
		}
		windows.push({ val, bar, heads, x: Array.from(batch.x.subarray(b * T, b * T + T)), y: Array.from(batch.y.subarray(b * T, b * T + T)) });
	}
	const params = model.layout.map(({ name, shape, offset, size }) => ({
		name, shape, offset, size,
		values: Array.from(model.theta.subarray(offset, offset + size)),
		grad: Array.from(model.grad.subarray(offset, offset + size))
	}));
	return { loss: f.loss, windows, params, grad: Array.from(model.grad), gradNorm: globalNorm(model.grad) };
}

/**
 * A reusable evaluator over one model's buffers, for figures that run the engine many
 * times (gradient checks, curvature, minibatches, live attention). Every method takes θ
 * explicitly and returns fresh plain arrays; nothing leaks between calls.
 *   loss(θ, sub?)            full-batch (or sub-batch) loss
 *   grad(θ, sub?)            ∇θ as a Float64Array copy
 *   hvp(θ, v, h)             Hessian–vector product by central differences of ∇θ
 *   sharpness(θ, iters, v0)  top Hessian eigenvalue by power iteration → {lambda, v}
 *   windows(idx)             a prepared sub-batch of the training windows idx (0-based)
 *   look(θ, ids)             forward on one sequence of 1-based ids → {A[h][i][j], probs[t][a]}
 */
export function makeEvaluator(config, train) {
	const { x, y, batch } = train;
	const model = createModel(config);
	const p = model.theta.length;
	const subs = new Map();
	const set = (theta) => model.theta.set(theta);
	const ev = {
		p,
		layout: model.layout,
		B: x.length,
		windows(idx) {
			const key = idx.join(',');
			if (!subs.has(key)) subs.set(key, createBatch(idx.map((b) => x[b]), idx.map((b) => y[b])));
			return subs.get(key);
		},
		loss(theta, sub = batch) { set(theta); return forward(model, sub).loss; },
		grad(theta, sub = batch) { set(theta); backward(model, forward(model, sub)); return Float64Array.from(model.grad); },
		hvp(theta, v, h = 1e-5) {
			const a = Float64Array.from(theta), b = Float64Array.from(theta);
			for (let i = 0; i < p; i++) { a[i] += h * v[i]; b[i] -= h * v[i]; }
			const ga = ev.grad(a), gb = ev.grad(b);
			for (let i = 0; i < p; i++) ga[i] = (ga[i] - gb[i]) / (2 * h);
			return ga;
		},
		sharpness(theta, iters = 20, v0 = null) {
			let v = v0 ? Float64Array.from(v0) : Float64Array.from({ length: p }, (_, i) => Math.sin(1.7 * i + 0.3));
			let n = Math.hypot(...v), lambda = 0;
			for (let i = 0; i < p; i++) v[i] /= n;
			for (let k = 0; k < iters; k++) {
				const w = ev.hvp(theta, v);
				lambda = 0;
				for (let i = 0; i < p; i++) lambda += w[i] * v[i];
				n = Math.hypot(...w);
				for (let i = 0; i < p; i++) v[i] = w[i] / n;
			}
			return { lambda, v };
		},
		look(theta, ids) {
			set(theta);
			const f = forward(model, createBatch([ids]));
			const n = ids.length, b = f.blocks[0];
			const A = Array.from({ length: config.H }, (_, h) => {
				const M = b.A.attention(h, 0);
				return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => M.get(i, j)));
			});
			const probs = Array.from({ length: n }, (_, t) => Array.from(f.probs.column(t)));
			return { A, probs };
		}
	};
	return ev;
}

/** Greedy (τ = 0) continuation of a prompt (1-based ids) at θ: the ids appended. */
export function greedy(config, theta, prompt, count, model = null) {
	const m = model ?? createModel(config, theta);
	if (model) m.theta.set(theta);
	const ids = Array.from(prompt);
	for (let k = 0; k < count; k++) {
		const ctx = ids.slice(-config.T);
		const f = forward(m, createBatch([ctx]));
		const z = f.Z.column(ctx.length - 1);
		let best = 0;
		for (let a = 1; a < z.length; a++) if (z[a] > z[best]) best = a;
		ids.push(best + 1);
	}
	return ids.slice(prompt.length);
}

export { createModel, forward, backward, globalNorm, seededRandom, drawTheta, matrixMask };

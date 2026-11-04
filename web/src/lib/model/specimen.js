// specimen.js — the thin adapter between figures and the model engine.
//
// Figures never import the engine directly: they ask this object for plain,
// copied arrays (the engine reuses its buffers). Swap the engine or the data
// source here and every figure follows. Runs identically on server (SSR) and
// client (hydration), so a figure's static state is the same number everywhere.

import { createModel, createBatch, forward, parameterLayout, sample, seededRandom } from '#specimen';
import * as learning from './learning.js';

/**
 * @param {{config, vocab:string[], corpus:string, theta:number[], theta0?:number[], loss?:number[], provenance}} data
 */
export function makeSpecimen(data) {
	const config = data.config;
	const { T, H, D } = config;
	const d = D / H;
	const model = createModel(config, Float64Array.from(data.theta));
	const layout = parameterLayout(config);
	const windows = [];
	for (let s = 0; s + T < data.corpus.length; s++) windows.push(s);
	const cache = new Map();

	const ids = (text) => Array.from(text, (ch) => data.vocab.indexOf(ch) + 1);

	/** Column j of a [rows, N] tensor as a plain array (copied). */
	const col = (tensor, j, from = 0, to = tensor.shape[0]) => Array.from(tensor.column(j).subarray(from, to));

	/**
	 * Forward pass on the length-T window starting at corpus position `start`.
	 * Everything a figure might need about that pass, as plain nested arrays:
	 * per head h: A[h][i][j] (rows = readers i, columns = sources j), S[h][i][j]
	 * (scores BEFORE the mask, future entries included), q[h][i], k[h][j], v[h][j]
	 * (each d-vectors), o[h][i]; per position: X0, Xmid, Xout, ln1 (D-vectors), probs.
	 */
	function windowTrace(start) {
		if (cache.has(start)) return cache.get(start);
		const text = data.corpus.slice(start, start + T);
		const tr = forward(model, createBatch([ids(text)]));
		const b = tr.blocks[0];
		const n = text.length;
		const heads = [];
		for (let h = 0; h < H; h++) {
			const A = b.A.attention(h, 0), S = b.S.attention(h, 0);
			const rows = (M) => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => M.get(i, j)));
			heads.push({
				A: rows(A), S: rows(S),
				q: Array.from({ length: n }, (_, i) => col(b.Q, i, h * d, (h + 1) * d)),
				k: Array.from({ length: n }, (_, j) => col(b.K, j, h * d, (h + 1) * d)),
				v: Array.from({ length: n }, (_, j) => col(b.V, j, h * d, (h + 1) * d)),
				o: Array.from({ length: n }, (_, i) => col(b.O, i, h * d, (h + 1) * d))
			});
		}
		const per = (t) => Array.from({ length: n }, (_, i) => col(t, i));
		const out = {
			start, text, chars: Array.from(text), ids: ids(text), heads,
			X0: per(tr.X0), ln1: per(b.ln1), Xmid: per(b.Xmid), Xout: per(b.Xout), Z: per(tr.Z), probs: per(tr.probs)
		};
		cache.set(start, out);
		return out;
	}

	/** Named parameter views over a θ vector (default: the trained θ). */
	function params(theta = data.theta) {
		return layout.map(({ name, shape, offset, size }) => ({ name, shape, offset, size, values: theta.slice(offset, offset + size) }));
	}

	// ── Chapters 6–8: training, gradients and adjoints (lib/model/learning.js) ──────
	// Display and training windows both require a complete target (B = 10). `recipe` is the artifact's run; `run(opts)` starts a fresh one from θ₀
	// (or opts.theta0), and is checked against the artifact's loss trajectory to 1e-9 absolute.
	const train = learning.trainingBatch(data.corpus, data.vocab, T);
	const floorInfo = learning.entropyFloor(data.corpus, T);
	const recipe = learning.recipeFrom(data.provenance?.hyper, data.provenance?.steps);
	const gradCache = new Map();
	let evaluatorCache = null;
	const lab = {
		/** {x, y}: 1-based ids of the B training windows; batch: the prepared engine batch. */
		train,
		/** Entropy floor of the training set and the contexts that make it (computed). */
		floor: floorInfo.floor,
		ambiguous: floorInfo.ambiguous,
		recipe,
		steps: data.provenance?.steps ?? (data.loss?.length ? data.loss.length - 1 : 200),
		/** θ₀ for a seed: seed 0 is the book's θ₀; others are fresh draws from the same distributions. */
		initTheta(seed = 0) {
			if (!seed || !data.init?.params) return Float64Array.from(data.theta0 ?? data.theta);
			return learning.drawTheta(layout, data.init.params, seed);
		},
		/** A new training run (a learning.Run): opts override the recipe. */
		run(opts = {}) {
			const { theta0, ...rest } = opts;
			return new learning.Run(config, theta0 ?? data.theta0 ?? data.theta, train.batch, { ...recipe, ...rest });
		},
		/** Forward + backward of the full-batch loss at θ (default θ₀), every value and adjoint. */
		gradTrace(theta = data.theta0 ?? data.theta) {
			const key = theta === data.theta0 ? 'theta0' : theta === data.theta ? 'theta' : null;
			if (key && gradCache.has(key)) return gradCache.get(key);
			const out = learning.gradTrace(config, theta, train.batch);
			if (key) gradCache.set(key, out);
			return out;
		},
		/** Shared evaluator for figures that run the engine many times (learning.makeEvaluator). */
		evaluator() { return (evaluatorCache ??= learning.makeEvaluator(config, train)); },
		greedy(theta, prompt, count) { return learning.greedy(config, theta, ids(prompt), count).map((a) => data.vocab[a - 1]).join(''); },
		scheduleRate: learning.scheduleRate,
		matrixMask: () => learning.matrixMask(layout)
	};

	// ── Chapters 3, 4, 9: predictions, the run replayed, sampling ─────────────────
	// predictions(θ): every prediction of the training set at θ (one forward pass).
	// trajectory(): θ after every step of the recorded run, replayed from θ₀ with the
	// artifact's recipe and kept only if it reproduces the artifact's loss at every
	// step to 1e-9 (else null). speak(): the engine's sampler, each draw u recorded.
	const predCache = new WeakMap();
	function predictions(theta = data.theta) {
		if (predCache.has(theta)) return predCache.get(theta);
		const f = forward(createModel(config, Float64Array.from(theta)), train.batch);
		const N = train.batch.N;
		const out = {
			B: train.x.length, T, N, x: train.x, y: train.y, loss: f.loss,
			Z: Array.from({ length: N }, (_, n) => col(f.Z, n)),
			probs: Array.from({ length: N }, (_, n) => col(f.probs, n))
		};
		predCache.set(theta, out);
		return out;
	}
	let traj;
	function trajectory() {
		if (traj !== undefined) return traj;
		traj = null;
		const steps = data.provenance?.steps ?? 0;
		if (!data.theta0 || !steps || !(data.loss?.length > steps)) return traj;
		const r = lab.run();
		const thetas = [r.theta.slice()];
		for (let k = 0; k < steps; k++) { r.step(); thetas.push(r.theta.slice()); }
		const loss = [...r.loss, r.currentLoss()];
		const err = Math.max(...loss.map((l, k) => Math.abs(l - data.loss[k])));
		if (err <= 1e-9) traj = { steps, thetas, loss, err };
		return traj;
	}
	function speak(prompt, count, { temperature = 1, seed = 1 } = {}) {
		const rng = seededRandom(seed);
		const tokens = typeof prompt === 'string' ? ids(prompt) : Array.from(prompt);
		const steps = [];
		for (let k = 0; k < count; k++) {
			const context = tokens.slice(-T);
			const z = col(forward(model, createBatch([context])).Z, context.length - 1);
			let u = null;
			const next = sample(z, { temperature, rng: () => (u = rng.uniform()) });
			steps.push({ context, z, u, next });
			tokens.push(next);
		}
		return { tokens, steps };
	}

	return {
		config, d, vocab: data.vocab, corpus: data.corpus, windows, layout,
		paramCount: layout.at(-1).offset + layout.at(-1).size,
		theta: data.theta, theta0: data.theta0, loss: data.loss ?? [],
		provenance: data.provenance,
		windowTrace, params, ids,
		lab,
		predictions, trajectory, speak,
		// (extension, chapter 2) the fixed positional code: P[t] is column t (0-based) of P ∈ ℝ^{D×T}
		P: Array.from({ length: T }, (_, t) => col(model.P, t))
	};
}

/** Display form of a character: the space is drawn as an open box. */
export const glyphOf = (ch) => (ch === ' ' ? '␣' : ch);

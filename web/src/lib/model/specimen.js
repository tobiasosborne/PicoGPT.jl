// specimen.js — the thin adapter between figures and the model engine.
//
// Figures never import the engine directly: they ask this object for plain,
// copied arrays (the engine reuses its buffers). Swap the engine or the data
// source here and every figure follows. Runs identically on server (SSR) and
// client (hydration), so a figure's static state is the same number everywhere.

import { createModel, createBatch, forward, parameterLayout } from '#specimen';

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
	for (let s = 0; s + T <= data.corpus.length; s++) windows.push(s);
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

	return {
		config, d, vocab: data.vocab, corpus: data.corpus, windows, layout,
		paramCount: layout.at(-1).offset + layout.at(-1).size,
		theta: data.theta, theta0: data.theta0, loss: data.loss ?? [],
		provenance: data.provenance,
		windowTrace, params, ids
	};
}

/** Display form of a character: the space is drawn as an open box. */
export const glyphOf = (ch) => (ch === ' ' ? '␣' : ch);

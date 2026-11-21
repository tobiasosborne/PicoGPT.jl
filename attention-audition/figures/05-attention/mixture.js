// mixture.js — reading the mean-field export (attention-audition/artifacts/mixture.json)
// in figures. Plain functions, no state, no numbers of their own.
//
// The export's tensors are {shape, data} in Julia's column-major order; its windows b,
// positions i and heads are 1-based. Every index taken here is 0-based. One reader of one
// head, as the figure draws it:
//   A[j]       attention weight of source j (j = 0 … i−1, the reader itself last)
//   P[j][y]    the prediction with this head's output replaced by v_j, the other head held
//              at its usual output ("source j alone")
//   mix[y]     Σ_j A_j P_j(y), the arithmetic mixture (the export's p_single)
//   geo[y]     ∏_j P_j(y)^A_j / Z, the normalised geometric mixture (computed here, from
//              the logarithms, so that a tiny P_j(y) is not lost)
//   p[y]       the model's own prediction (both head outputs at their means)

/** Entry of a column-major tensor {shape, data} at 0-based indices. */
export function at(t, ...idx) {
	let k = 0, m = 1;
	for (let d = 0; d < idx.length; d++) { k += idx[d] * m; m *= t.shape[d]; }
	return t.data[k];
}

/** Total variation ½ Σ |p − q|. */
export const tv = (p, q) => p.reduce((s, v, k) => s + Math.abs(v - q[k]), 0) / 2;

export const argmax = (v) => v.reduce((b, x, k) => (x > v[b] ? k : b), 0);

/** The single-head slice of one reader record: everything the plate draws. */
export function slice(rec, head, V) {
	const h = head - 1;
	const n = rec.i;
	const A = Array.from({ length: n }, (_, j) => at(rec.attention_rows, j, h));
	const P = Array.from({ length: n }, (_, j) => Array.from({ length: V }, (_, y) => at(rec.p_by_source, y, j, h)));
	const p = rec.p.data.slice();
	const mix = Array.from({ length: V }, (_, y) => at(rec.p_single, y, h));
	// weighted surprise of each character under each source: A_j · (−ln P_j(y))
	const S = P.map((row, j) => row.map((q) => A[j] * -Math.log(q)));
	const Sbar = Array.from({ length: V }, (_, y) => S.reduce((s, row) => s + row[y], 0));
	const m = Math.min(...Sbar);
	const w = Sbar.map((e) => Math.exp(-(e - m)));
	const Z = w.reduce((a, b) => a + b, 0);
	const geo = w.map((x) => x / Z);
	return {
		b: rec.b, i: rec.i, window: rec.window, head,
		target: rec.target_id - 1,
		chars: Array.from(rec.window),
		A, P, p, mix, geo, S, Sbar,
		tvMix: tv(mix, p), tvGeo: tv(geo, p)
	};
}

/** The source that votes the arithmetic mixture's favourite out: the largest weighted
 *  surprise A_j · (−ln P_j(ŷ)) at ŷ = argmax of the arithmetic mixture. */
export function dissenter(s) {
	const y = argmax(s.mix);
	return s.S.reduce((b, row, j) => (row[y] > s.S[b][y] ? j : b), 0);
}

/** Deterministic beeswarm on one axis: each dot at the free offset nearest the line
 *  (alternating sides), so no two dots overlap. xs in px; returns offsets in px. */
export function swarm(xs, r) {
	const order = xs.map((x, k) => ({ x, k })).sort((a, b) => a.x - b.x);
	const placed = [];
	const off = new Array(xs.length).fill(0);
	const d = 2 * r + 0.6;
	for (const { x, k } of order) {
		const near = placed.filter((q) => Math.abs(q.x - x) < d);
		let best = 0;
		for (let s = 0; s < 60; s++) {
			const cand = s === 0 ? 0 : (s % 2 ? 1 : -1) * Math.ceil(s / 2) * (d * 0.5);
			if (near.every((q) => (q.x - x) ** 2 + (q.y - cand) ** 2 >= d * d)) { best = cand; break; }
		}
		placed.push({ x, y: best });
		off[k] = best;
	}
	return off;
}

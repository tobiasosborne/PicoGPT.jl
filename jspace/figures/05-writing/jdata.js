// jdata.js — reading the J-lens exports (jspace/artifacts/*-jlens.json) in figures of
// chapters 5 and 6. Pure functions of the data; no numbers of their own.
//
// The exports store every array of rank ≥ 2 as {shape, data}, column-major (Julia order),
// with 1-based mathematical indices in the README. Here every index is 0-based:
// at(t, i, j, k) is the README's t[i+1, j+1, k+1].

/** Entry of a column-major {shape, data} tensor at 0-based indices. */
export function at(t, ...idx) {
	let o = 0, m = 1;
	for (let k = 0; k < idx.length; k++) { o += idx[k] * m; m *= t.shape[k]; }
	return t.data[o];
}

/** A whole fibre along axis `axis` (0-based), the other indices fixed (null at `axis`). */
export function fibre(t, idx, axis) {
	return Array.from({ length: t.shape[axis] }, (_, k) => at(t, ...idx.map((v, d) => (d === axis ? k : v))));
}

/** The specimen export's token column n (0-based) ↔ window b and position t (0-based). */
export const column = (b, t, T) => b * T + t;

/** Probability column of the specimen's interned intervention table, for a readout id
 *  (1-based; 0 means "beyond the window" and returns null). */
export function tableProbs(J, id) {
	if (!id) return null;
	const tab = J.interventions.probability_table;
	return Array.from({ length: tab.shape[0] }, (_, a) => at(tab, a, id - 1));
}

/** Lens vector v_a at depth index j, as an array of D numbers: (Wout J_j)[a, :]. */
export const lensVector = (J, a, j) => fibre(J.lenses.vectors, [a, null, j], 1);

/** Depth label in the book's notation: 0, ½, 1, 1½, … */
export function depthLabel(d) {
	const whole = Math.floor(d), half = d - whole > 0.25;
	if (!half) return String(whole);
	return whole ? `${whole}½` : '½';
}

export const dot = (u, v) => u.reduce((s, x, k) => s + x * v[k], 0);
export const norm = (u) => Math.sqrt(dot(u, u));

/** Total variation distance between two distributions. */
export const tv = (p, q) => 0.5 * p.reduce((s, x, k) => s + Math.abs(x - q[k]), 0);

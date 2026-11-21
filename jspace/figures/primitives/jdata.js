// jdata.js — reading the J-lens exports (jspace/artifacts/*-jlens.json) in figures.
//
// Shared by the fig-j-* figures. Plain functions, no state, no numbers of their own.
// The exports' tensors are {shape, data} in Julia's column-major order with 1-based
// indices in their documentation (jspace/artifacts/README.md); every index taken by the
// functions here is 0-based. Depth index j ↔ depth[j] = j/2 (0, ½, 1, …, L).

/** Entry of a column-major tensor {shape, data} at 0-based indices. */
export function at(t, ...idx) {
	let k = 0, m = 1;
	for (let d = 0; d < idx.length; d++) { k += idx[d] * m; m *= t.shape[d]; }
	return t.data[k];
}

/** tensor[:, :, ...rest] as an accessor (i, j) ↦ value. */
export const matrixOf = (t, ...rest) => (i, j) => at(t, i, j, ...rest);

/** tensor[:, ...rest] as a plain array (the fibre along the first axis). */
export function fibre(t, ...rest) {
	return Array.from({ length: t.shape[0] }, (_, i) => at(t, i, ...rest));
}

/** Rows of an m×n accessor, as nested arrays (for MatrixGlyph `rows`). */
export function rowsOf(get, m, n = m) {
	return Array.from({ length: m }, (_, i) => Array.from({ length: n }, (_, j) => get(i, j)));
}

/** Index of the largest entry (the first, on ties). */
export function argmax(v) {
	let b = 0;
	for (let k = 1; k < v.length; k++) if (v[k] > v[b]) b = k;
	return b;
}

/** Frobenius norm of an m×n accessor. */
export function frob(get, m, n = m) {
	let s = 0;
	for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) s += get(i, j) ** 2;
	return Math.sqrt(s);
}

/** Total variation distance ½ Σ |p − q| between two distributions. */
export const tv = (p, q) => p.reduce((s, v, k) => s + Math.abs(v - q[k]), 0) / 2;

/** A depth as the book prints it: 0, ½, 1, 1½, 2, … */
export function depthLabel(d) {
	const w = Math.floor(d + 1e-9);
	return d - w > 0.25 ? (w ? `${w}½` : '½') : String(w);
}

/** What happened just before depth d, in words (depth_kind from the export). */
export function depthWords(d, kind) {
	if (kind === 'embed' || d === 0) return 'after the embedding';
	const b = Math.ceil(d - 1e-9);
	return kind === 'attention' ? `after the attention of block ${b}` : `after block ${b}`;
}

/** Horizontal (or vertical) positions of depth indices on an axis from a to b. */
export function depthScale(count, a, b) {
	return (j) => (count > 1 ? a + ((b - a) * j) / (count - 1) : (a + b) / 2);
}

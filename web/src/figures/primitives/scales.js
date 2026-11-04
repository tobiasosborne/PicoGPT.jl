// scales.js — shared encodings. Every glyph that shows numbers on a common
// footing (small multiples) must take its scale from here, from ALL the numbers
// it will be compared with — never per glyph.

/** Largest |x| over (nested) arrays; never 0. */
export function maxAbs(...arrays) {
	let m = 0;
	const walk = (a) => { for (const x of a) Array.isArray(x) || ArrayBuffer.isView(x) ? walk(x) : (m = Math.max(m, Math.abs(x))); };
	walk(arrays);
	return m || 1;
}

/** Spoke glyph geometry: |x| = max reaches rmax. Pass max over EVERY glyph compared. */
export function spokeScale(rmax, max) {
	return { rmax, scale: rmax / max };
}

/** Radius for an area-encoded dot: area ∝ value / vmax (value ≥ 0). */
export const areaRadius = (value, vmax, rmax) => rmax * Math.sqrt(Math.max(0, value) / vmax);

/** Square side for an area-encoded square (Hinton): area ∝ |value| / vmax. */
export const areaSide = (value, vmax, smax) => smax * Math.sqrt(Math.min(1, Math.abs(value) / vmax));

/** Number formats. Book convention: minus sign U+2212, no leading zero only in dense tables. */
export function fmt(x, digits = 2) {
	if (!Number.isFinite(x)) return x > 0 ? '∞' : x < 0 ? '−∞' : '—';
	const s = Math.abs(x).toFixed(digits);
	return (x < 0 && +s !== 0 ? '−' : '') + s;
}
export function fmtDense(x, digits = 2) {
	const s = fmt(x, digits);
	return s.replace(/^(−?)0\./, '$1.');
}

/** Characters as drawn in figures: the space becomes an open box. */
export const showChar = (ch) => (ch === ' ' ? '␣' : ch);

/** Smallest of the user's motion preference and an explicit override. */
export function reducedMotion() {
	return typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches;
}

export const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
export const lerp = (a, b, t) => a + (b - a) * t;
/** Smooth phase: maps t∈[a,b] to [0,1] with cubic ease, clamped. */
export function phase(t, a, b) {
	const u = clamp((t - a) / (b - a), 0, 1);
	return u * u * (3 - 2 * u);
}

/** Print figures are laid out at width / PRINT_SCALE and scaled down to the column,
 *  so a 12 px label prints at about 7 pt (12 px × 0.75 pt/px × 0.78). */
export const PRINT_SCALE = 0.78;
export const layoutWidth = (measured, mode, min = 320) => Math.max(min, mode === 'print' ? measured / PRINT_SCALE : measured);

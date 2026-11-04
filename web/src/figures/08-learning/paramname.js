// Names of the parameter arrays as the book sets them: an italic symbol and a
// subscript (E, γ₁, W_q, b_out …), from the layout name ('blocks.0.Wq', 'gammaf', …).
// Also small helpers for log axes shared by the chapter 8 figures.

export function paramName(name) {
	const base = name.split('.').at(-1);
	const m = base.match(/^(gamma|beta)(\d|f)$/);
	if (m) return { sym: m[1] === 'gamma' ? 'γ' : 'β', sub: m[2] };
	const w = base.match(/^([Wb])(q|k|v|o|1|2|out)$/);
	if (w) return { sym: w[1], sub: w[2] };
	return { sym: base, sub: '' };
}

/** Plain-text form, e.g. "W_q" → "Wq" with a Unicode subscript where one exists. */
const SUBS = { 1: '₁', 2: '₂', q: 'q', k: 'k', v: 'v', o: 'ₒ', f: 'f', out: 'out' };
export function paramText(name) {
	const { sym, sub } = paramName(name);
	if (!sub) return sym;
	return sym + (/^\d$/.test(sub) ? SUBS[sub] : '_' + sub);
}

const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
/** 10^e as type: "10⁻⁸"; e = 0 → "1"; e = 1 → "10". */
export function pow10(e) {
	if (e === 0) return '1';
	if (e === 1) return '10';
	return '10' + String(e).replace('-', '−').split('').map((c) => (c === '−' ? '⁻' : SUP[c])).join('');
}

/** a × 10^e in type, with `digits` significant figures: "3.1 × 10⁻⁵". */
export function sci(x, digits = 2) {
	if (x === 0) return '0';
	if (!Number.isFinite(x)) return x > 0 ? '∞' : '−∞';
	const e = Math.floor(Math.log10(Math.abs(x)));
	if (e >= -2 && e <= 2) return (x < 0 ? '−' : '') + Math.abs(x).toFixed(Math.max(0, digits - 1 - e));
	let m = Math.abs(x) / 10 ** e, ee = e;
	if (+m.toFixed(digits - 1) >= 10) { m /= 10; ee += 1; }
	return (x < 0 ? '−' : '') + m.toFixed(digits - 1) + ' × ' + pow10(ee);
}

/** Deterministic jitter in [0, 1) for mark i (no randomness in figures). */
export const jitter = (i) => {
	const s = Math.sin(i * 12.9898 + 78.233) * 43758.5453;
	return s - Math.floor(s);
};

// initvar.js — the computation behind fig-init-variance (and its key): the root-mean-
// square size of each intermediate of the forward pass at initialisation, for the book's
// variances and with every weight matrix except E scaled (standard deviation ÷3, ×3),
// over SEEDS seeds (seed 0 is the book's θ₀). Also the normalised attention entropy
// and the initial loss.

export const INIT_KEY = 'fig-init-variance:v1';

export const SCHEMES = [
	{ f: 1 / 3, name: 'variances ÷ 9', short: 'variance ÷ 9' },
	{ f: 1, name: 'the book’s variances', short: 'the book’s' },
	{ f: 3, name: 'variances × 9', short: 'variance × 9' }
];
export const SEEDS = 8;
// rows: [key, symbol, subscript, words]
export const ROWS = [
	['X0', 'X', '0', 'the stream, embedded'],
	['Q', 'q', '', 'queries'],
	['S', 'S', '', 'scores q·k/√d'],
	['attn', 'W', 'o', 'attention’s update'],
	['Xmid', 'X', 'mid', 'the stream'],
	['U', 'u', '', 'MLP pre-activation'],
	['Y', 'y', '', 'MLP’s update'],
	['Xout', 'X', 'out', 'the stream'],
	['Z', 'z', '', 'logits']
];

const rms = (cols) => { let s = 0, n = 0; for (const c of cols) for (const x of c) { s += x * x; n++; } return Math.sqrt(s / n); };

export function initJob(specimen) {
	return function* () {
		const lay = specimen.layout;
		const out = [];
		for (const sc of SCHEMES) {
			const seeds = [];
			for (let seed = 0; seed < SEEDS; seed++) {
				const theta = Float64Array.from(specimen.lab.initTheta(seed));
				for (const l of lay) if (l.shape.length === 2 && l.name !== 'E') for (let i = l.offset; i < l.offset + l.size; i++) theta[i] *= sc.f;
				const tr = specimen.lab.gradTrace(theta);
				const all = (k) => tr.windows.flatMap((w) => w.val[k]);
				const v = {};
				for (const [k] of ROWS) if (k !== 'S') v[k] = rms(all(k));
				// scores and entropy over the allowed (causal) entries only
				let ss = 0, ns = 0, H = 0, nh = 0;
				for (const w of tr.windows) for (const h of w.heads) for (let i = 0; i < h.S.length; i++) {
					let e = 0;
					for (let j = 0; j <= i; j++) { ss += h.S[i][j] ** 2; ns++; const a = h.A[i][j]; if (a > 0) e -= a * Math.log(a); }
					if (i > 0) { H += e / Math.log(i + 1); nh++; }
				}
				v.S = Math.sqrt(ss / ns);
				v.H = H / nh;
				v.loss = tr.loss;
				seeds.push(v);
				yield;
			}
			out.push(seeds);
		}
		return out;
	};
}


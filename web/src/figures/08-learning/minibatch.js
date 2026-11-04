// minibatch.js — the computation behind fig-minibatch (and its key): every minibatch
// gradient of the specimen at θ₀ and at the trained θ, projected onto a fixed plane.
// u₁ = the direction of the full gradient g; u₂ = the main direction in which the single
// windows disagree with it (orthogonal to u₁). For b = 1, 2, 5, 10 windows: every subset,
// its real gradient (a backward pass on that sub-batch), its projection, and the rms
// distance from g, in ℝᵖ (rms) and in the plane (rmsP).

export const MINIBATCH_KEY = 'fig-minibatch:v2';
export const SIZES = [1, 2, 5, 10];

function subsets(n, k) {
	const out = [], cur = [];
	const rec = (s) => {
		if (cur.length === k) { out.push(cur.slice()); return; }
		for (let i = s; i < n; i++) { cur.push(i); rec(i + 1); cur.pop(); }
	};
	rec(0);
	return out;
}
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const norm = (a) => Math.sqrt(dot(a, a));

export function minibatchJob(specimen) {
	return function* () {
		const ev = specimen.lab.evaluator();
		const M = ev.B;
		const out = [];
		for (const theta of [specimen.theta0, specimen.theta]) {
			const g = ev.grad(theta);
			const gn = norm(g);
			const u1 = g.map((x) => x / gn);
			const single = Array.from({ length: M }, (_, b) => ev.grad(theta, ev.windows([b])));
			// main direction of disagreement, orthogonal to u1: top eigenvector of the Gram
			// matrix of the deviations (power iteration on an M×M matrix)
			const dev = single.map((v) => { const d = v.map((x, i) => x - g[i]); const c = dot(d, u1); return d.map((x, i) => x - c * u1[i]); });
			const K = dev.map((a) => dev.map((b) => dot(a, b)));
			let a = Array.from({ length: M }, (_, i) => Math.cos(i + 0.4));
			for (let it = 0; it < 400; it++) { const b = K.map((r) => dot(r, a)); const n = norm(b); a = b.map((x) => x / n); }
			let u2 = new Float64Array(g.length);
			a.forEach((c, b) => { for (let i = 0; i < u2.length; i++) u2[i] += c * dev[b][i]; });
			const n2 = norm(u2);
			u2 = u2.map((x) => x / n2);
			const P = (v) => [dot(v, u1), dot(v, u2)];
			const clouds = [];
			let meanErr = 0;
			for (const b of SIZES) {
				const sets = subsets(M, b);
				const pts = [];
				const mean = new Float64Array(g.length);
				let ss = 0, ssP = 0;
				const Pg = P(g);
				for (const S of sets) {
					const v = b === 1 ? single[S[0]] : b === M ? g : ev.grad(theta, ev.windows(S));
					for (let i = 0; i < mean.length; i++) mean[i] += v[i] / sets.length;
					let d2 = 0;
					for (let i = 0; i < v.length; i++) d2 += (v[i] - g[i]) ** 2;
					ss += d2;
					const pv = P(v);
					ssP += (pv[0] - Pg[0]) ** 2 + (pv[1] - Pg[1]) ** 2;
					pts.push([...pv, S.join('.'), norm(v)]);
				}
				let e = 0;
				for (let i = 0; i < mean.length; i++) e = Math.max(e, Math.abs(mean[i] - g[i]));
				meanErr = Math.max(meanErr, e);
				clouds.push({ b, n: sets.length, pts, rms: Math.sqrt(ss / sets.length), rmsP: Math.sqrt(ssP / sets.length) });
				yield;
			}
			out.push({ g: P(g), gnorm: gn, clouds, meanErr });
		}
		return out;
	};
}

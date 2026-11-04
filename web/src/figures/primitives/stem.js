// stem.js — the numbers behind the residual-stem glyph (ResidualStem.svelte).
//
// From one window of specimen.lab.gradTrace(θ) it extracts, per token column t:
//   forward  x0 (= X0), a (the attention update Wo o + bo), mid (= Xmid),
//            y (the MLP update), out (= Xout), and the next-character distribution;
//   backward x̄0, x̄mid, x̄out (the adjoints of the stream), and what each branch hands
//            back to the stream: att = x̄0 − x̄mid, mlp = x̄mid − x̄out (adjoints add at a fork);
//            z̄ = (p − e_y)/N, the seed the loss sends into the column.
// and, between columns, the exact decomposition of the attention branch:
//   arcF[i][j] = Σ_h Wo_h A^h_ij v^h_j                 what source j hands to reader i
//   arcB[i][j] = LN₁*_j( Σ_h Wv_hᵀ A^h_ij ō^h_i + Wk_hᵀ S̄^h_ij q^h_i/√d )
//                                                       what reader i hands back to source j
// (both D-vectors; Σ_j arcF[i][j] + bo = a_i, and Σ_i arcB[i][j] plus the query term
// LN₁*_j(Wqᵀ q̄_j) = att_j). Norms are reported as RMS, ‖v‖/√D, like the stem widths.

export const rms = (v) => Math.hypot(...v) / Math.sqrt(v.length);
const sub = (a, b) => a.map((x, k) => x - b[k]);

/** Column-major D×D (or r×c) parameter → accessor M(i, j). */
function matrix(params, name) {
	const p = params.find((q) => q.name.endsWith(name));
	const r = p.shape[0];
	return { at: (i, j) => p.values[i + r * j], rows: r, cols: p.shape[1] ?? 1, values: p.values };
}

/** LayerNorm pullback for one column x with gain γ (ε as in the model). */
function lnPullback(x, gamma, ybar, eps) {
	const D = x.length;
	const mu = x.reduce((s, v) => s + v, 0) / D;
	const u = x.map((v) => v - mu);
	const sigma = Math.sqrt(u.reduce((s, v) => s + v * v, 0) / D + eps);
	const xh = u.map((v) => v / sigma);
	const g = ybar.map((v, k) => v * gamma[k]);
	const mg = g.reduce((s, v) => s + v, 0) / D;
	const mgx = g.reduce((s, v, k) => s + v * xh[k], 0) / D;
	return g.map((v, k) => (v - mg - xh[k] * mgx) / sigma);
}

/**
 * Everything the stem figures draw, for window b (0-based) of a gradTrace.
 * Returns {cols: [...T], arcF, arcB} (arcs as D-vectors; use rms for widths).
 */
export function stemWindow(gt, b, config) {
	const { D, H, T } = config;
	const d = D / H, eps = config.eps ?? 1e-5;
	const w = gt.windows[b];
	const Wo = matrix(gt.params, 'Wo'), Wv = matrix(gt.params, 'Wv'), Wk = matrix(gt.params, 'Wk'), Wq = matrix(gt.params, 'Wq');
	const gamma1 = gt.params.find((q) => q.name.endsWith('gamma1')).values;
	const N = gt.windows.length * T;

	const cols = Array.from({ length: T }, (_, t) => {
		const v = w.val, g = w.bar;
		const probs = v.probs[t];
		let pred = 0;
		for (let a = 1; a < probs.length; a++) if (probs[a] > probs[pred]) pred = a;
		const target = w.y[t]; // gradTrace keeps the prepared batch's 0-based ids
		return {
			x0: v.X0[t], a: v.attn[t], mid: v.Xmid[t], y: v.Y[t], out: v.Xout[t],
			probs, pred, target,
			loss: -Math.log(probs[target]),
			zbar: g.Z[t],
			bx0: g.X0[t], bmid: g.Xmid[t], bout: g.Xout[t],
			batt: sub(g.X0[t], g.Xmid[t]),
			bmlp: sub(g.Xmid[t], g.Xout[t]),
			N
		};
	});

	const arcF = Array.from({ length: T }, () => Array(T).fill(null));
	const arcB = Array.from({ length: T }, () => Array(T).fill(null));
	for (let i = 0; i < T; i++)
		for (let j = 0; j <= i; j++) {
			// forward: Σ_h Wo[:, h-rows] (A_ij v_j^h)
			const f = new Array(D).fill(0);
			const back = new Array(D).fill(0);
			for (let h = 0; h < H; h++) {
				const A = w.heads[h].A[i][j], Sb = w.heads[h].Sbar[i][j];
				for (let r = 0; r < d; r++) {
					const row = h * d + r;
					const vj = w.val.V[j][row], obar = w.bar.O[i][row], qi = w.val.Q[i][row];
					for (let k = 0; k < D; k++) {
						f[k] += Wo.at(k, row) * A * vj;
						back[k] += Wv.at(row, k) * A * obar + (Wk.at(row, k) * Sb * qi) / Math.sqrt(d);
					}
				}
			}
			arcF[i][j] = f;
			arcB[i][j] = lnPullback(w.val.X0[j], gamma1, back, eps);
		}
	// what each column hands back to itself through its own query (no arc: it stays on its branch)
	const query = Array.from({ length: T }, (_, j) => {
		const back = new Array(D).fill(0);
		for (let r = 0; r < D; r++) for (let k = 0; k < D; k++) back[k] += Wq.at(r, k) * w.bar.Q[j][r];
		return lnPullback(w.val.X0[j], gamma1, back, eps);
	});
	return { cols, arcF, arcB, query, x: w.x, y: w.y };
}

/**
 * Shared geometry of the stem figures (fig-residual-stem, fig-forward-backward).
 * Wide: row labels | one detail stem (the selected column, with spoke glyphs of radius
 * rG) | an overview of all T stems (no glyphs). Narrow: the overview alone, full width.
 * The stations (Y) are shared, so every row reads straight across.
 */
export function stemGeometry({ W, T, top = 60, narrow = false }) {
	const padR = narrow ? 4 : 6;
	const padL = narrow ? 4 : 96;
	const rG = narrow ? 0 : Math.max(20, Math.min(32, W * 0.036));
	const dx = padL + (narrow ? 0 : rG + 4);
	const detail = { x: dx, bx: dx + 2 * rG + 14 };
	const detailW = narrow ? 0 : detail.bx + rG + 12 - padL;
	const ox = narrow ? padL : padL + detailW + 10;
	const lane = (W - padR - ox) / T;
	const xs = Array.from({ length: T }, (_, t) => ox + lane * (t + 0.3));
	const bxs = xs.map((x) => x + lane * 0.42);
	const leg = narrow ? 22 : rG + 8;
	const Y = {};
	Y.char = top;
	Y.x0 = top + 12 + Math.max(rG, 8);
	Y.f1 = Y.x0 + rG + 12;
	Y.n1 = Y.f1 + leg;
	Y.p1 = Y.n1 + leg;
	Y.mid = Y.p1 + rG + 10;
	Y.f2 = Y.mid + rG + 12;
	Y.n2 = Y.f2 + leg;
	Y.p2 = Y.n2 + leg;
	Y.out = Y.p2 + rG + 10;
	Y.end = Y.out + rG + 12;
	return { padL, padR, rG, detailW, detail, ox, lane, xs, bxs, Y };
}

// The J-space supplement: an edition of the PicoGPT pipeline (see web/src/lib/editions/book.js
// for what an edition module is). Build it with BOOK_EDITION=jspace (web) or
// `node print/build.mjs --edition=jspace` (PDF). Plain data and pure functions only: this
// module is imported by client code as well as by the server.

export const OUTLINE = [
	{ num: 0, slug: 'prologue', title: 'Prologue', gist: 'What the paper claims, in plain words. What we will compute, on which models, and what we cannot.' },
	{ num: 1, slug: 'lenses', title: 'Reading the stream early', gist: 'The residual stream at every depth; the readout; the logit lens and why it fails early: the coordinates change between layers.' },
	{ num: 2, slug: 'jacobian', title: 'The averaged Jacobian', gist: 'The derivative of the rest of the model, one 8×8 block per pair of positions; averaging over positions and texts separates disposition from use; computing it by pullbacks.' },
	{ num: 3, slug: 'jlens', title: 'The Jacobian lens', gist: 'Replace the rest of the model by its averaged derivative. The J-lens vectors are covectors: the averaged gradient of one logit. Readouts at every depth.' },
	{ num: 4, slug: 'jspace', title: 'The J-space', gist: 'Sparse nonnegative combinations of J-lens vectors: a union of cones, not a subspace. The distance to it, the J-component and the remainder, measured.' },
	{ num: 5, slug: 'writing', title: 'Reading and writing', gist: 'Probes, steering, ablation, and swapping one concept for another in lens coordinates with a pseudoinverse; what each does to the model’s output.' },
	{ num: 6, slug: 'workspace', title: 'Is it a workspace?', gist: 'The paper’s structural signatures, tested on a small model: a band of depths, a limited capacity, and weights that broadcast J-space directions.' },
	{ num: 7, slug: 'epilogue', title: 'Epilogue', gist: 'What the paper does that we could not, what the small models leave open, and a dictionary.', unnumbered: true }
];

export const BOOK = {
	title: 'Poised to speak',
	subtitle: 'The Jacobian lens and the J‑space, worked through on a transformer with 1007 parameters, and one with 810,049',
	author: 'Tobias J. Osborne',
	repo: 'https://github.com/tobiasosborne/PicoGPT.jl',
	subject: 'A supplement to PicoGPT: the Jacobian lens and the J-space of Gurnee, Sofroniew, Lindsey et al. (2026), computed on two small transformers. Print edition.',
	description: 'A supplement to PicoGPT: the mathematics of the Jacobian lens and the J-space, every number computed on a transformer small enough to see whole.'
};

export const PATHS = {
	chapters: 'jspace/book/chapters',
	figures: 'jspace/figures',
	traces: 'jspace/traces'
};

export const FRONT = {
	theta: false,
	facts: false,
	map: false,
	note: 'A supplement to <i>PicoGPT</i>. Every number in it comes from the book’s two models: the specimen, with 1007 parameters, and the Tiny Shakespeare model, with 810,049.',
	provenance: 'Every number in every figure comes from the two models of <i>PicoGPT</i>: the specimen (artifacts/specimen.json) and the Tiny Shakespeare model, through the analysis in jspace/src/JLens.jl, which uses the book’s hand-written pullbacks and no automatic differentiation. Its exports are jspace/artifacts/*.json. The paper is Gurnee, Sofroniew, Lindsey et al., <i>Verbalizable Representations Form a Global Workspace in Language Models</i>, Transformer Circuits Thread, 2026.'
};

export const PRINT = { out: 'jspace/PoisedToSpeak.pdf', work: 'jspace/.work', proof: 'jspace/proof' };

// ── data for the figures ──────────────────────────────────────────────────────────
// Figures of this edition are named fig-j-*. They read the J-lens exports through
// getContext('artifacts')(): `jspace.specimen` and `jspace.shakespeare`, the files whole
// (the print page carries every figure, so condensing per figure would gain little).
export function artifacts(ids, read) {
	if (!ids.some((id) => /^fig-j-/.test(id))) return {};
	return {
		jspace: {
			specimen: read('jspace/artifacts/specimen-jlens.json'),
			shakespeare: read('jspace/artifacts/shakespeare-jlens.json')
		}
	};
}

// ── load-bearing prose readings ({{value:name}}) ─────────────────────────────────
// Filled in once the analysis export exists; unknown names fail the build (BOOK_STRICT).
export function proseValues(read) {
	const a = read('artifacts/specimen.json');
	const values = {
		specimenParams: String(a.param_count),
		shakespeareParams: (810049).toLocaleString('en-GB')
	};
	Object.assign(values, writingValues(read));
	Object.assign(values, workspaceValues(read));
	Object.assign(values, lensValues(read));
	Object.assign(values, jlensValues(read));
	return values;
}

// Chapters 1–2 (Reading the stream early; The averaged Jacobian): logit-lens and Jacobian readings.
function lensValues(read) {
	const S = read('jspace/artifacts/specimen-jlens.json'), G = read('jspace/artifacts/shakespeare-jlens.json');
	const at = (t, ...i) => { let o = 0, m = 1; i.forEach((k, d) => { o += k * m; m *= t.shape[d]; }); return t.data[o]; };
	const V = S.vocab.length, D = S.model.config.D, N = S.lenses.model_probs.shape[1];
	const top = (t, ...rest) => { let b = 0; for (let a = 1; a < V; a++) if (at(t, a, ...rest) > at(t, b, ...rest)) b = a; return b; };
	const agree = (j) => { let k = 0; for (let n = 0; n < N; n++) k += top(S.lenses.logit_probs, n, j) === top(S.lenses.model_probs, n); return String(k); };
	const pct = (x) => String(Math.round(100 * x));
	const gb = (x) => x.toLocaleString('en-GB');
	// the worked example of chapter 1: window 1, position 7, depth ½ (the logit lens must disagree)
	const ex = 6, half = S.depth.indexOf(0.5), ch = (a) => S.vocab[a];
	if (top(S.lenses.logit_probs, ex, half) === top(S.lenses.model_probs, ex)) throw new Error('lensExample: the logit lens agrees with the model at the worked example');
	const M = G.lenses.metrics.map((m) => m.logit_lens);
	if (M.at(-2).model_agreement >= 0.5) throw new Error('lensShLogitAgreeLast: the prose says the last update changes the top character at more than half of the positions');
	// chapter 2: one window's averaged Jacobian against the grand average, at depth ½
	const P = S.jacobians.per_window_attention, J = S.jacobians.averaged;
	const dev = Array.from({ length: P.shape[2] }, (_, b) => {
		let d = 0, n = 0;
		for (let i = 0; i < D; i++) for (let q = 0; q < D; q++) { d += (at(P, i, q, b) - at(J, i, q, half)) ** 2; n += at(J, i, q, half) ** 2; }
		return Math.sqrt(d / n);
	});
	return {
		lensSpecimenColumns: String(N),
		lensLogitAgreeZero: agree(0),
		lensLogitAgreeHalf: agree(half),
		lensExampleModel: ch(top(S.lenses.model_probs, ex)),
		lensExampleLogit: ch(top(S.lenses.logit_probs, ex, half)),
		lensShPositions: gb(G.sampling.activation_positions),
		lensShPerWindow: String(G.sampling.activation_positions_per_window.length),
		lensShWindows: String(G.sampling.windows),
		lensShLogitAgreeZero: pct(M[0].model_agreement),
		lensShLogitAccZero: pct(M[0].top1_accuracy),
		lensShAccModel: pct(M.at(-1).top1_accuracy),
		lensShLogitAgreeLast: pct(M.at(-2).model_agreement),
		jacWindowDevMin: pct(Math.min(...dev)),
		jacWindowDevMax: pct(Math.max(...dev)),
		jacShSources: gb(G.sampling.jacobian_positions),
		jacShPairs: gb(G.sampling.causal_pairs),
		jacShLength: String(G.sampling.window_length)
	};
}

// Chapter 5 (Reading and writing): readings of the J-lens exports' interventions.
function writingValues(read) {
	const S = read('jspace/artifacts/specimen-jlens.json'), G = read('jspace/artifacts/shakespeare-jlens.json');
	const at = (t, ...i) => { let o = 0, m = 1; i.forEach((k, d) => { o += k * m; m *= t.shape[d]; }); return t.data[o]; };
	const depth = (d) => (d % 1 ? (Math.floor(d) ? `${Math.floor(d)}½` : '½') : String(d));
	const q = (ch) => `‘${ch === ' ' ? '␣' : ch}’`;
	const name = (ch) => (ch === '\n' ? 'a line break' : ch === ' ' ? 'a space' : q(ch));
	const list = (xs) => (xs.length < 2 ? xs.join('') : `${xs.slice(0, -1).join(', ')} and ${xs.at(-1)}`);
	const V = S.vocab.length, N = S.lenses.model_probs.shape[1], A = S.interventions.alphas;
	const probs = (id) => Array.from({ length: V }, (_, c) => at(S.interventions.probability_table, c, id - 1));
	const top = (p) => p.indexOf(Math.max(...p));
	// steering: of the (column, character) pairs whose character is not already the model's choice,
	// how many does α make the model's choice, at each depth
	const steer = (j, alpha) => {
		const B = S.interventions.by_depth[j], i = A.indexOf(alpha);
		let made = 0, all = 0;
		for (let n = 0; n < N; n++) {
			const base = top(Array.from({ length: V }, (_, c) => at(S.lenses.model_probs, c, n)));
			for (let c = 0; c < V; c++) if (c !== base) { all++; if (top(probs(at(B.steer, n, c, i))) === c) made++; }
		}
		return `${made} of ${all}`;
	};
	// the concept split: J-component vs remainder, and cos(v_a, r_a)
	const comp = S.interventions.concept_components, iJ = comp.indexOf('j_k3'), iR = comp.indexOf('remainder'), iV = comp.indexOf('lens_vector');
	const cases = [];
	S.interventions.by_depth.forEach((B, j) => {
		for (let c = 0; c < V; c++) {
			const vec = (k) => Array.from({ length: B.concept_vectors.shape[0] }, (_, r) => at(B.concept_vectors, r, c, k));
			const r = vec(iR), v = vec(iV), dot = (x, y) => x.reduce((s, xi, k) => s + xi * y[k], 0);
			const cos = dot(v, r) / Math.sqrt(dot(v, v) * dot(r, r) || 1);
			cases.push({ j, c, J: at(B.concept_mean_delta_logp, c, iJ), R: at(B.concept_mean_delta_logp, c, iR), cos });
		}
	});
	const ex = cases.filter((k) => k.R >= k.J);
	const byDepth = [...new Set(ex.map((k) => k.j))].map((j) => `${list(ex.filter((k) => k.j === j).map((k) => q(S.vocab[k.c])))} at depth ${depth(S.depth[j])}`);
	const far = ex.filter((k) => Math.abs(k.cos) >= 0.01), zero = ex.filter((k) => Math.abs(k.cos) < 0.01);
	const exSentence = !ex.length
		? 'There are none: the remainder never does as well as the J-component.'
		: `The remainder does at least as well as the J-component for ${list(byDepth)}.` +
			(far.length ? ` For ${list(far.map((k) => q(S.vocab[k.c])))} the cosine is far from zero: the lens vector is not among the fit's vectors, and the arithmetic above does not apply.` : '') +
			(zero.length ? ` For ${list(zero.map((k) => q(S.vocab[k.c])))} it is zero, and the model has departed from its first-order picture.` : '');
	// the Shakespeare swap
	const sw = G.interventions.swap, one = sw.alphas.indexOf(1);
	const after = Array.from({ length: G.vocab.length }, (_, c) => at(sw.probabilities, c, one));
	const b0 = top(sw.baseline), b1 = top(after);
	// lengths of the lens vectors and of the stream's columns, per depth (steering strength is |α|·‖v_a‖)
	const D = S.model.config.D;
	const vnorm = (a, j) => Math.sqrt(Array.from({ length: D }, (_, q) => at(S.lenses.vectors, a, q, j) ** 2).reduce((s, x) => s + x, 0));
	const norms = (j) => Array.from({ length: V }, (_, a) => vnorm(a, j));
	const colMean = (j) => S.j_space[j].norm.reduce((s, x) => s + x, 0) / S.j_space[j].norm.length;
	const f1 = (x) => x.toFixed(1), jl = S.depth.length - 1;
	return {
		wrNormMin0: f1(Math.min(...norms(0))), wrNormMax0: f1(Math.max(...norms(0))), wrColMean0: f1(colMean(0)),
		wrNormMinL: f1(Math.min(...norms(jl))), wrNormMaxL: f1(Math.max(...norms(jl))), wrColMeanL: f1(colMean(jl)),
		steerOneZero: steer(0, 1), steerOneHalf: steer(1, 1), steerOneOne: steer(2, 1), steerFourOne: steer(2, A.at(-1)),
		steerAlphaMax: String(A.at(-1)),
		privilegedJWins: String(cases.filter((k) => k.J > k.R).length),
		privilegedCases: String(cases.length),
		privilegedOrthogonal: String(cases.filter((k) => Math.abs(k.cos) < 0.01).length),
		privilegedExceptions: exSentence,
		gremioDepth: depth(sw.depth),
		gremioPair: `${q(G.vocab[sw.pair[0] - 1])} and ${q(G.vocab[sw.pair[1] - 1])}`,
		gremioBefore: `${name(G.vocab[b0])}, with probability ${sw.baseline[b0].toFixed(2)}`,
		gremioAfter: `${name(G.vocab[b1])}, with probability ${after[b1].toFixed(2)}`
	};
}

// Chapter 6 (Is it a workspace?): the structural statistics of both exports.
function workspaceValues(read) {
	const S = read('jspace/artifacts/specimen-jlens.json'), G = read('jspace/artifacts/shakespeare-jlens.json');
	const at = (t, ...i) => { let o = 0, m = 1; i.forEach((k, d) => { o += k * m; m *= t.shape[d]; }); return t.data[o]; };
	const depth = (d) => (d % 1 ? (Math.floor(d) ? `${Math.floor(d)}½` : '½') : String(d));
	const list = (xs) => (xs.length < 2 ? xs.join('') : `${xs.slice(0, -1).join(', ')} and ${xs.at(-1)}`);
	const f2 = (x) => x.toFixed(2), n = G.depth.length, V = G.vocab.length, D = G.model.config.D;
	const cka = (i, j) => at(G.cka, i, j);
	let inner = 1;
	for (let i = 0; i < n - 1; i++) for (let j = 0; j < n - 1; j++) inner = Math.min(inner, cka(i, j));
	const last = G.depth.slice(0, -1).map((_, i) => cka(i, n - 1));
	const effDim = (j) => {
		const sv = G.jacobians.readout_singular_values, s2 = Array.from({ length: sv.shape[0] }, (_, r) => at(sv, r, j) ** 2);
		const total = s2.reduce((a, b) => a + b, 0);
		let acc = 0, k = 0;
		while (k < s2.length && acc < 0.9 * total) acc += s2[k++];
		return k;
	};
	const m = G.lenses.metrics, kj = m.map((q) => q.j_lens.mean_excess_kurtosis);
	const positive = G.depth.filter((_, j) => kj[j] > 0).map(depth);
	const gap = m.slice(0, -1).map((q) => 100 * (q.logit_lens.top1_accuracy - q.j_lens.top1_accuracy));
	const below = G.depth.filter((_, j) => G.j_space[j].span_fraction < G.j_space[j].random_span_fraction).map(depth);
	const rnd = G.j_space.map((q) => q.random_span_fraction), srnd = S.j_space.map((q) => q.random_span_fraction.reduce((a, b) => a + b, 0) / q.random_span_fraction.length);
	const mlp = (M, k) => list(M.structure.mlp.map((b) => f2(b.populations[k].median)));
	const heads = G.structure.attention_ov, pop = (h, k) => h.populations[k];
	const lp = (k) => heads.map((h) => pop(h, k).label_preservation), sh = S.structure.attention_ov;
	return {
		wsDepths: String(n), wsBlocks: String(G.model.config.L),
		wsCkaInnerMin: f2(inner), wsCkaLastMin: f2(Math.min(...last)), wsCkaLastMax: f2(Math.max(...last)),
		wsKurtosisPositive: positive.length ? `only at depth ${list(positive)}` : 'at no depth',
		wsAccGapMin: gap.length ? Math.min(...gap).toFixed(1) : '0', wsAccGapMax: gap.length ? Math.max(...gap).toFixed(1) : '0',
		wsEffDims: list(G.depth.map((_, j) => String(effDim(j)))), wsRank: String(Math.min(V, D)),
		wsOccupancy: list(G.j_space.map((q) => (q.occupancy == null ? 'none' : String(q.occupancy)))),
		wsSpanBelow: below.length === n ? 'every depth' : below.length ? `depth ${list(below)}` : 'no depth',
		wsSpanExpected: f2(Math.min(V, D) / D), wsSpanRandomMin: f2(Math.min(...rnd)), wsSpanRandomMax: f2(Math.max(...rnd)),
		wsSpecSpanExpected: f2((S.vocab.length) / S.model.config.D), wsSpecSpanRandom: list(srnd.map(f2)),
		wsSpanLast: f2(G.j_space[n - 1].span_fraction),
		wsMlpJ: mlp(G, 'j_lens'), wsMlpLogit: mlp(G, 'logit_lens'), wsMlpNeurons: list(G.structure.mlp.filter((b) => b.populations.previous_neurons).map((b) => f2(b.populations.previous_neurons.median))),
		wsSpecMlpJ: f2(S.structure.mlp[0].populations.j_lens.median), wsSpecMlpLogit: f2(S.structure.mlp[0].populations.logit_lens.median),
		wsHeads: String(heads.length),
		wsHeadJMin: f2(Math.min(...lp('j_lens'))), wsHeadJMax: f2(Math.max(...lp('j_lens'))),
		wsHeadRotMin: f2(Math.min(...lp('rotated_j_lens'))).replace('-', '−'), wsHeadRotMax: f2(Math.max(...lp('rotated_j_lens'))),
		wsHeadLogitMax: f2(Math.max(...lp('logit_lens'))),
		wsHeadPreserveWins: String(heads.filter((h) => pop(h, 'j_lens').label_preservation > pop(h, 'rotated_j_lens').label_preservation).length),
		wsHeadGainWins: String(heads.filter((h) => pop(h, 'j_lens').gain > pop(h, 'rotated_j_lens').gain).length),
		wsSpecHeadJ: list(sh.map((h) => f2(pop(h, 'j_lens').label_preservation).replace('-', '−'))),
		wsSpecHeadRot: list(sh.map((h) => f2(pop(h, 'rotated_j_lens').label_preservation).replace('-', '−')))
	};
}

// Chapters 3 (The Jacobian lens) and 4 (The J-space): readings of the J-lens exports.
function jlensValues(read) {
	const S = read('jspace/artifacts/specimen-jlens.json'), G = read('jspace/artifacts/shakespeare-jlens.json');
	const at = (t, ...i) => { let o = 0, m = 1; i.forEach((k, d) => { o += k * m; m *= t.shape[d]; }); return t.data[o]; };
	const depth = (d) => (d % 1 ? (Math.floor(d) ? `${Math.floor(d)}½` : '½') : String(d));
	const pct = (f) => `${Math.round(100 * f)}%`;
	const q = (ch) => `‘${ch === ' ' ? '␣' : ch === '\n' ? '¶' : ch}’`;
	const top = (p) => p.indexOf(Math.max(...p));
	const sci = (x) => { const [m, e] = x.toExponential(1).split('e'); return `$${m}\\times10^{${+e}}$`; };
	const V = S.vocab.length, N = S.lenses.model_probs.shape[1], D = S.model.config.D, J = S.depth.length;
	const col = (t, n, j) => Array.from({ length: V }, (_, a) => (j == null ? at(t, a, n) : at(t, a, n, j)));
	// chapter 3, specimen: top-1 agreement of each lens with the model, as counts of the N columns
	const agree = (t, j) => { let c = 0; for (let n = 0; n < N; n++) if (top(col(t, n, j)) === top(col(S.lenses.model_probs, n))) c++; return String(c); };
	// chapter 3, Shakespeare: the two conventions at the last depth, and the gap at intermediate depths
	const M = G.lenses.metrics, L = G.lenses.literal, T = G.model.config.T, eps = 1e-5;
	const gaps = M.slice(0, -1).map((m) => 100 * (m.logit_lens.top1_accuracy - m.j_lens.top1_accuracy));
	const j3 = G.depth.indexOf(3);
	// chapter 3, the prompt: letters of the speakers' names (capitals before a ':'), and how often each
	// lens reads ':' under them, at the depth where the J-lens does so most often
	const text = Array.from(G.lenses.prompt.text), colon = G.vocab.indexOf(':') + 1;
	const letters = [];
	text.forEach((c, t) => { if (/[A-Z]/.test(c)) { let u = t; while (/[A-Z]/.test(text[u] ?? '')) u++; if (text[u] === ':') letters.push({ t, ahead: u - t }); } });
	const colonAt = (j, lens) => letters.filter(({ t }) => at(G.lenses.prompt.ids, 0, t, j, lens) === colon).length;
	let jc = 0;
	for (let j = 0; j < G.depth.length - 1; j++) if (colonAt(j, 0) > colonAt(jc, 0)) jc = j;
	// chapter 3, specimen: lengths of the lens vectors against the readout rows at depth 0, and obtuse pairs
	const vecAt = (t, a, j) => Array.from({ length: D }, (_, i) => at(t, a, i, j));
	const len = (u) => Math.sqrt(u.reduce((s, x) => s + x * x, 0));
	const ratios = Array.from({ length: V }, (_, a) => len(vecAt(S.lenses.vectors, a, 0)) / len(vecAt(S.lenses.logit_vectors, a, 0)));
	const lenRatioMax = Math.max(...ratios).toFixed(1);
	const shorter = ratios.map((r, a) => (r < 1 ? q(S.vocab[a]) : null)).filter(Boolean);
	const obtuse = [];
	for (let j = 0; j < J - 1; j++) for (let a = 0; a < V; a++) { const u = vecAt(S.lenses.vectors, a, j), w = vecAt(S.lenses.logit_vectors, a, j); if (u.reduce((s, x, i) => s + x * w[i], 0) < 0) obtuse.push(`${q(S.vocab[a])} at depth ${depth(S.depth[j])}`); }
	const lenNote = (shorter.length ? `, though ${shorter.join(', ')} at depth 0 ${shorter.length > 1 ? 'are' : 'is'} shorter than ${shorter.length > 1 ? 'their' : 'its'} readout row` : '') + (obtuse.length ? `${shorter.length ? ', and' : ', though'} ${obtuse.join(', ')} ${obtuse.length > 1 ? 'make' : 'makes'} a slightly obtuse angle with ${obtuse.length > 1 ? 'theirs' : 'its own'}` : '');
	// chapter 4, specimen: mean missed share at k = V, J and random; spans; nesting of the best subsets
	const miss = (j, k) => { const s = S.j_space[j]; let x = 0, r = 0; for (let n = 0; n < N; n++) { x += at(s.distance, k - 1, n) ** 2 / s.norm[n] ** 2; r += at(s.random_fraction, k - 1, n); } return [x / N, r / N]; };
	const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
	let notNested = 0, cases = 0;
	for (let j = 0; j < J; j++) for (let n = 0; n < N; n++) for (let k = 1; k < V; k++) {
		const sel = (kk) => new Set(Array.from({ length: kk }, (_, s) => at(S.j_space[j].selected, s, kk - 1, n)));
		const a = sel(k), b = sel(k + 1);
		cases++;
		if (![...a].every((x) => b.has(x))) notNested++;
	}
	// chapter 4, the cone figure's default (window 1, position 4, depth ½, k = 2): its best pair
	const jh = S.depth.indexOf(0.5), n0 = 3;
	const pair = [0, 1].map((s) => S.vocab[at(S.j_space[jh].selected, s, 1, n0) - 1]);
	const vec = (a, j) => Array.from({ length: D }, (_, i) => at(S.lenses.vectors, a, i, j));
	const cos = (u, w) => { const d = (x, y) => x.reduce((s, xi, i) => s + xi * y[i], 0); return d(u, w) / Math.sqrt(d(u, u) * d(w, w)); };
	let opp = { c: 1 };
	for (let a = 0; a < V; a++) for (let b = a + 1; b < V; b++) { const c = cos(vec(a, jh), vec(b, jh)); if (c < opp.c) opp = { a, b, c }; }
	// chapter 4, Shakespeare
	const Gs = G.j_space, last = Gs.length - 1, k25 = Gs[0].k.indexOf(25);
	let occEarly = 0;
	for (let j = 1; j < last; j++) if ((Gs[j].occupancy ?? 0) > (Gs[occEarly].occupancy ?? 0)) occEarly = j;
	const spans = Gs.map((g) => g.span_fraction), jMin = spans.indexOf(Math.min(...spans));
	return {
		jlSpecColumns: String(N),
		jlSpecAgreeZeroJ: agree(S.lenses.j_probs, 0), jlSpecAgreeZeroLogit: agree(S.lenses.logit_probs, 0),
		jlSpecAgreeHalfJ: agree(S.lenses.j_probs, jh), jlSpecAgreeHalfLogit: agree(S.lenses.logit_probs, jh),
		jlSpecLensesAgreeZero: pct(S.lenses.j_logit_agreement[0]), jlSpecLensesAgreeHalf: pct(S.lenses.j_logit_agreement[jh]),
		jlLitErrSpecimen: sci(S.lenses.literal.final_max_probability_error),
		jlLitErrShakespeare: L.final_max_probability_error.toFixed(2),
		jlLitAgreeLast: `${(100 * L.metrics.at(-1).j_lens.model_agreement).toFixed(1)}%`,
		jlEpsEffShakespeare: (eps * ((T + 1) / 2) ** 2).toFixed(2),
		jlShAgreeJThree: pct(M[j3].j_lens.model_agreement), jlShAgreeLogitThree: pct(M[j3].logit_lens.model_agreement),
		jlShGapMin: Math.min(...gaps).toFixed(1), jlShGapMax: Math.max(...gaps).toFixed(1),
		jlShPositions: M[0].j_lens.positions.toLocaleString('en-GB'),
		jlNameLetters: String(letters.length), jlColonAheadMax: String(Math.max(...letters.map((l) => l.ahead))),
		jlColonDepth: depth(G.depth[jc]), jlColonJ: String(colonAt(jc, 0)), jlColonLogit: String(colonAt(jc, 1)),
		jlColonTable: `at depths $0,\\tfrac12,\\dots,4$ the J-lens reads it under $${G.depth.map((_, j) => colonAt(j, 0)).join(',')}$ of the letters and the logit lens under $${G.depth.map((_, j) => colonAt(j, 1)).join(',')}$`,
		jlLenRatioMax: lenRatioMax, jlLenNote: lenNote,
		jsSpecMissLastJ: pct(miss(J - 1, V)[0]), jsSpecMissLastRandom: pct(miss(J - 1, V)[1]),
		jsSpecMissZeroJ: pct(miss(0, V)[0]), jsSpecMissZeroRandom: pct(miss(0, V)[1]),
		jsSpecMissHalfJ: pct(miss(jh, V)[0]), jsSpecMissHalfRandom: pct(miss(jh, V)[1]),
		jsSpecSpanZero: pct(mean(S.j_space[0].span_fraction)), jsSpecSpanLast: pct(mean(S.j_space[J - 1].span_fraction)),
		jsSpecCapturedZero: pct(1 - miss(0, V)[0]),
		jsNotNested: String(notNested), jsNestCases: String(cases),
		jsSpecTopOne: String(Array.from({ length: N }, (_, n) => n).filter((n) => at(S.j_space[J - 1].selected, 0, 0, n) - 1 === top(col(S.lenses.model_probs, n))).length),
		jsConePair: `${q(pair[0])} and ${q(pair[1])}`,
		jsConeMiss: pct(at(S.j_space[jh].distance, 1, n0) ** 2 / S.j_space[jh].norm[n0] ** 2),
		jsOppositePair: `${q(S.vocab[opp.a])} and ${q(S.vocab[opp.b])}`, jsOppositeCos: `${opp.c < 0 ? "−" : ""}${Math.abs(opp.c).toFixed(2)}`,
		jsShOccLast: String(Gs[last].occupancy), jsShOccEarly: String(Gs[occEarly].occupancy), jsShOccEarlyDepth: depth(G.depth[occEarly]),
		jsShTwentyFiveJ: pct(1 - Gs[last].residual_fraction[k25]), jsShTwentyFiveRandom: pct(1 - Gs[last].random_residual_fraction[k25]),
		jsShSpanLast: pct(Gs[last].span_fraction), jsShSpanRandomLast: pct(Gs[last].random_span_fraction),
		jsShSpanMin: pct(spans[jMin]), jsShSpanMinDepth: depth(G.depth[jMin])
	};
}

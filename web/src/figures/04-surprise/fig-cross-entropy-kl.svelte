<script module>
	export const meta = { size: 'wide', title: 'The loss as entropy plus divergence' };
</script>

<script>
	// Figure: the loss, regrouped by context. A prediction's context is the window prefix the
	// model sees; equal contexts get equal predictions q_c. For each context c with n_c
	// predictions, p̂_c is the distribution of what actually followed, and
	//     L = Σ_c (n_c/N) H(p̂_c, q_c) = Σ_c (n_c/N) H(p̂_c) + Σ_c (n_c/N) KL(p̂_c ‖ q_c).
	// Rows: every context the corpus leaves open (p̂ outlined, q solid, per character), then
	// all the contexts it decides, together; last, the total: floor + reducible part.
	// Everything is computed from the corpus, its windows and the model. Scrub the training.
	import { getContext, onMount } from 'svelte';
	import { measure } from '../primitives/measure.js';
	import Scrubber from '../primitives/Scrubber.svelte';
	import { fmt, fmtProb, showChar, layoutWidth } from '../primitives/scales.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const V = $derived(specimen.config.V);
	const T = $derived(specimen.config.T);
	const vocab = $derived(specimen.vocab);
	const lnV = $derived(Math.log(V));
	const interactive = $derived(mode !== 'print');
	const steps = $derived(specimen.provenance?.steps ?? 0);
	let width = $state(868);

	// training clock (as in fig-surprise)
	// svelte-ignore state_referenced_locally
	let t = $state(attrs.t != null ? +attrs.t : 1);
	$effect(() => { if (attrs.t != null) t = +attrs.t; });
	const stepAt = (u) => Math.round((steps + 1) ** u - 1);
	const atStep = (s) => Math.log(s + 1) / Math.log(steps + 1);
	let traj = $state(null);
	let tried = false;
	function ensure() { if (!tried) { tried = true; traj = specimen.trajectory(); } }
	onMount(() => { if (interactive) (window.requestIdleCallback ?? ((f) => setTimeout(f, 400)))(ensure); });
	const step = $derived(stepAt(t));
	$effect(() => { if (interactive && step > 0 && step < steps) ensure(); });
	const theta = $derived(step >= steps ? specimen.theta : step === 0 ? specimen.theta0 : traj ? traj.thetas[step] : t < 0.5 ? specimen.theta0 : specimen.theta);
	const shownStep = $derived(step === 0 || step >= steps || traj ? step : t < 0.5 ? 0 : steps);

	const H = (p) => -p.reduce((s, x) => s + (x > 0 ? x * Math.log(x) : 0), 0);
	const KL = (p, q) => p.reduce((s, x, a) => s + (x > 0 ? x * Math.log(x / q[a]) : 0), 0);

	// contexts: computed from the corpus and the training windows (θ-independent)
	const groups = $derived.by(() => {
		const pr = specimen.predictions();
		const m = new Map();
		for (let b = 0; b < pr.B; b++)
			for (let k = 0; k < T; k++) {
				const ctx = specimen.corpus.slice(b, b + k + 1);
				const y = pr.y[b][k] - 1;
				if (!m.has(ctx)) m.set(ctx, { ctx, n: 0, counts: new Array(V).fill(0), cols: [], wins: [] });
				const g = m.get(ctx);
				g.n++; g.counts[y]++; g.cols.push(b * T + k); g.wins.push({ b, k, y });
			}
		return [...m.values()].map((g) => ({ ...g, p: g.counts.map((c) => c / g.n), open: g.counts.filter((c) => c > 0).length > 1 }));
	});
	const N = $derived(groups.reduce((s, g) => s + g.n, 0));
	const open = $derived(groups.filter((g) => g.open));
	const decided = $derived(groups.filter((g) => !g.open));
	const nDecided = $derived(decided.reduce((s, g) => s + g.n, 0));

	// the floor, as a formula: contexts whose followers are uniform on m characters give n·ln m
	const floorTerms = $derived.by(() => {
		const byM = new Map();
		let exact = true;
		for (const g of open) {
			const nz = g.counts.filter((c) => c > 0);
			if (nz.some((c) => c !== nz[0])) exact = false;
			const mm = nz.length;
			byM.set(mm, (byM.get(mm) ?? 0) + g.n);
		}
		return { exact, terms: [...byM.entries()].sort((a, b) => a[0] - b[0]) };
	});
	const floor = $derived(groups.reduce((s, g) => s + g.n * H(g.p), 0) / N);

	// the model's side, at θ
	const model = $derived.by(() => {
		const pr = specimen.predictions(theta);
		const rows = groups.map((g) => {
			const q = pr.probs[g.cols[0]];
			return { q, H: H(g.p), KL: KL(g.p, q) };
		});
		return { rows, loss: pr.loss };
	});
	const rowOf = (g) => model.rows[groups.indexOf(g)];
	const klDecided = $derived(decided.reduce((s, g) => s + g.n * rowOf(g).KL, 0));
	const klOpen = $derived(open.reduce((s, g) => s + g.n * rowOf(g).KL, 0));
	const klPart = $derived((klDecided + klOpen) / N);

	// geometry
	const W = $derived(layoutWidth(width, mode, 340));
	const narrow = $derived(W < 700);
	const colA = 0;
	const colB = $derived(narrow ? 0 : 168);
	const slot = $derived(narrow ? Math.min(40, (W - 20) / V) : 38);
	const colC = $derived(narrow ? 0 : colB + slot * V + 34);
	const barMax = $derived(narrow ? W - 150 : W - colC - 120); // px for ln V nats
	const nats = (v) => (v / lnV) * barMax;
	const distH = 44;
	const rowOpenH = $derived(narrow ? 190 : 108);
	const top = 34;
	const yOpen = (i) => top + i * rowOpenH;
	const yDec = $derived(top + open.length * rowOpenH + 6);
	const yTot = $derived(yDec + (narrow ? 96 : 70));
	const Hsvg = $derived(yTot + 82);
</script>

{#snippet split(x, y, h, kl, labels = true)}
	<rect class="fx-ce__h" x={x} y={y - 7} width={Math.max(0, nats(h))} height="14" />
	<rect class="bar--value" x={x + nats(h)} y={y - 7} width={Math.max(0, nats(kl))} height="14" />
	{#if labels}
		<text class="lbl lbl--micro" x={x + nats(h) + nats(kl) + 6} y={y + 4}><tspan class="m">H</tspan> {fmt(h, 3)} + KL {fmt(kl, 3)}</text>
	{/if}
{/snippet}

<div class="fx fx-ce" use:measure={(w) => (width = w)}>
	<svg viewBox="0 0 {W} {Hsvg}" width={W} height={Hsvg} role="img" aria-label="Loss {fmt(model.loss, 4)} = floor {fmt(floor, 4)} + divergence {fmt(klPart, 4)} at step {shownStep}">
		<!-- column heads -->
		{#if !narrow}
			<text class="lbl lbl--micro" x={colA} y={top - 18}>context, and the windows it opens</text>
			<text class="lbl lbl--micro" x={colB} y={top - 18}>what followed (outline) · what the model predicts (solid)</text>
			<text class="lbl lbl--micro" x={colC} y={top - 18}>cross-entropy per prediction = entropy + divergence, in nats</text>
		{/if}

		{#each open as g, i (g.ctx)}
			{@const y = yOpen(i)}
			{@const r = rowOf(g)}
			{@const yb = narrow ? y + 34 + g.wins.length * 15 + distH + 6 : y + distH + 10}
			{@const ys = narrow ? yb + 40 : y + 30}
			<g>
				<text class="lbl lbl--head" x={colA} y={y + 14}>after ‘<tspan class="fx-ce__tok">{Array.from(g.ctx).map(showChar).join('')}</tspan>’ <tspan class="fx-ce__n">× {g.n}</tspan></text>
				{#each g.wins as w, k (k)}
					<text class="fx-ce__win" x={colA} y={y + 34 + k * 15}>{#each Array.from(specimen.corpus.slice(w.b, w.b + T)) as ch, j (j)}<tspan class:is-ctx={j <= w.k} class:is-y={j === w.k + 1}>{showChar(ch)}</tspan>{/each}<tspan class="fx-ce__winl" dx="8">window {w.b + 1}</tspan></text>
				{/each}
				<!-- distributions over the vocabulary -->
				{#each vocab as ch, a (a)}
					{@const x = colB + slot * (a + 0.5)}
					<rect class="fx-ce__data" x={x - 9} y={yb - g.p[a] * distH} width="18" height={g.p[a] * distH} />
					<rect class="bar--value" x={x - 4} y={yb - r.q[a] * distH} width="8" height={r.q[a] * distH} />
					<text class="tok fx-ce__vch" x={x} y={yb + 16} text-anchor="middle">{showChar(ch)}</text>
					{#if g.p[a] > 0}<text class="lbl lbl--micro" x={x} y={yb - distH - 5} text-anchor="middle">{fmtProb(r.q[a], 2)}</text>{/if}
				{/each}
				<line class="axis" x1={colB} x2={colB + slot * V} y1={yb} y2={yb} />
				<!-- H + KL -->
				{@render split(narrow ? 0 : colC, ys, r.H, r.KL)}
				<text class="lbl lbl--micro" x={narrow ? 0 : colC} y={ys + 24}>entropy of what followed: {floorTerms.exact ? `ln ${g.counts.filter((c) => c > 0).length}` : fmt(r.H, 3)}</text>
			</g>
		{/each}

		<!-- the decided contexts, together -->
		<g>
			<text class="lbl lbl--head" x={colA} y={yDec + 14}>every other context <tspan class="fx-ce__n">× {nDecided}</tspan></text>
			<text class="lbl lbl--micro" x={colA} y={yDec + 30}>{decided.length} contexts, each followed by</text>
			<text class="lbl lbl--micro" x={colA} y={yDec + 44}>one character only: entropy 0</text>
			{@render split(narrow ? 0 : colC, narrow ? yDec + 66 : yDec + 26, 0, klDecided / nDecided)}
			{#if !narrow}<text class="lbl lbl--micro" x={colB} y={yDec + 30}>the model should give that character 1;</text>
			<text class="lbl lbl--micro" x={colB} y={yDec + 44}>whatever it withholds is divergence</text>{/if}
		</g>

		<!-- the total -->
		<g>
			<line class="rule" x1={colA} x2={W} y1={yTot - 8} y2={yTot - 8} />
			<text class="lbl lbl--head" x={colA} y={yTot + 14}>the loss, step {shownStep}</text>
			<text class="lbl lbl--micro" x={colA} y={yTot + 30}>the mean over all {N} predictions</text>
			{@render split(narrow ? 0 : colC, narrow ? yTot + 50 : yTot + 10, floor, klPart, false)}
			<text class="lbl" x={narrow ? 0 : colC} y={narrow ? yTot + 76 : yTot + 38}><tspan class="fx-ce__strong">{fmt(model.loss, 4)}</tspan> = {fmt(floor, 4)} + {fmt(klPart, 4)}</text>
			<text class="lbl lbl--micro" x={narrow ? 0 : colC} y={narrow ? yTot + 92 : yTot + 56}>the floor, {#if floorTerms.exact}({floorTerms.terms.map(([m, c]) => `${c} ln ${m}`).join(' + ')})/{N}{:else}Σ n H / N{/if}, cannot be removed by training</text>
		</g>
	</svg>
	{#if interactive}
		<Scrubber bind:t phases={(narrow ? [0, 10] : [0, 1, 10, 100]).filter((x) => x < steps).map((x) => ({ at: atStep(x), label: x === 0 ? 'untrained' : `step ${x}` })).concat([{ at: 1, label: `trained (${steps})` }])} label="training step" duration={4200} />
		<p class="readout" aria-live="polite">
			<span class="readout__who">Step {shownStep}.</span>
			<span class="readout__part">Loss {fmt(model.loss, 4)} nats: the floor {fmt(floor, 4)}, which depends only on the text, plus a divergence of {fmt(klPart, 4)}, of which {fmt(klOpen / N, 4)} comes from the open contexts and {fmt(klDecided / N, 4)} from the decided ones.</span>
			<span class="readout__hint">Drag along the line to replay the training. The outlined bars never move: they are the text.</span>
		</p>
	{/if}
</div>

<style>
	.fx-ce :global(.m) { font-family: var(--serif); font-style: italic; fill: var(--ink); font-size: 13px; }
	.fx-ce__tok { font-family: var(--serif); font-size: 15px; fill: var(--ink); }
	.fx-ce__n { font-weight: 400; fill: var(--ink-2); }
	.fx-ce__win { font-family: var(--serif); font-size: 13px; fill: var(--ink-3); }
	.fx-ce__win .is-ctx { fill: var(--ink); font-weight: 600; }
	.fx-ce__win .is-y { fill: var(--value-deep); font-weight: 600; }
	.fx-ce__winl { font-family: var(--sans); font-size: 10.5px; fill: var(--ink-3); }
	.fx-ce__data { fill: none; stroke: var(--ink); stroke-width: 1px; }
	.fx-ce__vch { font-size: 14px; fill: var(--ink-2); }
	.fx-ce__h { fill: none; stroke: var(--ink); stroke-width: 1px; }
	.fx-ce__strong { fill: var(--ink); font-weight: 600; }
</style>

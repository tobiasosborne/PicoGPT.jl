<script module>
	export const meta = { size: 'wide', title: 'Steering one column along a J-lens vector' };
</script>

<script>
	// Figure: what does adding α v_a to one column do to the model's output there?
	// For one column x of the specimen (window, position) and one character a, each depth
	// ℓ ∈ {0, ½, 1} is a row of small multiples: the model's output distribution at that
	// position after x ← x + α v_a, for α = −4…4 (the export's grid), and after the ablation
	// x ← x − ⟨x, v̂_a⟩ v̂_a. The bar of a is in full ink, the others faded; a line joins the
	// tops of a's bars, so each row also draws p_a(α). The ablation is the point α* =
	// −⟨x, v_a⟩/‖v_a‖² of the same line, marked on each row's axis.
	// Numbers: J-lens export (interventions.by_depth[j].steer / .ablate, lenses.vectors) and
	// the stream x from the specimen engine. Interactive: choose the column on the corpus line,
	// the character a, and α on the track (or click a column of multiples).
	import { getContext } from 'svelte';
	import { measure } from '#figures/primitives/measure.js';
	import CorpusStrip from '#figures/primitives/CorpusStrip.svelte';
	import TickTrack from '../primitives/TickTrack.svelte';
	import CharRow from './CharRow.svelte';
	import { fmt, fmtProb, showChar, clamp, layoutWidth } from '#figures/primitives/scales.js';
	import { at, tableProbs, lensVector, depthLabel, dot } from './jdata.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const J = $derived(getContext('artifacts')()?.jspace?.specimen ?? null);
	const interactive = $derived(mode !== 'print');
	let width = $state(868);

	const T = $derived(specimen.config.T);
	const V = $derived(specimen.config.V);
	const vocab = $derived(specimen.vocab);
	const nWin = $derived(J?.sampling.windows ?? 10);

	// svelte-ignore state_referenced_locally
	let start = $state(clamp((+attrs.window || 1) - 1, 0, 9));
	// svelte-ignore state_referenced_locally
	let pos = $state(clamp((+attrs.position || 4) - 1, 0, 7));
	// svelte-ignore state_referenced_locally
	let a = $state(Math.max(0, specimen.vocab.indexOf(attrs.char ?? 'o')));
	// svelte-ignore state_referenced_locally
	let alpha = $state(attrs.alpha != null ? +attrs.alpha : 2);

	const alphas = $derived(J?.interventions.alphas ?? []);
	const depths = $derived(J?.depth ?? []);
	const n = $derived(start * T + pos);
	const tr = $derived(specimen.windowTrace(start));
	const streams = $derived([tr.X0, tr.Xmid, tr.Xout]);
	const KIND = ['after the embedding', 'after attention', 'after the MLP'];

	// per depth: the distributions along the steering line, the ablation, the probe
	const rows = $derived.by(() => {
		if (!J) return [];
		return depths.map((d, j) => {
			const D = J.interventions.by_depth[j];
			const probs = alphas.map((_, i) => tableProbs(J, at(D.steer, n, a, i)));
			const ablated = tableProbs(J, at(D.ablate, 0, n, a));
			const v = lensVector(J, a, j);
			const x = streams[j][pos];
			const probe = dot(v, x), vv = dot(v, v);
			return { d, j, probs, ablated, probe, vnorm: Math.sqrt(vv), astar: -probe / vv };
		});
	});
	// the saturation limit: for |α| → ∞ the column is α v_a + O(1), the final LayerNorm sees v_a
	// alone, and the output tends to softmax(Wout LN(v_a) + bout) (ε → 0 in the limit)
	const P = $derived(Object.fromEntries(specimen.params().map((q) => [q.name, q.values])));
	function readBare(v) {
		const Dn = v.length, m = v.reduce((x, y) => x + y, 0) / Dn;
		const c = v.map((x) => x - m), sd = Math.sqrt(c.reduce((t, x) => t + x * x, 0) / Dn);
		const h = c.map((x, q) => P.gammaf[q] * (x / sd) + P.betaf[q]);
		const z = Array.from({ length: V }, (_, b) => P.bout[b] + h.reduce((t, x, q) => t + P.Wout[b + q * V] * x, 0));
		const M = Math.max(...z), e = z.map((x) => Math.exp(x - M)), S = e.reduce((x, y) => x + y, 0);
		return e.map((x) => x / S);
	}
	const limits = $derived(J ? depths.map((_, j) => readBare(lensVector(J, a, j))) : []);
	const y = $derived(specimen.lab.train.y[start][pos] - 1);
	const ctx = $derived(Array.from(specimen.corpus.slice(start, start + pos + 1)).map(showChar).join(''));

	// geometry
	const W = $derived(layoutWidth(width, mode, 340));
	const narrow = $derived(W < 620);
	const shown = $derived(narrow ? alphas.map((al, i) => ({ al, i })).filter((o) => o.al % 2 === 0) : alphas.map((al, i) => ({ al, i })));
	const padL = $derived(narrow ? 30 : 104);
	const padR = 8;
	const gap = $derived(narrow ? 14 : 22);
	const slots = $derived(shown.length + 1);
	const mw = $derived((W - padL - padR - gap) / slots);
	const xm = (k) => padL + mw * k + (k >= shown.length ? gap : 0); // left edge of multiple k
	const bw = $derived((mw * 0.78) / V);
	const barX = (k, c) => xm(k) + mw * 0.11 + bw * c;
	const hBar = $derived(narrow ? 40 : 52);
	const top = $derived(interactive ? (narrow ? 104 : 74) : 46);
	const rowH = $derived(hBar + (narrow ? 44 : 46));
	const base = (j) => top + 16 + j * rowH + hBar; // baseline of row j
	const H = $derived(base(2) + (interactive ? 78 : 46));

	// x of a value α on a row's axis: interpolate between the multiples' centres
	const centre = (k) => xm(k) + mw / 2;
	function xAlpha(al) {
		const s = shown, lo = s[0].al, hi = s[s.length - 1].al;
		if (al <= lo) return centre(0);
		if (al >= hi) return centre(s.length - 1);
		for (let k = 0; k < s.length - 1; k++) if (al <= s[k + 1].al) return centre(k) + ((al - s[k].al) / (s[k + 1].al - s[k].al)) * (centre(k + 1) - centre(k));
		return centre(s.length - 1);
	}
	const pLabel = (p) => { const f = fmtProb(p, 2); return f.startsWith('≈') ? `p ${f}` : `p = ${f}`; };
	const alphaIndex = $derived(Math.max(0, alphas.indexOf(alpha)));
	const shownK = $derived(shown.findIndex((o) => o.al === alpha));

	const readout = $derived.by(() => {
		if (!rows.length) return null;
		const ch = `‘${showChar(vocab[a])}’`;
		const parts = rows.map((r) => {
			const p = r.probs[alphaIndex][a];
			const outside = r.astar < alphas[0] || r.astar > alphas.at(-1);
			return `depth ${depthLabel(r.d)}: probe ⟨v, x⟩ = ${fmt(r.probe, 2)}, p(${ch}) = ${fmtProb(p, 2, true)} at α = ${fmt(alpha, 0)}, tending to ${fmtProb(limits[r.j][a], 2, true)} as α grows; ablation (α* = ${fmt(r.astar, 2)}${outside ? ', off the grid' : ''}) gives ${fmtProb(r.ablated[a], 2, true)}`;
		});
		return { who: `‘${ctx}’, window ${start + 1}, position ${pos + 1} (next: ‘${showChar(vocab[y])}’); steering towards ${ch}.`, parts };
	});
</script>

{#snippet multiple(p, k, j, faded = false)}
	{@const b0 = base(j)}
	<g class="steer__mult" class:is-faded={faded}>
		<line class="rule" x1={xm(k) + mw * 0.06} x2={xm(k) + mw * 0.94} y1={b0} y2={b0} />
		{#each p as q, c (c)}
			<rect class="bar--value steer__bar" class:is-a={c === a} x={barX(k, c) + bw * 0.12} y={b0 - q * hBar} width={bw * 0.76} height={Math.max(0, q * hBar)} />
		{/each}
	</g>
{/snippet}

<div class="fx fx-steer" use:measure={(w) => (width = w)}>
	{#if !J}
		<p class="fig__pending">The J-lens export (jspace/artifacts/specimen-jlens.json) is missing.</p>
	{:else}
		<svg viewBox="0 0 {W} {H}" width={W} height={H} role={interactive ? 'group' : 'img'} aria-label="Steering the column ‘{ctx}’ towards ‘{showChar(vocab[a])}’: {readout?.parts.join('; ')}">
			{#if interactive}
				<CorpusStrip corpus={specimen.corpus} {T} bind:start bind:pos maxStart={nWin - 1} x={narrow ? 8 : padL - 8} y={30} width={narrow ? W - 16 : Math.min(330, W * 0.4)} showTarget interactive label="edited column" />
				<CharRow {vocab} bind:value={a} x={narrow ? 112 : W - padR - 22 * V} y={narrow ? 78 : 34} step={22} name="steer towards" label="the character a" />
			{:else}
				<text class="lbl" x={padL} y={18}><tspan class="steer__tok">‘{ctx}’</tspan> (window {start + 1}, position {pos + 1}; next ‘{showChar(vocab[y])}’), steered towards <tspan class="steer__tok steer__strong">‘{showChar(vocab[a])}’</tspan></text>
			{/if}

			<!-- the selected α: a wash across the rows (selection only) -->
			{#if interactive && shownK >= 0}
				<rect class="steer__wash" x={xm(shownK) + 1} y={top - 2} width={mw - 2} height={base(2) - top + 22} />
			{/if}

			<!-- α labels over the first row -->
			{#each shown as o, k (k)}
				<text class="lbl lbl--micro steer__alpha" class:is-on={interactive && o.al === alpha} x={centre(k)} y={top + 2} text-anchor="middle">{o.al === 0 ? 'α = 0' : fmt(o.al, 0)}</text>
			{/each}
			<text class="lbl lbl--micro" x={xm(shown.length) + mw / 2} y={top + 2} text-anchor="middle">ablated</text>
			<text class="lbl lbl--micro" x={xm(shown.length) - gap / 2} y={top + 2} text-anchor="middle">∞</text>

			{#each rows as r, j (j)}
				{@const b0 = base(j)}
				<!-- depth label -->
				{#if narrow}
					<text class="lbl lbl--head" x={4} y={b0 - hBar / 2 + 4}>{depthLabel(r.d)}</text>
				{:else}
					<text class="lbl lbl--head" x={padL - 14} y={b0 - hBar + 12} text-anchor="end">depth {depthLabel(r.d)}</text>
					<text class="lbl lbl--micro" x={padL - 14} y={b0 - hBar + 26} text-anchor="end">{KIND[j]}</text>
					<text class="lbl lbl--micro" x={padL - 14} y={b0 - hBar + 40} text-anchor="end">⟨v, x⟩ = {fmt(r.probe, 2)}</text>
				{/if}
				{#each shown as o, k (k)}
					{@render multiple(r.probs[o.i], k, j)}
				{/each}
				<!-- p_a(α): the tops of a's bars, joined -->
				<polyline class="steer__curve" points={shown.map((o, k) => `${barX(k, a) + bw / 2},${b0 - r.probs[o.i][a] * hBar}`).join(' ')} />
				{#each shown as o, k (k)}
					<circle class="dot--value" cx={barX(k, a) + bw / 2} cy={b0 - r.probs[o.i][a] * hBar} r="2" />
				{/each}
				<!-- α → ∞: the output the lens vector produces alone -->
				{@const pinf = limits[j][a]}
				{@const xinf = xm(shown.length) - gap / 2}
				<line class="steer__limit" x1={barX(shown.length - 1, a) + bw / 2} y1={b0 - r.probs[shown.at(-1).i][a] * hBar} x2={xinf} y2={b0 - pinf * hBar} />
				<circle class="dot--open" cx={xinf} cy={b0 - pinf * hBar} r="2.6" />
				<!-- the ablation -->
				{@render multiple(r.ablated, shown.length, j)}
				{@const inside = r.astar >= shown[0].al && r.astar <= shown.at(-1).al}
				{@const xa = xAlpha(r.astar)}
				<line class="steer__tick" x1={xa} x2={xa} y1={b0 + 2} y2={b0 + 9} />
				<text class="lbl lbl--micro steer__astar" x={inside ? xa : r.astar < 0 ? xa - 4 : xa + 4} y={b0 + 20} text-anchor={inside ? 'middle' : r.astar < 0 ? 'start' : 'end'}>{inside ? `α* ${fmt(r.astar, 2)}` : r.astar < 0 ? `← α* = ${fmt(r.astar, 1)}` : `α* = ${fmt(r.astar, 1)} →`}</text>
				<!-- the probability of a, at α and ablated, as a number -->
				<text class="lbl lbl--micro lbl--value" x={xm(shown.length) + mw * 0.5} y={b0 + 20} text-anchor="middle">{pLabel(r.ablated[a])}</text>
			{/each}

			<!-- characters under the last row -->
			{#each shown.concat([{ al: null }]) as _, k (k)}
				{#each vocab as ch, c (c)}
					{#if !narrow || c === a}
						<text class="tok steer__ch" class:is-a={c === a} x={barX(k, c) + bw / 2} y={base(2) + 40} text-anchor="middle">{showChar(ch)}</text>
					{/if}
				{/each}
			{/each}

			{#if interactive}
				<TickTrack options={alphas.map((al) => ({ value: al, label: fmt(al, 0) }))} bind:value={alpha} xs={alphas.map((al) => xAlpha(al))} y={H - 22} name="α" />
			{/if}
		</svg>
		{#if interactive && readout}
			<p class="readout" aria-live="polite">
				<span class="readout__who">{readout.who}</span>
				{#each readout.parts as part, k (k)}<span class="readout__part">{part}{k < readout.parts.length - 1 ? ';' : '.'}</span>{/each}
				<span class="readout__hint">Click a character of the line to choose the edited column; choose the character a; drag α along the track.</span>
			</p>
		{/if}
	{/if}
</div>

<style>
	.fx-steer :global(.steer__bar) { opacity: 0.28; }
	.fx-steer :global(.steer__bar.is-a) { opacity: 1; }
	.steer__curve { fill: none; stroke: var(--value); stroke-width: var(--line); }
	.steer__tick { stroke: var(--ink); stroke-width: 1px; }
	.steer__limit { stroke: var(--value); stroke-width: var(--hair); stroke-dasharray: 2 2; }
	.steer__astar { fill: var(--ink-2); }
	.steer__wash { fill: var(--value-tint); opacity: 0.55; }
	.steer__alpha.is-on { fill: var(--ink); font-weight: 650; }
	.steer__ch { font-size: 11.5px; fill: var(--ink-3); }
	.steer__ch.is-a { fill: var(--ink); font-weight: 650; }
	.steer__tok { font-family: var(--serif); font-size: 14px; fill: var(--ink); }
	.steer__strong { font-weight: 650; }
</style>

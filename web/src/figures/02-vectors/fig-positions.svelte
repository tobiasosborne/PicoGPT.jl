<script module>
	export const meta = { size: 'wide', title: 'How is position written down?' };
</script>

<script>
	// Figure: the positional code P ∈ ℝ^{D×T}. One band per row i: the defining function
	// sin(ω_r τ) or cos(ω_r τ) as a hairline over continuous τ, and its samples at the T
	// positions as stems from the zero line (these are the numbers of P, from the adapter).
	// Right: P as a Hinton matrix, row for row; far right: the selected column P_t as a
	// spoke glyph. Pointing at a position selects its column everywhere.
	import { getContext } from 'svelte';
	import { measure } from '../primitives/measure.js';
		import SpokeGlyph from '../primitives/SpokeGlyph.svelte';
	import { fmt, clamp, layoutWidth } from '../primitives/scales.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const D = $derived(specimen.config.D);
	const T = $derived(specimen.config.T);
	const P = $derived(specimen.P); // P[t][i], t 0-based
	// svelte-ignore state_referenced_locally
	let sel = $state(clamp((+attrs.pos || 4) - 1, 0, specimen.config.T - 1));
	let width = $state(868);
	const interactive = $derived(mode !== 'print');

	// the defining frequencies (rows 2r+1, 2r+2 share ω_r); labels and the continuous curves
	const omega = (r) => Math.pow(10000, (-2 * r) / D);
	const fmtOmega = (w) => String(+w.toPrecision(3));

	// geometry
	const W = $derived(layoutWidth(width, mode, 330));
	const narrow = $derived(W < 640);
	const band = $derived(narrow ? 30 : 34);
	const top = 30;
	const xL = $derived(narrow ? 58 : 132); // curve area left
	const hc = $derived(band * 0.8); // Hinton pitch: one row per band, squares up to 0.68·band
	const mxW = $derived(narrow ? 0 : hc * T);
	const rG = $derived(narrow ? 0 : 58);
	const xR = $derived(narrow ? W - 10 : W - mxW - 2 * rG - 90); // curve area right
	const mx = $derived(xR + 34);
	const gx = $derived(W - rG - 14);
	const span = $derived(xR - xL);
	const xt = (t) => xL + (span * t) / (T - 1); // τ = t, positions 0…T−1
	const amp = $derived(band * 0.38);
	const yRow = (i) => top + band * (i + 0.5);
	const H = $derived(top + band * D + 40);

	function curve(i) {
		const r = Math.floor(i / 2), w = omega(r), f = i % 2 === 0 ? Math.sin : Math.cos;
		const n = 120, pts = [];
		for (let k = 0; k <= n; k++) {
			const tau = ((T - 1) * k) / n;
			pts.push(`${(xt(tau)).toFixed(2)},${(yRow(i) - amp * f(w * tau)).toFixed(2)}`);
		}
		return 'M' + pts.join('L');
	}
	function key(e) {
		if (!interactive) return;
		if (e.key === 'ArrowLeft') { sel = Math.max(0, sel - 1); e.preventDefault(); }
		if (e.key === 'ArrowRight') { sel = Math.min(T - 1, sel + 1); e.preventDefault(); }
	}
	const readout = $derived(`(${P[sel].map((v) => fmt(v, 2)).join(', ')}); its length is ${fmt(Math.hypot(...P[sel]), 2)}, as at every position`);
</script>

<div class="fx fx-pos" use:measure={(w) => (width = w)}>
	<!-- svelte-ignore a11y_no_noninteractive_tabindex (composite figure supports arrow-key navigation) -->
	<svg
		viewBox="0 0 {W} {H}"
		width={W}
		height={H}
		role={interactive ? 'group' : 'img'}
		tabindex={interactive ? 0 : undefined}
		aria-label="The positional code, position {sel + 1}: {readout}"
		onkeydown={key}
	>
		<!-- the selected position, through every band -->
		<rect class="fx-pos__sel" x={xt(sel) - span / (T - 1) / 2} y={top - 6} width={span / (T - 1)} height={band * D + 12} />
		{#each Array.from({ length: D }) as _, i (i)}
			{@const y = yRow(i)}
			{@const r = Math.floor(i / 2)}
			<line class="rule" x1={xL - 8} x2={xR + 8} y1={y} y2={y} />
			<path class="fx-pos__curve" d={curve(i)} />
			{#each P as col, t (t)}
				{@const v = col[i]}
				<line class="stem" x1={xt(t)} x2={xt(t)} y1={y} y2={y - amp * v} />
				<circle class={v >= 0 ? 'dot--value' : 'dot--open'} cx={xt(t)} cy={y - amp * v} r={t === sel ? 3.3 : 2.4} />
			{/each}
			<text class="lbl lbl--micro" x={xL - 14} y={y + 3.5} text-anchor="end">{i + 1}</text>
			<text class="lbl" x={xL - (narrow ? 30 : 34)} y={y + 4} text-anchor="end">{i % 2 === 0 ? 'sin' : 'cos'}</text>
			{#if i % 2 === 0 && !narrow}
				<text class="lbl lbl--micro fx-pos__omega" x={xL - 70} y={y + band / 2 + 4} text-anchor="end"><tspan class="m">ω</tspan><tspan class="m sub" dy="3">{r}</tspan><tspan dy="-3">{'\u00a0'}= {fmtOmega(omega(r))}</tspan></text>
				<line class="fx-pos__brace" x1={xL - 62} x2={xL - 62} y1={y - band * 0.36} y2={y + band * 1.36} />
			{/if}
		{/each}
		<!-- position axis -->
		{#each P as _, t (t)}
			<text class="lbl lbl--micro" class:lbl--strong={t === sel} x={xt(t)} y={top + band * D + 18} text-anchor="middle">{t + 1}</text>
		{/each}
		<text class="lbl lbl--micro" x={narrow ? xL - 14 : xR + 8} y={top + band * D + 18} text-anchor={narrow ? 'end' : 'start'}>{narrow ? '' : 'position '}<tspan class="m m--s">t</tspan></text>
		<text class="lbl lbl--micro" x={xL} y={top - 12}>row <tspan class="m m--s">i</tspan> of <tspan class="m m--s">P</tspan>, sampled at <tspan class="m m--s">τ = t − 1</tspan></text>

		{#if !narrow}
			<!-- P as written -->
			<text class="lbl lbl--micro" x={mx} y={top - 12}><tspan class="m m--s">P</tspan>, as a matrix</text>
			<rect class="matrix__frame" x={mx} y={top} width={hc * T} height={band * D} />
			<g class="matrix matrix--value">
				{#each P as col, t (t)}
					{#each col as v, i (i)}
						{@const side = band * 0.62 * Math.sqrt(Math.min(1, Math.abs(v)))}
						<rect class="matrix__cell" class:matrix__cell--neg={v < 0} x={mx + hc * (t + 0.5) - side / 2} y={yRow(i) - side / 2} width={side} height={side} />
					{/each}
				{/each}
			</g>
			<rect class="fx-pos__colbox" x={mx + hc * sel} y={top} width={hc} height={band * D} />
			<!-- the column added at position t -->
			<SpokeGlyph values={P[sel]} x={gx} y={top + (band * D) / 2} rmax={rG} scale={rG / 1} weight={1.4} tip={2.8} />
			<text class="lbl" x={gx} y={top + (band * D) / 2 + rG + 26} text-anchor="middle"><tspan class="m m--s">P</tspan><tspan class="m sub" dy="3">{sel + 1}</tspan><tspan dy="-3">, added at position {sel + 1}</tspan></text>
		{/if}

		{#if interactive}
			{#each P as _, t (t)}
				<rect class="hit" x={xt(t) - span / (T - 1) / 2} y={top - 6} width={span / (T - 1)} height={band * D + 34} role="button" tabindex="0" aria-label="position {t + 1}" onpointerenter={() => (sel = t)} onfocus={() => (sel = t)} onclick={() => (sel = t)} onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); (sel = t); } }} />
				{#if !narrow}<rect class="hit" x={mx + hc * t} y={top} width={hc} height={band * D} role="button" aria-label="read this entry; details in the live readout" tabindex="0" onpointerenter={() => (sel = t)} onfocus={() => (sel = t)} onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); (sel = t); } }} />{/if}
			{/each}
		{/if}
	</svg>
	{#if interactive}
		<p class="readout" aria-live="polite">
			<span class="readout__who">Position {sel + 1}.</span>
			<span class="readout__part"><i>P</i><sub>{sel + 1}</sub> = {readout}.</span>
			<span class="readout__hint">Point at a position, or use ←/→.</span>
		</p>
	{/if}
</div>

<style>
	.fx-pos :global(.m) { font-family: var(--serif); font-style: italic; font-size: 13px; fill: var(--ink); }
	.fx-pos :global(.m--s) { font-size: 12.5px; }
	.fx-pos :global(.sub) { font-size: 9.5px; }
	.fx-pos__curve { fill: none; stroke: var(--ink-3); stroke-width: var(--hair); }
	.fx-pos__sel { fill: var(--value-tint); opacity: 0.55; }
	.fx-pos__colbox { fill: none; stroke: var(--ink); stroke-width: 0.8px; }
	.fx-pos__brace { stroke: var(--ink-3); stroke-width: var(--hair); }
	.fx-pos__omega { fill: var(--ink-2); }
	:global([data-mode='print']) .fx-pos__sel { display: none; }
</style>

<script module>
	export const meta = { size: 'wide', title: 'How big is big' };
</script>

<script>
	// Figure: parameter counts as areas, as a chain of zooms. Panel k is a square of area
	// ∝ p_k, drawn at a fixed size; in its corner sits the square of the model before it, at
	// its true relative size, joined to the previous panel by two hairlines (the zoom).
	// Panel 1 is the specimen itself: its trained θ, one square per parameter (area ∝ |θᵢ|).
	// Counts: the specimen's from its layout; the Shakespeare model's from the artifact
	// (artifacts/shakespeare/loss.json, via the page's artifacts); GPT-2 and GPT-3 from
	// their papers, cited in the caption: the only external numbers in the book.
	import { getContext } from 'svelte';
	import { measure } from '../primitives/measure.js';
	import { layoutWidth } from '../primitives/scales.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const artifacts = $derived(getContext('artifacts')?.() ?? {});
	let width = $state(868);

	// published counts (Radford et al. 2019, table 2: 1542M; Brown et al. 2020: 175.0B)
	const PUBLISHED = [
		{ name: 'GPT-2', note: 'largest, 2019', p: 1.542e9 },
		{ name: 'GPT-3', note: '2020', p: 175e9 }
	];
	const models = $derived([
		{ name: 'the specimen', note: 'this book', p: specimen.paramCount },
		...(artifacts.shakespeareModel?.param_count ? [{ name: 'Shakespeare', note: 'chapter 9', p: artifacts.shakespeareModel.param_count }] : []),
		...PUBLISHED
	]);
	const grid = $derived(Math.ceil(Math.sqrt(specimen.paramCount)));
	const tmax = $derived(Math.max(...specimen.theta.map(Math.abs)));

	// geometry
	const W = $derived(layoutWidth(width, mode, 330));
	const n = $derived(models.length);
	const narrow = $derived(W < 620);
	const cols = $derived(narrow ? 2 : n);
	const gap = $derived(narrow ? 34 : 48);
	const S = $derived(Math.min(170, (W - gap * (cols - 1) - 10) / cols));
	const rowH = $derived(S + 96);
	const px = (k) => 5 + (k % cols) * (S + gap);
	const py = (k) => 20 + Math.floor(k / cols) * rowH;
	const H = $derived(20 + Math.ceil(n / cols) * rowH);
	const rel = (k) => (k > 0 ? Math.sqrt(models[k - 1].p / models[k].p) : 1);

	const fmtCount = (p) => {
		if (p >= 1e9) return `${+(p / 1e9).toPrecision(3)} billion`;
		if (p >= 1e6) return `${+(p / 1e6).toPrecision(3)} million`;
		return p.toLocaleString('en-GB').replace(/,/g, ' ');
	};
	const ratio = (k) => (k > 0 ? Math.round(models[k].p / models[k - 1].p) : null);
</script>

<div class="fx fx-scale" use:measure={(w) => (width = w)}>
	<svg viewBox="0 0 {W} {H}" width={W} height={H} role="img" aria-label="Parameter counts as areas: {models.map((m) => `${m.name} ${fmtCount(m.p)}`).join('; ')}.">
		{#each models as m, k (m.name)}
			{@const x = px(k)}
			{@const y = py(k)}
			{@const r = rel(k)}
			<!-- the model's square -->
			<rect class="fx-scale__sq" class:fx-scale__sq--first={k === 0} x={x} y={y} width={S} height={S} />
			{#if k === 0}
				<!-- the specimen: its own parameters, one square each -->
				<g class="matrix matrix--value">
					{#each specimen.theta as v, i (i)}
						{@const cw = S / grid}
						{@const s = cw * 0.9 * Math.sqrt(Math.abs(v) / tmax)}
						{#if s > 0.25}<rect class="matrix__cell" class:matrix__cell--neg={v < 0} x={x + cw * (i % grid) + (cw - s) / 2} y={y + cw * Math.floor(i / grid) + (cw - s) / 2} width={s} height={s} />{/if}
					{/each}
				</g>
			{:else}
				<!-- the previous model, at its true size, in the corner -->
				{@const s = Math.max(0.8, S * r)}
				<rect class="fx-scale__prev" x={x + 3} y={y + 3} width={s} height={s} />
				{#if k % cols !== 0}
					<!-- zoom lines back to the previous panel -->
					<line class="fx-scale__zoom" x1={px(k - 1) + S} y1={py(k - 1)} x2={x + 3 + s} y2={y + 3} />
					<line class="fx-scale__zoom" x1={px(k - 1) + S} y1={py(k - 1) + S} x2={x + 3 + s} y2={y + 3 + s} />
				{/if}
				<text class="lbl lbl--micro" x={x + 3 + s + 5} y={y + 3 + Math.max(8, s) + 2}>{models[k - 1].name}</text>
			{/if}
			<!-- labels -->
			<text class="lbl lbl--head" x={x} y={y + S + 20}>{m.name}</text>
			<text class="lbl" x={x} y={y + S + 37}>{fmtCount(m.p)} parameters</text>
			<text class="lbl lbl--micro" x={x} y={y + S + 52}>{m.note}</text>
			{#if ratio(k)}<text class="lbl lbl--micro" x={x} y={y + S + 66}>×{ratio(k).toLocaleString('en-GB').replace(/,/g, ' ')} the one before</text>{/if}
		{/each}
		{#if !artifacts.shakespeareModel}
			<text class="lbl lbl--micro" x={5} y={H - 4}>(the Shakespeare model’s artifact is missing; it is left out)</text>
		{/if}
	</svg>
</div>

<style>
	.fx-scale__sq { fill: none; stroke: var(--ink-3); stroke-width: var(--hair); }
	.fx-scale__sq--first { stroke: var(--rule); }
	.fx-scale__prev { fill: var(--value); }
	.fx-scale__zoom { stroke: var(--rule); stroke-width: var(--hair); }
</style>

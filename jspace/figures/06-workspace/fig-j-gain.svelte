<script module>
	export const meta = { size: 'wide', title: 'Do the MLPs amplify J-lens directions?' };
</script>

<script>
	// Figure: the MLP gain of four populations of unit directions, block by block, both models.
	// The gain of a unit direction v is ‖W2 relu(W1 LN(v) + b1) + b2‖, the MLP branch alone
	// (its LayerNorm included, the identity path not), divided by the median over 2000 random
	// unit directions; so random directions have median 1 (hairline) and their quartiles are
	// the dashed hairlines. Each population: its median as a mark, its quartiles as a
	// vertical hairline. J-lens vectors at the MLP's input depth: covector-ink bar; rows of
	// Wout (the logit lens): open ring in ink; columns of the previous block's W2 (directions
	// neurons write): value-ink dot (absent in block 1). Numbers: both exports'
	// structure.mlp[b].populations.
	import { getContext } from 'svelte';
	import { measure } from '#figures/primitives/measure.js';
	import { fmt, layoutWidth } from '#figures/primitives/scales.js';
	import { depthLabel } from '../primitives/jdata.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const art = $derived(getContext('artifacts')()?.jspace ?? {});
	const interactive = $derived(mode !== 'print');
	let width = $state(868);
	let focus = $state(null);

	const POP = [
		{ key: 'j_lens', name: 'J-lens vectors', cls: 'bar' },
		{ key: 'logit_lens', name: 'rows of W_out', cls: 'ring' },
		{ key: 'previous_neurons', name: 'neuron outputs', cls: 'dot' }
	];
	// one column per block: the specimen's one, then the Shakespeare model's
	const cols = $derived([
		...(art.specimen?.structure.mlp ?? []).map((m) => ({ model: 'specimen', m })),
		...(art.shakespeare?.structure.mlp ?? []).map((m) => ({ model: 'Shakespeare', m }))
	]);
	const vals = $derived(cols.flatMap((c) => Object.values(c.m.populations).flatMap((p) => [p.q25, p.q75])));
	const lo = $derived(Math.min(0.5, Math.floor(Math.min(...vals) * 4) / 4));
	const hi = $derived(Math.max(1.5, Math.ceil(Math.max(...vals) * 4) / 4));

	const W = $derived(layoutWidth(width, mode, 340));
	const narrow = $derived(W < 620);
	const padL = $derived(narrow ? 34 : 70);
	const padR = 8;
	const sep = $derived(narrow ? 18 : 34); // the gap between the two models
	const cw = $derived((W - padL - padR - sep) / Math.max(1, cols.length));
	const cx = (k) => padL + cw * k + cw / 2 + (k >= 1 && cols[0]?.model === 'specimen' ? sep : 0);
	const top = 44, hPlot = $derived(narrow ? 200 : 240);
	const y = (v) => top + hPlot - ((v - lo) / (hi - lo)) * hPlot;
	const H = $derived(top + hPlot + 58);
	const off = $derived(Math.min(18, cw * 0.18));
	const ticks = $derived(Array.from({ length: Math.round((hi - lo) / 0.25) + 1 }, (_, k) => lo + 0.25 * k).filter((t) => Math.abs(t * 2 - Math.round(t * 2)) < 1e-9));

	const readout = $derived.by(() => {
		if (focus == null) return null;
		const c = cols[focus], P = c.m.populations;
		return `${c.model}, block ${c.m.block} (J-lens vectors at depth ${depthLabel(c.m.block - 0.5)}): ` + POP.filter((p) => P[p.key]).map((p) => `${p.name} median ${fmt(P[p.key].median, 2)} (quartiles ${fmt(P[p.key].q25, 2)}–${fmt(P[p.key].q75, 2)})`).join('; ') + `; random quartiles ${fmt(P.random.q25, 2)}–${fmt(P.random.q75, 2)}.`;
	});
</script>

<div class="fx fx-gain" use:measure={(w) => (width = w)}>
	{#if !cols.length}
		<p class="fig__pending">The J-lens exports are missing.</p>
	{:else}
		<svg viewBox="0 0 {W} {H}" width={W} height={H} role={interactive ? 'group' : 'img'} aria-label="Median MLP gain, random directions = 1. {cols.map((c) => `${c.model} block ${c.m.block}: J-lens ${fmt(c.m.populations.j_lens.median, 2)}, rows of Wout ${fmt(c.m.populations.logit_lens.median, 2)}${c.m.populations.previous_neurons ? `, neuron outputs ${fmt(c.m.populations.previous_neurons.median, 2)}` : ''}`).join('; ')}">
			<!-- legend, once -->
			<g transform="translate({padL} 14)">
				<rect class="gain__bar" x="0" y="-2" width="14" height="3.2" /><text class="lbl lbl--micro" x="20" y="3">J-lens vectors</text>
				<circle class="gain__ring" cx={narrow ? 104 : 116} cy="0" r="3.4" /><text class="lbl lbl--micro" x={narrow ? 112 : 124} y="3">rows of W<tspan dy="2" font-size="8">out</tspan></text>
				<circle class="dot--value" cx={narrow ? 196 : 222} cy="0" r="3.4" /><text class="lbl lbl--micro" x={narrow ? 204 : 230} y="3">neuron outputs</text>
				{#if !narrow}<line class="gain__rq" x1="332" x2="350" y1="-4" y2="-4" /><line class="gain__rq" x1="332" x2="350" y1="4" y2="4" /><text class="lbl lbl--micro" x="356" y="3">random directions: quartiles; median 1</text>{/if}
			</g>
			<!-- scale -->
			{#each ticks as t (t)}
				<line class={t === 1 ? 'gain__one' : 'rule'} x1={padL - 4} x2={W - padR} y1={y(t)} y2={y(t)} />
				<text class="lbl lbl--micro" x={padL - 8} y={y(t) + 3} text-anchor="end">{fmt(t, t % 1 ? 2 : 0)}</text>
			{/each}
			<text class="lbl lbl--micro" x={padL - 8} y={top - 8} text-anchor="end">gain</text>
			{#each cols as c, k (k)}
				{@const P = c.m.populations}
				{#each [P.random.q25, P.random.q75] as q (q)}
					<line class="gain__rq" x1={cx(k) - cw * 0.38} x2={cx(k) + cw * 0.38} y1={y(q)} y2={y(q)} />
				{/each}
				{#each POP as p, i (p.key)}
					{#if P[p.key]}
						{@const xx = cx(k) + (i - 1) * off}
						<line class="gain__iqr gain__iqr--{p.cls}" x1={xx} x2={xx} y1={y(P[p.key].q25)} y2={y(P[p.key].q75)} />
						{#if p.cls === 'bar'}<rect class="gain__bar" x={xx - 7} y={y(P[p.key].median) - 1.6} width="14" height="3.2" />
						{:else if p.cls === 'ring'}<circle class="gain__ring" cx={xx} cy={y(P[p.key].median)} r="3.4" />
						{:else}<circle class="dot--value" cx={xx} cy={y(P[p.key].median)} r="3.4" />{/if}
					{/if}
				{/each}
				<text class="lbl" x={cx(k)} y={top + hPlot + 18} text-anchor="middle">block {c.m.block}</text>
				{#if interactive}
					<rect class="hit" x={cx(k) - cw / 2} y={top} width={cw} height={hPlot + 24} role="button" tabindex="0" aria-label="{c.model} block {c.m.block}"
						onpointerenter={() => (focus = k)} onfocus={() => (focus = k)} onpointerleave={() => (focus = null)} onblur={() => (focus = null)} onkeydown={() => {}} />
				{/if}
			{/each}
			<!-- the two models -->
			{#each ['specimen', 'Shakespeare'] as name (name)}
				{@const ks = cols.map((c, k) => (c.model === name ? k : -1)).filter((k) => k >= 0)}
				{#if ks.length}
					<path class="leader" d="M{cx(ks[0]) - cw * 0.4} {top + hPlot + 28} v4 H{cx(ks.at(-1)) + cw * 0.4} v-4" />
					<text class="lbl lbl--head" x={(cx(ks[0]) + cx(ks.at(-1))) / 2} y={top + hPlot + 48} text-anchor="middle">{name === 'specimen' ? 'the specimen' : 'the Shakespeare model'}</text>
				{/if}
			{/each}
		</svg>
		{#if interactive}
			<p class="readout" aria-live="polite">
				{#if readout}<span class="readout__part">{readout}</span>{:else}<span class="readout__hint">Point at a block for its numbers.</span>{/if}
			</p>
		{/if}
	{/if}
</div>

<style>
	.gain__bar { fill: var(--covector); }
	.gain__ring { fill: var(--paper); stroke: var(--ink-2); stroke-width: 1.1px; }
	.gain__rq { stroke: var(--ink-3); stroke-width: var(--hair); stroke-dasharray: 3 2; }
	.gain__one { stroke: var(--ink-3); stroke-width: var(--hair); }
	.gain__iqr { stroke-width: 1px; }
	.gain__iqr--bar { stroke: var(--covector); }
	.gain__iqr--ring { stroke: var(--ink-2); }
	.gain__iqr--dot { stroke: var(--value); }
</style>

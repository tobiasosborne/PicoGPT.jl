<script module>
	export const meta = { size: 'wide', title: 'The J-space of the Shakespeare model, depth by depth' };
</script>

<script>
	// Figure: how much of the Shakespeare model's stream lies in its J-space, depth by depth,
	// and where does the J frame stop beating random directions?
	// Nine panels, the depths 0, ½, …, 4. In each, against k = 0 … V (V = 65): the share of
	// ‖x‖² captured by greedy nonnegative pursuit with the J-lens vectors (value ink) and with
	// random frames of V unit vectors (ink hairline; mean of 5 frames), averaged over 2048
	// held-out positions. The occupancy, the first k at which the J frame's gain from one more
	// vector falls below the random frame's, is a vertical hairline. At the right: the share in
	// the full span of the V vectors (value) and in random V-dimensional subspaces (ink), with
	// V/D, the expected share of a random subspace. The paper's typical k = 25 is ticked.
	// Data: jspace.shakespeare.j_space[j].{k, residual_fraction, random_residual_fraction,
	// occupancy, span_fraction, random_span_fraction}. Interactive: point at a panel.
	import { getContext } from 'svelte';
	import { measure } from '#figures/primitives/measure.js';
	import { fmt, layoutWidth } from '#figures/primitives/scales.js';
	import { depthLabel } from '../primitives/jdata.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const sh = $derived(getContext('artifacts')()?.jspace?.shakespeare ?? null);
	const interactive = $derived(mode !== 'print');
	let hover = $state(null);
	let width = $state(868);
	const W = $derived(layoutWidth(width, mode, 330));
	const narrow = $derived(W < 640);

	const V = $derived(sh?.model.config.V ?? 0), D = $derived(sh?.model.config.D ?? 1);
	const KREF = 25; // the paper's typical sparsity, ticked for orientation
	const panels = $derived(sh ? sh.depth.map((d, j) => {
		const S = sh.j_space[j];
		return { j, d, k: S.k, J: S.residual_fraction.map((r) => 1 - r), R: S.random_residual_fraction.map((r) => 1 - r), occ: S.occupancy, span: S.span_fraction, rspan: S.random_span_fraction };
	}) : []);

	// layout: 3 × 3 (2 columns when narrow)
	const ncol = $derived(narrow ? 2 : 3);
	const gapX = $derived(narrow ? 22 : 30), gapY = 30;
	const padL = 30;
	const panelW = $derived((W - padL - gapX * (ncol - 1)) / ncol);
	const plotW = $derived(panelW - 34);
	const plotH = $derived(narrow ? 92 : 104);
	const panelH = $derived(plotH + 44);
	const px = (i) => padL + (i % ncol) * (panelW + gapX);
	const py = (i) => 26 + Math.floor(i / ncol) * (panelH + gapY);
	const H = $derived(26 + Math.ceil(panels.length / ncol) * (panelH + gapY) - gapY + 8);
	const xk = (i, k) => px(i) + (plotW * k) / Math.max(1, V);
	const yv = (i, v) => py(i) + plotH * (1 - v);
	const xSpan = (i) => px(i) + plotW + 18;
	const path = (i, ys) => ys.map((v, k) => `${k ? 'L' : 'M'}${xk(i, k).toFixed(1)} ${yv(i, v).toFixed(1)}`).join(' ');
	const pct = (f) => `${fmt(100 * f, 0)}%`;

	const readout = $derived.by(() => {
		if (!panels.length) return '';
		const P = panels[hover ?? panels.length - 1];
		const occ = P.occ == null ? 'the J frame never falls behind' : `occupancy ${P.occ}`;
		return `Depth ${depthLabel(P.d)}: ${occ}; at k = ${KREF} the J frame captures ${pct(P.J[KREF])} of ‖x‖², random frames ${pct(P.R[KREF])}; with all ${V}, ${pct(P.J[V])} and ${pct(P.R[V])}. The span of the ${V} lens vectors holds ${pct(P.span)}, random ${V}-dimensional subspaces ${pct(P.rspan)} (${V}/${D} = ${pct(V / D)} expected).`;
	});
	const aria = $derived(panels.map((P) => `depth ${depthLabel(P.d)}: occupancy ${P.occ ?? 'none'}, span ${pct(P.span)} against ${pct(P.rspan)}`).join('; '));
</script>

<div class="fx fx-occ" use:measure={(w) => (width = w)}>
	{#if sh}
		<svg viewBox="0 0 {W} {H}" width={W} height={H} role={interactive ? 'group' : 'img'} aria-label="Greedy J-space fits of the Shakespeare model by depth. {aria}">
			{#each panels as P, i (P.j)}
				{@const on = hover === i}
				<g class="fx-occ__panel">
					<text class="lbl lbl--head" x={px(i)} y={py(i) - 10}>depth {depthLabel(P.d)}</text>
					<line class="axis" x1={px(i)} x2={px(i) + plotW} y1={yv(i, 0)} y2={yv(i, 0)} />
					{#each [0, 0.5, 1] as v (v)}
						<line class="rule" x1={px(i)} x2={px(i) + plotW} y1={yv(i, v)} y2={yv(i, v)} />
						{#if i % ncol === 0}<text class="lbl lbl--micro" x={px(i) - 5} y={yv(i, v) + 3.5} text-anchor="end">{v === 0.5 ? '½' : v}</text>{/if}
					{/each}
					{#each [0, KREF, V] as k (k)}
						<line class="axis" x1={xk(i, k)} x2={xk(i, k)} y1={yv(i, 0)} y2={yv(i, 0) + 3} />
						<text class="lbl lbl--micro" x={xk(i, k)} y={yv(i, 0) + 13} text-anchor="middle">{k}</text>
					{/each}
					{#if P.occ != null}
						<line class="fx-occ__occ" x1={xk(i, P.occ)} x2={xk(i, P.occ)} y1={yv(i, 0)} y2={yv(i, Math.max(P.J[P.occ], P.R[P.occ])) - 6} />
						<text class="lbl lbl--micro fx-occ__occl" x={xk(i, P.occ) + 4} y={yv(i, Math.max(P.J[P.occ], P.R[P.occ])) - 8}>{P.occ}</text>
					{/if}
					<path class="fx-occ__rand" d={path(i, P.R)} />
					<path class="fx-occ__j" class:is-on={on} d={path(i, P.J)} />
					<line class="rule" x1={xSpan(i)} x2={xSpan(i)} y1={yv(i, 0)} y2={yv(i, 1)} />
					<line class="fx-occ__spanj" x1={xSpan(i) - 7} x2={xSpan(i) + 7} y1={yv(i, P.span)} y2={yv(i, P.span)} />
					<line class="fx-occ__spanr" x1={xSpan(i) - 7} x2={xSpan(i) + 7} y1={yv(i, P.rspan)} y2={yv(i, P.rspan)} />
					{#if i === 0}
						{@const jAbove = P.J[V] >= P.R[V]}
						<text class="lbl lbl--micro" x={xSpan(i)} y={yv(i, 1) - 5} text-anchor="middle">span</text>
						<text class="lbl lbl--micro lbl--value" x={xk(i, V) - 2} y={yv(i, P.J[V]) + (jAbove ? -6 : 12)} text-anchor="end">J frame</text>
						<text class="lbl lbl--micro" x={xk(i, V) - 2} y={yv(i, P.R[V]) + (jAbove ? 12 : -6)} text-anchor="end">random</text>
					{/if}
					{#if interactive}
						<rect class="hit" x={px(i) - 6} y={py(i) - 22} width={panelW + 6} height={panelH + 10} role="button" tabindex="0" aria-label="depth {depthLabel(P.d)}" onpointerenter={() => (hover = i)} onfocus={() => (hover = i)} />
					{/if}
				</g>
			{/each}
		</svg>
		{#if interactive}
			<p class="readout" aria-live="polite"><span class="readout__part">{readout}</span><span class="readout__hint">Point at a depth.</span></p>
		{/if}
	{:else}
		<p class="fig__pending">The J-lens export (jspace/artifacts/shakespeare-jlens.json) is not available.</p>
	{/if}
</div>

<style>
	.fx-occ__rand { fill: none; stroke: var(--ink-2); stroke-width: var(--hair); }
	.fx-occ__j { fill: none; stroke: var(--value); stroke-width: var(--line); }
	.fx-occ__j.is-on { stroke-width: var(--heavy); }
	.fx-occ__occ { stroke: var(--ink); stroke-width: var(--hair); }
	.fx-occ__occl { fill: var(--ink); }
	.fx-occ__spanj { stroke: var(--value); stroke-width: 2.2px; }
	.fx-occ__spanr { stroke: var(--ink-2); stroke-width: 1.2px; }
</style>

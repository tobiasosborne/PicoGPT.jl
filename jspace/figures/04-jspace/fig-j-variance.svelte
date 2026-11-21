<script module>
	export const meta = { size: 'wide', title: 'How much of the stream lies in the J-space, against chance' };
</script>

<script>
	// Figure: how much of the specimen's stream lies in the J-space, against chance?
	// Three panels, the depths 0, ½, 1. In each, against the sparsity k = 0 … V: the share of
	// ‖x‖² that the nearest point of the J-space captures, 1 − d_k(x)²/‖x‖², averaged over all
	// 80 columns (value ink, dots; the 80 columns themselves as faint lines), and the same for
	// random frames of V unit vectors (ink, open dots; mean of 20 frames): the control. At the
	// right, the share in the full span of the V lens vectors (value) and in random
	// V-dimensional subspaces (ink), with V/D, what a random subspace holds on average.
	// The gap between the k = V cone and the span is the price of nonnegativity; the gap
	// between the span and all of it is the price of V < D.
	// Data: jspace.specimen.j_space[j].{distance, norm, random_fraction, span_fraction,
	// random_span_fraction}. Interactive: point at a k. Print: the same.
	import { getContext } from 'svelte';
	import { measure } from '#figures/primitives/measure.js';
	import { fmt, layoutWidth } from '#figures/primitives/scales.js';
	import { at, depthLabel } from '../primitives/jdata.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const js = $derived(getContext('artifacts')()?.jspace?.specimen ?? null);
	const V = $derived(specimen.config.V), D = $derived(specimen.config.D);
	const interactive = $derived(mode !== 'print');
	let hoverK = $state(null);
	let width = $state(868);
	const W = $derived(layoutWidth(width, mode, 330));
	const narrow = $derived(W < 640);

	const mean = (xs) => xs.reduce((s, v) => s + v, 0) / xs.length;
	const panels = $derived(js ? js.depth.map((d, j) => {
		const S = js.j_space[j], N = S.norm.length;
		const cols = Array.from({ length: N }, (_, n) => [0, ...Array.from({ length: V }, (_, kk) => 1 - at(S.distance, kk, n) ** 2 / S.norm[n] ** 2)]);
		const J = Array.from({ length: V + 1 }, (_, k) => mean(cols.map((c) => c[k])));
		const R = [0, ...Array.from({ length: V }, (_, kk) => mean(Array.from({ length: N }, (_, n) => 1 - at(S.random_fraction, kk, n))))];
		return { j, d, cols, J, R, span: mean(S.span_fraction), rspan: mean(S.random_span_fraction) };
	}) : []);

	// layout
	const cols = $derived(narrow ? 1 : panels.length);
	const gap = $derived(narrow ? 0 : 26);
	const padL = 34;
	const panelW = $derived((W - padL - gap * (cols - 1)) / cols);
	const plotW = $derived(panelW - 64);
	const plotH = $derived(narrow ? 150 : 170);
	const top0 = 30;
	const panelH = $derived(plotH + 62);
	const px = (i) => padL + (narrow ? 0 : i * (panelW + gap));
	const py = (i) => top0 + (narrow ? i * panelH : 0);
	const H = $derived(narrow ? top0 + panels.length * panelH : top0 + panelH);
	const xk = (i, k) => px(i) + (plotW * k) / V;
	const yv = (i, v) => py(i) + plotH * (1 - v);
	const xSpan = (i) => px(i) + plotW + 30;
	const path = (i, ys) => ys.map((v, k) => `${k ? 'L' : 'M'}${xk(i, k).toFixed(1)} ${yv(i, v).toFixed(1)}`).join(' ');
	const pct = (f) => `${fmt(100 * f, 0)}%`;

	const readout = $derived.by(() => {
		if (!panels.length) return '';
		const k = hoverK ?? V;
		return `k = ${k}: ` + panels.map((P) => `depth ${depthLabel(P.d)}, the J-space captures ${pct(P.J[k])} of ‖x‖² on average, random frames ${pct(P.R[k])}`).join('; ') + '.';
	});
	const aria = $derived(panels.map((P) => `depth ${depthLabel(P.d)}: at k = ${V} the J cone captures ${pct(P.J[V])}, random ${pct(P.R[V])}; the J span ${pct(P.span)}, random spans ${pct(P.rspan)}`).join('. '));
</script>

<div class="fx fx-var" use:measure={(w) => (width = w)}>
	{#if js}
		<svg viewBox="0 0 {W} {H}" width={W} height={H} role={interactive ? 'group' : 'img'} aria-label="Share of the stream captured by the J-space, by k. {aria}">
			{#each panels as P, i (P.j)}
				<g class="fx-var__panel">
					<text class="lbl lbl--head" x={px(i)} y={py(i) - 12}>depth {depthLabel(P.d)}</text>
					<!-- axes -->
					<line class="axis" x1={px(i)} x2={px(i) + plotW} y1={yv(i, 0)} y2={yv(i, 0)} />
					{#each [0, 0.5, 1] as v (v)}
						<line class="rule" x1={px(i)} x2={px(i) + plotW} y1={yv(i, v)} y2={yv(i, v)} />
						{#if i === 0 || narrow}<text class="lbl lbl--micro" x={px(i) - 6} y={yv(i, v) + 3.5} text-anchor="end">{v === 0.5 ? '½' : v}</text>{/if}
					{/each}
					{#each Array.from({ length: V + 1 }, (_, k) => k) as k (k)}
						<line class="axis" x1={xk(i, k)} x2={xk(i, k)} y1={yv(i, 0)} y2={yv(i, 0) + 3} />
						<text class="lbl lbl--micro" class:lbl--strong={hoverK === k} x={xk(i, k)} y={yv(i, 0) + 14} text-anchor="middle">{k}</text>
					{/each}
					<text class="lbl lbl--micro" x={px(i) + plotW / 2} y={yv(i, 0) + 28} text-anchor="middle">k, the number of lens vectors allowed</text>
					<!-- the columns, faint -->
					<g class="fx-var__cols">
						{#each P.cols as c, n (n)}<path d={path(i, c)} />{/each}
					</g>
					<!-- random frames, then the J frame -->
					<path class="fx-var__rand" d={path(i, P.R)} />
					{#each P.R as v, k (k)}{#if k}<circle class="fx-var__rdot" cx={xk(i, k)} cy={yv(i, v)} r="2.6" />{/if}{/each}
					<path class="fx-var__j" d={path(i, P.J)} />
					{#each P.J as v, k (k)}{#if k}<circle class="dot--value" cx={xk(i, k)} cy={yv(i, v)} r="3" />{/if}{/each}
					<!-- the spans -->
					<line class="fx-var__spanj" x1={xSpan(i) - 9} x2={xSpan(i) + 9} y1={yv(i, P.span)} y2={yv(i, P.span)} />
					<line class="fx-var__spanr" x1={xSpan(i) - 9} x2={xSpan(i) + 9} y1={yv(i, P.rspan)} y2={yv(i, P.rspan)} />
					<line class="rule" x1={xSpan(i)} x2={xSpan(i)} y1={yv(i, 0)} y2={yv(i, 1)} />
					<text class="lbl lbl--micro" x={xSpan(i)} y={yv(i, 0) + 14} text-anchor="middle">span</text>
					{#if i === panels.length - 1 || narrow}
						{@const jAbove = P.J[V] >= P.R[V]}
						<text class="lbl lbl--micro lbl--value" x={xk(i, V) - 4} y={yv(i, P.J[V]) + (jAbove ? 16 : -8)} text-anchor="end">J frame</text>
						<text class="lbl lbl--micro" x={xk(i, V) - 4} y={yv(i, P.R[V]) + (jAbove ? 16 : -8)} text-anchor="end">random</text>
						<text class="lbl lbl--micro lbl--value" x={xSpan(i) + 12} y={yv(i, P.span) + 3.5}>J</text>
						<text class="lbl lbl--micro" x={xSpan(i) + 12} y={yv(i, P.rspan) + 3.5}>random</text>
					{/if}
					{#if i === 0}
						<line class="fx-var__ref" x1={xSpan(i) - 12} x2={xSpan(i) + 12} y1={yv(i, V / D)} y2={yv(i, V / D)} />
						<text class="lbl lbl--micro" x={xSpan(i) + 15} y={yv(i, V / D) + 3.5}>{V}/{D}</text>
					{/if}
					{#if interactive}
						{#each Array.from({ length: V + 1 }, (_, k) => k) as k (k)}
							<rect class="hit" x={xk(i, k) - plotW / V / 2} y={py(i)} width={plotW / V} height={plotH + 18} role="button" tabindex="0" aria-label="k = {k}" onpointerenter={() => (hoverK = k)} onfocus={() => (hoverK = k)} />
						{/each}
					{/if}
				</g>
			{/each}
		</svg>
		{#if interactive}
			<p class="readout" aria-live="polite"><span class="readout__part">{readout}</span><span class="readout__hint">Point at a k.</span></p>
		{/if}
	{:else}
		<p class="fig__pending">The J-lens export (jspace/artifacts/specimen-jlens.json) is not available.</p>
	{/if}
</div>

<style>
	.fx-var__cols path { fill: none; stroke: var(--value); stroke-width: var(--hair); opacity: 0.13; }
	.fx-var__rand { fill: none; stroke: var(--ink-2); stroke-width: var(--hair); }
	.fx-var__rdot { fill: var(--paper); stroke: var(--ink-2); stroke-width: 1px; }
	.fx-var__j { fill: none; stroke: var(--value); stroke-width: var(--line); }
	.fx-var__spanj { stroke: var(--value); stroke-width: 2.2px; }
	.fx-var__spanr { stroke: var(--ink-2); stroke-width: 1.2px; }
	.fx-var__ref { stroke: var(--ink-3); stroke-width: var(--hair); stroke-dasharray: 2 2; }
</style>

<script>
	// Attention as arcs between characters. For a reader i, one arc to every
	// source j ≤ i; stroke width ∝ A_ij (weight 1 ↦ wmax). dir = -1 draws above the
	// line (head 1 by convention), +1 below (head 2). A reader reading itself is a
	// loop. Every arc is one number; arcs below `labelAt` are drawn but not labelled.
	import { fmtProb } from './scales.js';

	let {
		xs = [],
		y = 0,
		A = [],
		reader = 0,
		dir = -1,
		wmax = 12,
		aspect = 0.62,
		hmax = 1e9,
		loop = 9,
		labelAt = 0.1,
		labels = true,
		muted = false,
		ink = 'value',
		digits = 2
	} = $props();

	const arcs = $derived.by(() => {
		const out = [];
		const row = A[reader] ?? [];
		for (let j = 0; j <= reader && j < row.length; j++) {
			const w = row[j];
			const x1 = xs[j], x2 = xs[reader];
			if (j === reader) {
				const r = loop + w * wmax * 0.55; // keep the loop open however heavy
				const cy = y + dir * r;
				out.push({ j, w, self: true, cx: x2, cy, r, lx: x2, ly: y + dir * (2 * r + 9) });
			} else {
				const rx = Math.abs(x2 - x1) / 2;
				const ry = Math.min(hmax, rx * aspect); // constant aspect: arcs nest, never cross
				const mid = (x1 + x2) / 2;
				// elliptical arc from source to reader
				out.push({ j, w, self: false, x: x1, d: `M ${x1} ${y} A ${rx} ${ry} 0 0 ${dir < 0 ? 1 : 0} ${x2} ${y}`, lx: mid, ly: y + dir * (ry + 7) });
			}
		}
		return out;
	});
</script>

<g class="arcs arcs--{ink}" class:arcs--muted={muted}>
	{#each arcs as a (a.j)}
		{#if a.self}
			<circle class="arcs__arc arcs__arc--self" cx={a.cx} cy={a.cy} r={a.r} stroke-width={Math.max(0.0001, a.w * wmax)} />
		{:else}
			<path class="arcs__arc" d={a.d} stroke-width={Math.max(0.0001, a.w * wmax)} />
			{#if a.w >= 0.02}
				<path class="arcs__arc" d="M {a.x - 3} {y + dir * 6} L {a.x} {y} L {a.x + 3} {y + dir * 6}" stroke-width={Math.max(0.6, a.w * wmax * 0.3)} />
			{/if}
		{/if}
	{/each}
	{#if labels && !muted}
		{#each arcs as a (a.j)}
			{#if a.w >= labelAt}
				<text class="arcs__label" x={a.lx} y={a.ly} dy={dir < 0 ? 0 : '0.72em'} text-anchor="middle">{fmtProb(a.w, digits)}</text>
			{/if}
		{/each}
	{/if}
</g>

<script>
	// How to read the arcs: an enlarged detail of the figure's own reader — its
	// two nearest sources and itself — with leader lines. Real numbers throughout.
	import { getContext } from 'svelte';
	import HowToRead from '../primitives/HowToRead.svelte';
	import Arcs from '../primitives/Arcs.svelte';
	import { fmt, fmtProb, showChar, clamp } from '../primitives/scales.js';

	let { attrs = {} } = $props();
	const specimen = $derived(getContext('specimen')());
	const start = $derived(clamp((+attrs.window || 10) - 1, 0, specimen.windows.length - 1));
	const i = $derived(clamp((+attrs.reader || 8) - 1, 2, specimen.config.T - 1));
	const tr = $derived(specimen.windowTrace(start));
	// the detail: the reader and its strongest source in each head (true weights)
	const strongest = (A, not = -1) => A[i].slice(0, i).map((w, j) => ({ w, j })).filter((o) => o.j !== not).sort((a, b) => b.w - a.w)[0].j;
	const idx = $derived.by(() => {
		const a = strongest(tr.heads[0].A);
		let b = strongest(tr.heads[1].A);
		if (b === a) b = strongest(tr.heads[1].A, a);
		return [Math.min(a, b), Math.max(a, b), i];
	});
	const xs = [40, 105, 170];
	const sub = (A) => idx.map((r) => idx.map((c) => (c <= r ? A[r][c] : 0)));
	const A1 = $derived(sub(tr.heads[0].A));
	const A2 = $derived(sub(tr.heads[1].A));
	const yb = 118;
	// the heaviest head-1 arc in the detail gets the "width" note
	const top = $derived([0, 1].map((k) => ({ k, w: A1[2][k] })).sort((a, b) => b.w - a.w)[0]);
	const topX = $derived((xs[top.k] + xs[2]) / 2);
	const topY = $derived(yb - 30 - ((xs[2] - xs[top.k]) / 2) * 0.62);
	const loopR = $derived(9 + A1[2][2] * 13 * 0.55);
</script>

<HowToRead width={240} height={262} notes={[
	{ ax: topX, ay: topY, side: 'above', len: 14, text: `width ∝ weight: ‘${showChar(tr.chars[i])}’ at ${i + 1} reads\n‘${showChar(tr.chars[idx[top.k]])}’ at ${idx[top.k] + 1} with ${fmtProb(top.w)} in head 1`, align: 'middle' },
	{ ax: xs[2] + loopR * 0.72, ay: yb - 30 - loopR * 1.7, side: 'right', len: 8, text: 'loop:\nitself' },
	{ ax: xs[2] + 14, ay: yb - 9, side: 'right', len: 10, text: 'reader' },
	{ ax: xs[0] + 6, ay: yb + 9 + 10, side: 'below', len: 52, text: 'head 1 above the line,\nhead 2 below; arrows\npoint to sources', align: 'start', dx: -6 }
]}>
	{#snippet glyph()}
		<g class="fx" style="font-family: var(--sans)">
			<Arcs {xs} y={yb - 30} A={A1} reader={2} dir={-1} wmax={13} labels={false} />
			<Arcs {xs} y={yb + 9} A={A2} reader={2} dir={1} wmax={13} labels={false} />
			{#each idx as j, k (k)}
				<text x={xs[k]} y={yb} text-anchor="middle" font-size="32" font-weight={k === 2 ? 640 : 380} style="font-family: var(--serif); fill: var(--ink)">{showChar(tr.chars[j])}</text>
			{/each}
		</g>
	{/snippet}
</HowToRead>

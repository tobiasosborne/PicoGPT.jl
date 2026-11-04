<script>
	// How to read the arcs: an enlarged detail of the figure's own reader — its
	// strongest earlier source in each head, and itself — with leader lines. Real
	// numbers throughout; the notes name the reader's own loop and its heaviest arc.
	import { getContext } from 'svelte';
	import HowToRead from '../primitives/HowToRead.svelte';
	import Arcs, { loopGeom } from '../primitives/Arcs.svelte';
	import { fmtProb, showChar, clamp } from '../primitives/scales.js';

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
	const WMAX = 13, LOOP = 9, ASPECT = 0.62;
	const xs = [26, 88, 150];
	const sub = (A) => idx.map((r) => idx.map((c) => (c <= r ? A[r][c] : 0)));
	const A1 = $derived(sub(tr.heads[0].A));
	const A2 = $derived(sub(tr.heads[1].A));
	const yb = 152;
	const yUp = yb - 30, yDown = yb + 9;
	// the reader's own loop in head 1, from the same geometry the figure draws
	const lp = $derived(loopGeom(xs[2], yUp, A1[2][2], { wmax: WMAX, loop: LOOP, dir: -1 }));
	// the heaviest arc to an earlier character in head 2 (drawn below) gets the "width" note
	const top = $derived([0, 1].map((k) => ({ k, w: A2[2][k] })).sort((a, b) => b.w - a.w)[0]);
	const topX = $derived((xs[top.k] + xs[2]) / 2);
	const topY = $derived(yDown + ((xs[2] - xs[top.k]) / 2) * ASPECT + (top.w * WMAX) / 2);
	const rd = $derived(showChar(tr.chars[i]));
</script>

<HowToRead width={240} height={262} notes={[
	{ ax: lp.cx, ay: lp.cy - lp.outer, side: 'above', len: 12, text: `loop: ‘${rd}’ reads itself,\n${fmtProb(A1[2][2])} in head 1`, align: 'middle', dx: 8 },
	{ ax: xs[1], ay: yUp - 6, side: 'above', len: 62, text: 'head 1 above,\nhead 2 below;\narrows point\nto sources', align: 'end', dx: -6 },
	{ ax: xs[2] + 14, ay: yb - 9, side: 'right', len: 10, text: 'reader' },
	{ ax: topX, ay: topY, side: 'below', len: 12, text: `width ∝ weight: ‘${rd}’ reads\n‘${showChar(tr.chars[idx[top.k]])}’ at ${idx[top.k] + 1} with ${fmtProb(top.w)} in head 2`, align: 'middle' }
]}>
	{#snippet glyph()}
		<g class="fx" style="font-family: var(--sans)">
			<Arcs {xs} y={yUp} A={A1} reader={2} dir={-1} wmax={WMAX} loop={LOOP} aspect={ASPECT} labels={false} />
			<Arcs {xs} y={yDown} A={A2} reader={2} dir={1} wmax={WMAX} loop={LOOP} aspect={ASPECT} labels={false} />
			{#each idx as j, k (k)}
				<text x={xs[k]} y={yb} text-anchor="middle" font-size="32" font-weight={k === 2 ? 640 : 380} style="font-family: var(--serif); fill: var(--ink)">{showChar(tr.chars[j])}</text>
			{/each}
		</g>
	{/snippet}
</HowToRead>

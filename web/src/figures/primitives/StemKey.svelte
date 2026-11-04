<script>
	// "How to read" the residual stem: one enlarged stretch of a real column — x₀, the
	// attention branch with its update a, the ⊕, and x_mid — drawn by ResidualStem
	// itself, with hairline leaders to plain-language notes. Pass the column's vectors
	// (x0, a, mid) and the figure's width scale k (px per unit of ‖·‖/√D).
	import HowToRead from './HowToRead.svelte';
	import ResidualStem from './ResidualStem.svelte';
	import { rms } from './stem.js';
	import { fmt, maxAbs, spokeScale } from './scales.js';

	let { x0, a, mid, k = 7, label = 'a' } = $props();

	const rG = 15;
	const kk = $derived(k * 1.25);
	const sc = $derived(spokeScale(rG, maxAbs(x0, a, mid)));
	const x = 104, bx = 152;
	const Y = { x0: 26, f1: 62, n1: 96, p1: 130, mid: 166, f2: 230, n2: 260, p2: 290, out: 320, end: 350 };
	const w = $derived({ x0: rms(x0) * kk, mid: rms(mid) * kk, out: rms(mid) * kk });
	const wb = $derived({ a: rms(a) * kk, y: 0 });
	const notes = $derived([
		{ ax: x - w.x0 / 2 - 1, ay: (Y.f1 + Y.x0) / 2 + 8, side: 'left', len: 18, text: `width: the size\nof the column,\n‖x‖/√D = ${fmt(rms(x0))}` },
		{ ax: (x + bx) / 2 + 4, ay: Y.f1 + 2, side: 'above', len: 22, text: 'the branch\nreads a copy', align: 'start', dx: -2 },
		{ ax: bx - 2, ay: Y.p1 - 7, side: 'right', len: 20, text: `returns its\nupdate ${label}, of\nwidth ${fmt(rms(a))}` },
		{ ax: x - 5, ay: Y.p1, side: 'left', len: 18, text: `⊕: x ← x + ${label},\nEuler form; no\nsmall-step assumption` },
		{ ax: x + rG * 0.7, ay: Y.mid + rG * 0.7, side: 'right', len: 24, text: 'a spoke glyph:\nthe column at\nthat depth' }
	]);
</script>

<HowToRead width={244} height={202} {notes}>
	{#snippet glyph()}
		<g class="fx">
			<ResidualStem {x} {bx} {Y} {w} {wb} g={{ x0, a, mid, y: a.map(() => 0), out: mid }} {rG} scale={sc.scale} t={1} track={false} part="attention" />
		</g>
	{/snippet}
</HowToRead>

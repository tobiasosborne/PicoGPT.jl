<script>
	// "How to read" the pullback: the residual stem in covector ink. One enlarged stretch
	// of a real column's backward pass — x̄_mid arriving from below, copied into the
	// attention branch at the forward ⊕, the branch's return added at the forward fork
	// (the covector ⊕), and x̄₀ above — drawn by ResidualStem itself.
	// Pass the column's adjoints (bx0, batt, bmid) and the figure's width scale k.
	import HowToRead from './HowToRead.svelte';
	import ResidualStem from './ResidualStem.svelte';
	import { rms } from './stem.js';
	import { maxAbs, spokeScale } from './scales.js';

	let { bx0, batt, bmid, k = 7, sci = (v) => String(v) } = $props();

	const rG = 15;
	const kk = $derived(k * 1.25);
	const sc = $derived(spokeScale(rG, maxAbs(bx0, batt, bmid)));
	const x = 104, bx = 152;
	const Y = { x0: 26, f1: 62, n1: 96, p1: 130, mid: 166, f2: 230, n2: 260, p2: 290, out: 320, end: 350 };
	const w = $derived({ x0: rms(bx0) * kk, mid: rms(bmid) * kk, out: rms(bmid) * kk });
	const wb = $derived({ a: rms(batt) * kk, y: 0 });
	const notes = $derived([
		{ ax: x - w.mid / 2 - 1, ay: Y.p1 + 16, side: 'left', len: 14, text: `hatched: an\nadjoint, flowing\nup; its width is\n${sci(rms(bmid))}` },
		{ ax: bx - 2, ay: Y.p1 - 7, side: 'right', len: 20, text: 'copied into\nthe branch' },
		{ ax: bx - 2, ay: Y.f1 + 12, side: 'right', len: 20, text: `the branch\nhands back\n${sci(rms(batt))}` },
		{ ax: x - 5, ay: Y.f1, side: 'left', len: 14, text: '⊕: adjoints add\nwhere the\nforward pass\ncopied' },
		{ ax: x + rG * 0.75, ay: Y.x0 - rG * 0.6, side: 'right', len: 26, text: 'bar tips:\na covector' }
	]);
</script>

<HowToRead width={244} height={202} {notes}>
	{#snippet glyph()}
		<g class="fx">
			<ResidualStem {x} {bx} {Y} {w} {wb} g={{ x0: bx0, a: batt, mid: bmid, y: bmid.map(() => 0), out: bmid }} {rG} scale={sc.scale} t={1} track={false} ink="covector" part="attention" />
		</g>
	{/snippet}
</HowToRead>

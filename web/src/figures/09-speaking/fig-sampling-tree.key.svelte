<script>
	// How to read a sampling column: the figure's first step, enlarged. The column is the
	// interval [0, 1] cut into the probabilities of the characters (vocabulary order); the
	// draw u picks the segment it falls in.
	import { getContext } from 'svelte';
	import HowToRead from '../primitives/HowToRead.svelte';
	import { gibbs } from '../primitives/EnergyLadder.svelte';
	import { fmt, fmtProb, showChar } from '../primitives/scales.js';

	let { attrs = {} } = $props();
	const specimen = $derived(getContext('specimen')());
	const vocab = $derived(specimen.vocab);
	const st = $derived(specimen.speak(String(attrs.prompt ?? '␣').replace(/␣/g, ' '), 1, { temperature: attrs.tau != null ? +attrs.tau : 1, seed: +(attrs.seed ?? 1) }).steps[0]);
	const p = $derived(gibbs(st.z, attrs.tau != null ? +attrs.tau : 1));
	const F = $derived(p.reduce((acc, v) => [...acc, (acc.at(-1) ?? 0) + v], []));
	const x = 104, y0 = 22, h = 196, bw = 30;
	const a0 = $derived(st.next - 1);
	const alt = $derived(p.map((v, a) => ({ v, a })).filter((o) => o.a !== a0).sort((u, v) => v.v - u.v)[0].a);
	const mid = (a) => y0 + (F[a] - p[a] / 2) * h;
</script>

<HowToRead width={246} height={250} notes={[
	{ ax: x - bw / 2, ay: y0, side: 'left', len: 16, text: '0' , dot: false },
	{ ax: x - bw / 2, ay: y0 + h, side: 'left', len: 16, text: '1', dot: false },
	{ ax: x - bw / 2 - 2, ay: y0 + h * 0.2, side: 'left', len: 14, text: 'the interval [0, 1]\ncut into the\nprobabilities, in\nvocabulary order' },
	{ ax: x - bw / 2 - 6, ay: y0 + (st.u ?? 0) * h, side: 'left', len: 12, text: `u = ${fmt(st.u ?? 0, 3)}: a\nuniform random\nnumber` },
	{ ax: x + bw / 2, ay: y0 + (F[a0] - p[a0]) * h + 10, side: 'right', len: 18, text: `solid: the segment u\nfalls in, ‘${showChar(vocab[a0])}’ (${fmtProb(p[a0], 2)})` },
	{ ax: x + bw / 2, ay: mid(alt) - 6, side: 'right', len: 18, text: `outlined: a path\nnot taken, ‘${showChar(vocab[alt])}’` }
]}>
	{#snippet glyph()}
		<g class="fx">
			{#each p as pa, a (a)}
				{#if pa * h >= 0.4}
					<rect class={a === a0 ? 'bar--value' : ''} x={x - bw / 2} y={y0 + (F[a] - pa) * h} width={bw} height={pa * h} style={a === a0 ? '' : 'fill: none; stroke: var(--ink-3); stroke-width: var(--hair)'} />
					{#if pa * h >= 14}<text class="tok" x={x} y={y0 + (F[a] - pa / 2) * h + 5} text-anchor="middle" font-size="15" style={a === a0 ? 'fill: var(--paper)' : 'fill: var(--ink-2)'}>{showChar(vocab[a])}</text>{/if}
				{/if}
			{/each}
			{#if st.u != null}<line x1={x - bw / 2 - 6} x2={x + bw / 2 + 6} y1={y0 + st.u * h} y2={y0 + st.u * h} style="stroke: var(--ink); stroke-width: 1.6px" />{/if}
		</g>
	{/snippet}
</HowToRead>

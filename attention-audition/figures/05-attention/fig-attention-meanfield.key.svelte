<script>
	// How to read the two stacks: the arithmetic mixture's favourite character, enlarged, as
	// it is built in each stack of the figure's own reader — above, the sources' shares of
	// probability; below, their weighted surprises in nats. Real numbers throughout.
	import { getContext } from 'svelte';
	import HowToRead from '#figures/primitives/HowToRead.svelte';
	import { fmt, fmtProb, showChar, clamp } from '#figures/primitives/scales.js';
	import { slice, dissenter, argmax } from './mixture.js';

	let { attrs = {} } = $props();
	const mx = $derived(getContext('artifacts')()?.mixture ?? null);
	const s = $derived.by(() => {
		if (!mx) return null;
		const f = mx.figure_readers?.[0];
		const b = +attrs.window || f?.b || 1, i = +attrs.position || f?.i || 1;
		const rec = mx.readers.find((r) => r.b === b && r.i === i) ?? mx.readers[0];
		return slice(rec, clamp(+attrs.head || f?.head || 1, 1, mx.config.H), mx.vocab.data.length);
	});
	const y = $derived(s ? argmax(s.mix) : 0);
	const ch = $derived(s ? showChar(mx.vocab.data[y]) : '');
	const veto = $derived(s ? dissenter(s) : 0);
	const share = $derived(s ? s.P.map((row, j) => s.A[j] * row[y]) : []);
	const big = $derived(argmax(share));
	const x = 18, bw = 22;
	const kP = 130; // probability: px per unit
	const kS = 7.5; // surprise: px per nat
	const yA = $derived(42 + seg(share, share.length) * kP); // baselines: each column hangs
	const yS = $derived(yA + 62 + seg(surp, surp.length) * kS); // below its note
	const seg = (vals, j) => vals.slice(0, j).reduce((a, b) => a + b, 0);
	const surp = $derived(s ? s.S.map((row) => row[y]) : []);
	const topA = $derived(yA - seg(share, share.length) * kP);
	const topS = $derived(yS - seg(surp, surp.length) * kS);
	const midA = $derived(yA - (seg(share, big) + share[big] / 2) * kP);
	const midS = $derived(yS - (seg(surp, veto) + surp[veto] / 2) * kS);
</script>

{#if s}
	<HowToRead width={244} height={yS + 22} notes={[
		{ ax: x + bw / 2, ay: topA, side: 'above', len: 12, align: 'start', dx: -10, text: `height: the arithmetic mixture’s\np(‘${ch}’) = ${fmt(s.mix[y], 2)}, its sources’ shares added` },
		{ ax: x + bw, ay: midA, side: 'right', len: 16, text: `source ${big + 1}’s share: its weight\n${fmt(s.A[big], 2)} × its p(‘${ch}’) ${fmtProb(s.P[big][y], 2)}` },
		{ ax: x + bw, ay: yA - 1, side: 'right', len: 16, text: `source ${veto + 1} (weight ${fmt(s.A[veto], 2)}) adds\n${fmtProb(share[veto], 2, true)}: a share can only add` },
		{ ax: x + bw / 2, ay: topS, side: 'above', len: 12, align: 'start', dx: -10, text: `height: mean surprise ${fmt(s.Sbar[y], 2)} nats;\none nat above another: 1/e of its weight` },
		{ ax: x + bw, ay: midS, side: 'right', len: 16, text: `source ${veto + 1}: weight ${fmt(s.A[veto], 2)} ×\nsurprise ${fmt(-Math.log(s.P[veto][y]), 2)} nats` }
	]}>
		{#snippet glyph()}
			<g class="fx">
				{#each [[share, yA, kP], [surp, yS, kS]] as [vals, y0, k] (y0)}
					<line class="axis" x1={x - 3} x2={x + bw + 3} y1={y0} y2={y0} />
					{#each vals as v, j (j)}
						{@const h = v * k}
						{#if h > 0.05}
							{@const gap = h > 2.4 ? 1 : 0}
							<rect class="bar--value" x={x} y={y0 - seg(vals, j) * k - h + gap / 2} width={bw} height={Math.max(0.35, h - gap)} />
							{#if h >= 12}<text class="key-mf__segl" x={x + bw / 2} y={y0 - (seg(vals, j) + v / 2) * k + 3.4} text-anchor="middle">{j + 1}</text>{/if}
						{/if}
					{/each}
					<text class="tok" x={x + bw / 2} y={y0 + 14} text-anchor="middle" font-size="13">{ch}</text>
				{/each}
			</g>
		{/snippet}
	</HowToRead>
{/if}

<style>
	.key-mf__segl { font-family: var(--sans); font-size: 9.5px; fill: var(--paper); font-variant-numeric: tabular-nums lining-nums; }
</style>

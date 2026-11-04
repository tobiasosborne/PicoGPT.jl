<script>
	// How to read the surprise bars: one window of the figure, enlarged, untrained (outline)
	// and trained (solid). The window shown is the one whose first prediction keeps the
	// largest surprise after training, so the key shows a bar that falls and one that cannot.
	import { getContext } from 'svelte';
	import HowToRead from '../primitives/HowToRead.svelte';
	import { fmt, showChar } from '../primitives/scales.js';

	const specimen = $derived(getContext('specimen')());
	const T = $derived(specimen.config.T);
	const lnV = $derived(Math.log(specimen.config.V));
	const sur = (pr, b) => pr.probs.slice(b * T, b * T + T).map((p, t) => -Math.log(p[pr.y[b][t] - 1]));
	const P0 = $derived(specimen.predictions(specimen.theta0));
	const P1 = $derived(specimen.predictions(specimen.theta));
	const b = $derived.by(() => {
		let best = 0, v = -1;
		for (let k = 0; k < P1.B; k++) { const s = sur(P1, k)[0]; if (s > v) { v = s; best = k; } }
		return best;
	});
	const s0 = $derived(sur(P0, b));
	const s1 = $derived(sur(P1, b));
	const x0 = 8, cw = 21, base = 150, hmax = 74;
	const sMax = $derived(Math.max(lnV, ...s0) * 1.05);
	const h = (v) => (v / sMax) * hmax;
	const ys = $derived(Array.from(specimen.corpus.slice(b + 1, b + 1 + T)));
	const ctx1 = $derived(showChar(specimen.corpus[b]));
</script>

<HowToRead width={246} height={226} notes={[
	{ ax: x0 + cw * 0.5, ay: base - h(s1[0]), side: 'above', len: 64 - h(s1[0]) + 34, text: `solid: −ln q(‘${showChar(ys[0])}’) after ‘${ctx1}’,\ntrained: ${fmt(s1[0], 2)}`, align: 'start', dx: -8 },
	{ ax: x0 + cw * 5.5 - 7, ay: base - h(s0[5]) + 1, side: 'above', len: 22, text: `outline: untrained,\n${fmt(s0[5], 2)}`, align: 'start', dx: -6 },
	{ ax: x0 + cw * T, ay: base - h(lnV), side: 'right', len: 5, text: `ln ${specimen.config.V}:\nuniform\nguessing` },
	{ ax: x0 + cw * 3.5, ay: base + 26, side: 'below', len: 8, text: 'one window; each bar sits\nunder the character it predicts', align: 'start', dx: -60 }
]}>
	{#snippet glyph()}
		<g class="fx">
			<line x1={x0} x2={x0 + cw * T} y1={base} y2={base} style="stroke: var(--ink-3); stroke-width: var(--hair)" />
			<line x1={x0} x2={x0 + cw * T} y1={base - h(lnV)} y2={base - h(lnV)} style="stroke: var(--rule); stroke-width: var(--hair); stroke-dasharray: 2 2" />
			{#each ys as ch, t (t)}
				{@const x = x0 + cw * (t + 0.5)}
				<rect x={x - 7} y={base - h(s0[t])} width="14" height={h(s0[t])} style="fill: none; stroke: var(--ink-3); stroke-width: var(--hair)" />
				<rect class="bar--value" x={x - 5.5} y={base - h(s1[t])} width="11" height={h(s1[t])} />
				<text class="tok" x={x} y={base + 18} text-anchor="middle" font-size="16">{showChar(ch)}</text>
			{/each}
		</g>
	{/snippet}
</HowToRead>

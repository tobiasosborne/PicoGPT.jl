<script>
	// A reminder of the covector key introduced in the prologue: the figure's own prediction, two characters
	// enlarged — the one that came (y) and the other the loss pushes hardest — with p (value,
	// solid), e_y (outline) and p − e_y (covector, hatched, signed by direction).
	import { getContext } from 'svelte';
	import HowToRead from '../primitives/HowToRead.svelte';
	import { fmt, showChar, clamp } from '../primitives/scales.js';

	let { attrs = {} } = $props();
	const specimen = $derived(getContext('specimen')());
	const T = $derived(specimen.config.T);
	const start = $derived(clamp((+attrs.window || 3) - 1, 0, specimen.lab.train.x.length - 1));
	const pos = $derived(clamp((+attrs.position || 1) - 1, 0, T - 1));
	const pr = $derived(specimen.predictions(attrs.state === 'untrained' ? specimen.theta0 : specimen.theta));
	const n = $derived(start * T + pos);
	const p = $derived(pr.probs[n]);
	const y = $derived(pr.y[start][pos] - 1);
	const other = $derived(p.map((v, a) => ({ v, a })).filter((o) => o.a !== y).sort((u, v) => v.v - u.v)[0].a);
	const cols = $derived([other, y]);
	const xs = [44, 100];
	const base = 96, amp = 60, g0 = 178, gamp = 50;
</script>

<HowToRead title="Remember the two inks" width={246} height={268} notes={[
	{ ax: xs[1] + 13, ay: base - amp + 6, side: 'right', len: 12, text: 'outline: the one-hot\nvector of what came' },
	{ ax: xs[0], ay: base - p[other] * amp, side: 'above', len: 14, text: 'solid: p, the\nprediction', align: 'start', dx: -36 },
	{ ax: xs[1] + 9, ay: g0 + (1 - p[y]) * gamp * 0.6, side: 'right', len: 12, text: `hatched: the gradient,\n${fmt(p[y] - 1, 2)}; down: descent\nraises this logit` },
	{ ax: xs[0] - 9, ay: g0, side: 'below', len: 36, text: `up: descent lowers it, by ${fmt(p[other], 2)}`, align: 'start', dx: -2 }
]}>
	{#snippet glyph()}
		<g class="fx">
			<line class="axis" x1="16" x2="128" y1={base} y2={base} />
			<line class="axis" x1="16" x2="128" y1={g0} y2={g0} />
			{#each cols as a, k (k)}
				{#if a === y}<rect x={xs[k] - 13} y={base - amp} width="26" height={amp} style="fill: none; stroke: var(--ink); stroke-width: 1px" />{/if}
				<rect class="bar--value" x={xs[k] - 9} y={base - p[a] * amp} width="18" height={p[a] * amp} />
				{@const gv = p[a] - (a === y ? 1 : 0)}
				<rect class="bar--covector" x={xs[k] - 9} y={gv > 0 ? g0 - gv * gamp : g0} width="18" height={Math.abs(gv) * gamp} />
				<text class="tok" x={xs[k]} y={base + 18} text-anchor="middle" font-size="17">{showChar(specimen.vocab[a])}</text>
			{/each}
		</g>
	{/snippet}
</HowToRead>

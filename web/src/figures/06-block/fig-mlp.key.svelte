<script>
	// How to read one hidden unit of the MLP, enlarged: the figure's largest firing unit
	// and one that is switched off, for the figure's own column.
	import { getContext } from 'svelte';
	import HowToRead from '../primitives/HowToRead.svelte';
	import SpokeGlyph from '../primitives/SpokeGlyph.svelte';
	import { fmt, clamp, maxAbs, spokeScale } from '../primitives/scales.js';

	let { attrs = {} } = $props();
	const specimen = $derived(getContext('specimen')());
	const { D } = $derived(specimen.config);
	const b = $derived(clamp((+attrs.window || 1) - 1, 0, specimen.lab.train.x.length - 1));
	const j = $derived(clamp((+attrs.column || 4) - 1, 0, specimen.config.T - 1));
	const u = $derived(specimen.lab.gradTrace(specimen.theta).windows[b].val.U[j]);
	const W2 = $derived(specimen.params().find((p) => p.name.endsWith('W2')).values);
	const colOf = (k) => W2.slice(k * D, (k + 1) * D);
	const on = $derived(u.map((v, k) => ({ v, k, n: Math.max(0, v) * Math.hypot(...colOf(k)) })).sort((a, b) => b.n - a.n)[0]);
	const off = $derived(u.map((v, k) => ({ v, k })).filter((o) => o.v < 0).sort((a, b) => a.v - b.v)[0]);
	const uMax = $derived(Math.max(on.v, -off.v));
	const amp = 40, z = 66, gy = 148, rG = 30;
	const sc = $derived(spokeScale(rG, maxAbs(colOf(on.k).map((v) => v * on.v))));
	const x1 = 62, x2 = 132;
	const y1 = $derived(z - (on.v / uMax) * amp), y2 = $derived(z - (off.v / uMax) * amp);
</script>

<HowToRead width={244} height={232} notes={[
	{ ax: x1 + 4, ay: y1, side: 'right', len: 74, text: `u above the line:\nunit ${on.k + 1} fires, r = ${fmt(on.v, 2)}` },
	{ ax: x2 + 4, ay: y2, side: 'right', len: 4, text: `below: open; relu\nsets r = 0 (unit ${off.k + 1})` },
	{ ax: x2 + 2, ay: gy, side: 'right', len: 22, text: 'adds nothing' },
	{ ax: x1 - 6, ay: gy + 12, side: 'below', len: 30, text: `r × column ${on.k + 1} of W₂:\nwhat the unit adds to y`, align: 'start', dx: -50 }
]}>
	{#snippet glyph()}
		<g class="fx">
			<line x1="20" x2="170" y1={z} y2={z} stroke="var(--ink)" stroke-width="0.6" />
			<line x1={x1} x2={x1} y1={z} y2={y1} stroke="var(--value)" stroke-width="1.6" />
			<circle cx={x1} cy={y1} r="3.6" fill="var(--value)" />
			<line x1={x2} x2={x2} y1={z} y2={y2} stroke="var(--ink-3)" stroke-width="1.1" opacity="0.6" />
			<circle cx={x2} cy={y2} r="3.6" fill="var(--paper)" stroke="var(--ink-3)" stroke-width="1" />
			<SpokeGlyph values={colOf(on.k).map((v) => v * on.v)} x={x1} y={gy} rmax={rG} scale={sc.scale} compass={false} tip={2.4} />
			<circle cx={x2} cy={gy} r="1.3" fill="var(--ink-3)" />
		</g>
	{/snippet}
</HowToRead>

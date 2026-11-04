<script>
	// How to read the meridian plane: one column of the figure, enlarged — its point
	// (a, r) = (𝟏ᵀx/√D, ‖Πx‖), the move across onto the axis (centring) and the move
	// along the axis to just inside the circle ‖x‖ = √D (scaling). Real numbers.
	import { getContext } from 'svelte';
	import HowToRead from '../primitives/HowToRead.svelte';
	import { fmt, clamp } from '../primitives/scales.js';

	let { attrs = {} } = $props();
	const specimen = $derived(getContext('specimen')());
	const D = $derived(specimen.config.D);
	const b = $derived(clamp((+attrs.window || 1) - 1, 0, specimen.lab.train.x.length - 1));
	const j = $derived(clamp((+attrs.column || 4) - 1, 0, specimen.config.T - 1));
	const x = $derived(specimen.lab.gradTrace(specimen.theta).windows[b].val.Xmid[j]);
	const mu = $derived(x.reduce((s, v) => s + v, 0) / D);
	const a = $derived(mu * Math.sqrt(D));
	const r = $derived(Math.sqrt(x.reduce((s, v) => s + (v - mu) ** 2, 0)));
	const R = $derived(Math.sqrt(D));
	const u = $derived(Math.min(36, 132 / Math.max(r, R)));
	const cx = 122, cy = 200;
	const px = $derived(cx + a * u), py = $derived(cy - r * u);
	const top = $derived(cy - R * u);
	const left = $derived(a < 0);
	const notes = $derived([
		{ ax: px + (left ? -4 : 4), ay: py, side: left ? 'left' : 'right', len: 10, text: `x: ‖Πx‖\n= ${fmt(r, 2)}` },
		{ ax: (px + cx) / 2, ay: py - 2, side: 'above', len: 12, text: `centring: a = ${fmt(a, 2)} → 0` },
		{ ax: cx + 2, ay: (py + top) / 2, side: left ? 'right' : 'left', len: 16, text: 'scaling: down\nthe axis, to\njust inside' },
		{ ax: cx - R * u * Math.cos(0.32), ay: cy - R * u * Math.sin(0.32), side: 'above', len: 34, text: 'the sphere\n‖x‖ = √D', align: 'start', dx: -18 },
		{ ax: cx + R * u + 2, ay: cy, side: 'below', len: 4, text: 'along 𝟏', dot: false }
	]);
</script>

<HowToRead width={244} height={226} {notes}>
	{#snippet glyph()}
		<g class="fx">
			<line class="axis" x1={cx - R * u - 6} x2={cx + R * u + 12} y1={cy} y2={cy} />
			<line class="axis" x1={cx} x2={cx} y1={cy} y2={Math.min(py, top) - 8} />
			<path d="M {cx - R * u} {cy} A {R * u} {R * u} 0 0 1 {cx + R * u} {cy}" fill="none" stroke="var(--ink)" stroke-width="0.6" />
			<polyline points="{px},{py} {cx},{py} {cx},{top + 1}" fill="none" stroke="var(--value)" stroke-width="1.1" />
			<circle cx={px} cy={py} r="3.4" fill="var(--paper)" stroke="var(--value)" stroke-width="0.8" />
			<circle cx={cx} cy={top + 1} r="3.6" fill="var(--value)" />
		</g>
	{/snippet}
</HowToRead>

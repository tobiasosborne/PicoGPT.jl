<script>
	// A reminder of how to read a spoke glyph, on the figure's own query q_i, enlarged.
	// The full key is with the book's first spoke glyph (fig-one-hot, chapter 2).
	import { getContext } from 'svelte';
	import HowToRead from '../primitives/HowToRead.svelte';
	import SpokeGlyph from '../primitives/SpokeGlyph.svelte';
	import { fmt, clamp, maxAbs, spokeScale } from '../primitives/scales.js';

	let { attrs = {} } = $props();
	const specimen = $derived(getContext('specimen')());
	const start = $derived(clamp((+attrs.window || 10) - 1, 0, specimen.windows.length - 1));
	const i = $derived(clamp((+attrs.reader || 8) - 1, 0, specimen.config.T - 1));
	const h = $derived(clamp((+attrs.head || 1) - 1, 0, 1));
	const tr = $derived(specimen.windowTrace(start));
	const q = $derived(tr.heads[h].q[i]);
	const sc = $derived(spokeScale(52, maxAbs(q)));
	const cx = 100, cy = 78;
	// pick one positive and one negative component to point at
	const pos = $derived(q.map((v, r) => ({ v, r })).filter((o) => o.v > 0).sort((a, b) => b.v - a.v)[0]);
	const neg = $derived(q.map((v, r) => ({ v, r })).filter((o) => o.v < 0).sort((a, b) => a.v - b.v)[0]);
	const tipOf = (o) => {
		const a = -Math.PI / 2 + (2 * Math.PI * o.r) / q.length;
		const r = Math.abs(o.v) * sc.scale;
		return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) };
	};
	const sub = (r) => '₁₂₃₄₅₆₇₈₉'[r];
	const notes = $derived.by(() => {
		const out = [];
		if (pos) { const t = tipOf(pos); out.push({ ax: t.x + 3, ay: t.y, side: 'right', len: Math.max(8, 156 - t.x), text: `filled tip:\npositive, ${fmt(pos.v)}` }); }
		if (neg) { const t = tipOf(neg); out.push({ ax: t.x + 3, ay: t.y, side: 'right', len: Math.max(8, 156 - t.x), text: `open tip:\nnegative, ${fmt(neg.v)}` }); }
		out.push({ ax: cx - 3, ay: cy - sc.rmax * 0.5, side: 'left', len: 30, text: 'q₁ at 12\no’clock, then\nclockwise' });
		out.push({ ax: cx - sc.rmax, ay: cy + 0, side: 'below', len: 64, text: 'length ∝ |qᵣ|; the dots mark\nthe full scale of the figure', align: 'start', dx: -10 });
		return out;
	});
</script>

<HowToRead width={244} height={192} title="Reminder: spoke glyphs" {notes}>
	{#snippet glyph()}
		<g class="fx"><SpokeGlyph values={q} x={cx} y={cy} rmax={sc.rmax} scale={sc.scale} tip={3.2} weight={1.6} /></g>
	{/snippet}
</HowToRead>

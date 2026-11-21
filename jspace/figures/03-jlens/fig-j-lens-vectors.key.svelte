<script>
	// Reminder: a covector as a spoke glyph — the figure's own J-lens vector for the last
	// character of the vocabulary at depth ½, enlarged, with leaders to one positive and one
	// negative component (real numbers).
	import { getContext } from 'svelte';
	import HowToRead from '#figures/primitives/HowToRead.svelte';
	import SpokeGlyph from '#figures/primitives/SpokeGlyph.svelte';
	import { fmt, showChar, maxAbs, spokeScale } from '#figures/primitives/scales.js';
	import { at, depthLabel } from '../primitives/jdata.js';

	const specimen = $derived(getContext('specimen')());
	const js = $derived(getContext('artifacts')()?.jspace?.specimen ?? null);
	const D = $derived(specimen.config.D), V = $derived(specimen.config.V);
	// the vector whose largest positive and largest negative components point to opposite
	// sides (so that their notes do not meet), and are both long
	const pick = $derived.by(() => {
		let best = { a: 0, j: 0, score: -1 };
		if (!js) return best;
		for (let j = 0; j < js.depth.length; j++) for (let a = 0; a < V; a++) {
			const v = Array.from({ length: D }, (_, q) => at(js.lenses.vectors, a, q, j));
			const p = v.map((x, q) => ({ x, q })).filter((o) => o.x > 0).sort((u, w) => w.x - u.x)[0];
			const n = v.map((x, q) => ({ x, q })).filter((o) => o.x < 0).sort((u, w) => u.x - w.x)[0];
			if (!p || !n) continue;
			const side = (o) => Math.sign(Math.round(1e6 * Math.sin((2 * Math.PI * o.q) / D)));
			if (side(p) * side(n) >= 0) continue;
			const m = Math.max(...v.map(Math.abs)), score = Math.min(p.x, -n.x) / m;
			if (score > best.score) best = { a, j, score };
		}
		return best;
	});
	const j = $derived(pick.j), a = $derived(pick.a);
	const v = $derived(js ? Array.from({ length: D }, (_, q) => at(js.lenses.vectors, a, q, j)) : []);
	const sc = $derived(spokeScale(56, maxAbs(v)));
	const cx = 125, cy = 104;
	const tipOf = (o) => {
		const ang = -Math.PI / 2 + (2 * Math.PI * o.q) / D, r = Math.abs(o.x) * sc.scale;
		return { x: cx + r * Math.cos(ang), y: cy + r * Math.sin(ang) };
	};
	const pos = $derived(v.map((x, q) => ({ x, q })).filter((o) => o.x > 0).sort((p, q) => q.x - p.x)[0]);
	const neg = $derived(v.map((x, q) => ({ x, q })).filter((o) => o.x < 0).sort((p, q) => p.x - q.x)[0]);
	const note = (o, word) => {
		const t = tipOf(o), right = t.x >= cx;
		return { ax: t.x + (right ? 4 : -4), ay: t.y, side: right ? 'right' : 'left', len: Math.max(6, right ? 186 - t.x : t.x - 64), text: `${word},\ncomponent ${o.q + 1}\nis ${fmt(o.x)}` };
	};
	const notes = $derived([
		...(pos ? [note(pos, 'filled bar: +')] : []),
		...(neg ? [note(neg, 'open bar: −')] : []),
		{ ax: cx, ay: cy - sc.rmax - 3, side: 'above', len: 8, text: 'component 1 at 12 o’clock, then clockwise', align: 'middle' },
		{ ax: cx, ay: cy + sc.rmax + 3, side: 'below', len: 8, text: 'bars, not dots: a covector, in covector\nink; length ∝ |component|', align: 'middle' }
	]);
</script>

{#if js}
	<HowToRead width={250} height={226} title="Reminder: covector spokes" {notes}>
		{#snippet glyph()}
			<g class="fx"><SpokeGlyph values={v} x={cx} y={cy} rmax={sc.rmax} scale={sc.scale} ink="covector" tip={3.2} weight={1.5} /></g>
		{/snippet}
	</HowToRead>
	<p class="key__caption">The J-lens vector of ‘{showChar(specimen.vocab[a])}’ at depth {depthLabel(js.depth[j])}.</p>
{/if}

<style>
	.key__caption { font-family: var(--sans); font-size: var(--fs-micro, 11px); color: var(--ink-3); margin: 0.2em 0 0; text-indent: 0; }
</style>

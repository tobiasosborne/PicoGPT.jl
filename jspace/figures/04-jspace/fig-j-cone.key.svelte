<script>
	// How to read a cone glyph: the figure's own pair at its own column (the best pair), laid
	// flat in the plane of the two vectors with the bisector up, enlarged, with leaders. Real
	// numbers throughout: the vectors and the nearest point are those the figure draws.
	import { getContext } from 'svelte';
	import HowToRead from '#figures/primitives/HowToRead.svelte';
	import { clamp } from '#figures/primitives/scales.js';
	import ConeGlyph from '../primitives/ConeGlyph.svelte';
	import { at } from '../primitives/jdata.js';
	import { pairFit, polarEdges } from '../primitives/cone.js';

	let { attrs = {} } = $props();
	const specimen = $derived(getContext('specimen')());
	const js = $derived(getContext('artifacts')()?.jspace?.specimen ?? null);
	const V = $derived(specimen.config.V), D = $derived(specimen.config.D);
	const start = $derived(clamp((+attrs.window || 1) - 1, 0, 9));
	const pos = $derived(clamp((+attrs.column || 4) - 1, 0, 7));
	const depth = $derived(attrs.depth != null ? clamp(Math.round(2 * +attrs.depth), 0, 2) : 1);
	const tr = $derived(specimen.windowTrace(start));
	const x = $derived([tr.X0, tr.Xmid, tr.Xout][depth][pos]);
	const vecs = $derived(js ? Array.from({ length: V }, (_, a) => Array.from({ length: D }, (_, q) => at(js.lenses.vectors, a, q, depth))) : []);
	const f = $derived.by(() => {
		let best = null;
		for (let u = 1; u < V; u++) for (let s = 0; s < u; s++) { const g = { s, u, ...pairFit(vecs[s], vecs[u], x) }; if (!best || g.frac < best.frac) best = g; }
		return best;
	});
	const R = $derived(1.04 * Math.max(f.plane.a[0], Math.hypot(...f.plane.b), Math.hypot(...f.p)));
	const cx = 118, cy = 190, k = $derived(96 / R);
	const rot = $derived(Math.PI / 2 - f.plane.angle / 2);
	const P = $derived((u, v) => [cx + k * (u * Math.cos(rot) - v * Math.sin(rot)), cy - k * (u * Math.sin(rot) + v * Math.cos(rot))]);
	const unit = (v) => { const n = Math.hypot(...v); return [v[0] / n, v[1] / n]; };
	// the longer vector's tip (it reaches furthest out), a point in the hatching on the other
	// side, the nearest point, one ghost and one dotted polar edge
	const tips = $derived([P(...f.plane.a), P(...f.plane.b)]);
	const far = $derived(f.plane.a[0] >= Math.hypot(...f.plane.b) ? 0 : 1);
	const tip = $derived(tips[far]);
	const toRight = $derived(tip[0] >= cx);
	const hatchPt = $derived.by(() => { const t = f.plane.angle * (toRight ? 0.8 : 0.2); return P(0.86 * R * Math.cos(t), 0.86 * R * Math.sin(t)); });
	const qq = $derived(P(...f.q));
	const ghost = $derived.by(() => { const d = unit(far === 0 ? f.plane.a : f.plane.b); return P(-d[0] * R * 0.62, -d[1] * R * 0.62); });
	const polarPt = $derived.by(() => { const e = polarEdges(f.plane)[far === 0 ? 's' : 'u']; return P(e[0] * R * 0.55, e[1] * R * 0.55); });
	const inside = $derived(f.region === 'cone');
</script>

{#if js}
	<HowToRead width={250} height={330} title="How to read a cone" notes={[
		{ ax: tip[0], ay: tip[1], side: 'above', len: Math.max(8, tip[1] - 92), text: 'a lens vector;\nthe bar marks\na covector', align: toRight ? 'end' : 'start', dx: toRight ? 246 - tip[0] : 4 - tip[0] },
		{ ax: hatchPt[0], ay: hatchPt[1], side: 'above', len: Math.max(8, hatchPt[1] - 46), text: 'the cone: c₁v₁ + c₂v₂,\nboth c ≥ 0 (hatched)', align: toRight ? 'start' : 'end', dx: toRight ? 4 - hatchPt[0] : 246 - hatchPt[0] },
		{ ax: qq[0], ay: qq[1], side: toRight ? 'left' : 'right', len: toRight ? Math.max(6, qq[0] - 60) : Math.max(6, 190 - qq[0]), text: inside ? 'x_J: the\nnearest\npoint\n(= p here)' : 'x_J: the\nnearest\npoint' },
		{ ax: ghost[0], ay: ghost[1], side: 'below', len: Math.max(10, 270 - ghost[1]), text: 'dashed: −v,\nin the plane,\nnot in the cone', align: toRight ? 'start' : 'end', dx: toRight ? 4 - ghost[0] : 246 - ghost[0] },
		{ ax: polarPt[0], ay: polarPt[1], side: 'below', len: Math.max(10, 270 - polarPt[1]), text: 'dotted: past\nthem the nearest\npoint is 0', align: toRight ? 'end' : 'start', dx: toRight ? 246 - polarPt[0] : 4 - polarPt[0] }
	]}>
		{#snippet glyph()}
			<g class="fx"><ConeGlyph plane={f.plane} {R} {P} ghost polar hatch={9} tip={3} p={f.p} q={f.q} labels={false} /></g>
		{/snippet}
	</HowToRead>
{/if}

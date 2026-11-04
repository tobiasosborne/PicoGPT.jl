<script>
	// How to read a spoke glyph (a vector): the figure's own column E e_a, enlarged.
	// The book's first spoke glyph is in this figure, so the key lives here (DESIGN §6).
	// Notes sit in a column to the right; straight hairline leaders run to them from the
	// marks they describe (sorted by height, so the leaders never cross).
	import { getContext } from 'svelte';
	import HowToRead from '../primitives/HowToRead.svelte';
	import SpokeGlyph from '../primitives/SpokeGlyph.svelte';
	import { fmt, clamp, maxAbs, spokeScale } from '../primitives/scales.js';

	let { attrs = {} } = $props();
	const specimen = $derived(getContext('specimen')());
	const D = $derived(specimen.config.D);
	const a = $derived(clamp((+attrs.char || 5) - 1, 0, specimen.config.V - 1));
	const E = $derived(specimen.params().find((p) => p.name === 'E').values);
	const x = $derived(Array.from({ length: D }, (_, i) => E[i + D * a]));
	const sc = $derived(spokeScale(50, maxAbs(x)));
	const cx = 70, cy = 104, TX = 142, LH = 13;
	const SUB = '₁₂₃₄₅₆₇₈₉';
	const tips = $derived(
		x.map((v, k) => {
			const ang = -Math.PI / 2 + (2 * Math.PI * k) / D;
			const r = Math.abs(v) * sc.scale;
			return { v, k, x: cx + r * Math.cos(ang), y: cy + r * Math.sin(ang) };
		})
	);
	const notes = $derived.by(() => {
		const rest = tips.slice(1);
		const neg = [...rest].filter((o) => o.v < 0).sort((p, q) => p.v - q.v)[0];
		const pos = [...rest].filter((o) => o.v > 0).sort((p, q) => q.v - p.v)[0];
		const sign = [
			pos && { o: pos, lines: ['filled tip: positive', `x${SUB[pos.k]} = ${fmt(pos.v)}`] },
			neg && { o: neg, lines: ['open tip: negative', `x${SUB[neg.k]} = ${fmt(neg.v)}`] }
		].filter(Boolean).sort((p, q) => p.o.y - q.o.y);
		const out = [{ o: tips[0], lines: [`x${SUB[0]} at 12 o’clock,`, `then clockwise to x${SUB[D - 1]}`], ty: 22 }];
		sign.forEach((n, k) => out.push({ ...n, ty: 84 + k * 46 }));
		return out;
	});
	const scaleDot = $derived({ x: cx - sc.rmax * Math.SQRT1_2, y: cy + sc.rmax * Math.SQRT1_2 });
</script>

<HowToRead width={250} height={222} notes={[]}>
	{#snippet glyph()}
		<g class="fx"><SpokeGlyph values={x} x={cx} y={cy} rmax={sc.rmax} scale={sc.scale} tip={3.2} weight={1.6} /></g>
		{#each notes as n, k (k)}
			<line class="key__leader" x1={n.o.x + (n.o.x > cx ? 3 : 0)} y1={n.o.y + (k === 0 ? -4 : 0)} x2={TX - 4} y2={n.ty - 4} />
			<circle class="key__anchor" cx={n.o.x + (n.o.x > cx ? 3 : 0)} cy={n.o.y + (k === 0 ? -4 : 0)} r="1.5" />
			<text class="key__text" x={TX} y={n.ty}>
				{#each n.lines as line, i (i)}<tspan x={TX} dy={i ? LH : 0}>{line}</tspan>{/each}
			</text>
		{/each}
		<line class="key__leader" x1={scaleDot.x} y1={scaleDot.y} x2={scaleDot.x} y2={186} />
		<circle class="key__anchor" cx={scaleDot.x} cy={scaleDot.y} r="1.5" />
		<text class="key__text" x={scaleDot.x - 6} y={198}>
			<tspan x={scaleDot.x - 6}>length ∝ |xₖ|. The faint dots mark</tspan><tspan x={scaleDot.x - 6} dy={LH}>the figure’s full scale, one per spoke</tspan>
		</text>
	{/snippet}
</HowToRead>

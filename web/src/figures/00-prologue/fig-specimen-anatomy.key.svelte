<script>
	// How to read a matrix glyph (a Hinton diagram): an enlarged corner of the figure's own
	// [W_q | b_q], on the plate's scale (one reference for all of θ). First matrix glyph of
	// the book, so its key lives here.
	import { getContext } from 'svelte';
	import HowToRead from '../primitives/HowToRead.svelte';
	import { fmt } from '../primitives/scales.js';

	let { attrs = {} } = $props();
	const specimen = $derived(getContext('specimen')());
	const D = $derived(specimen.config.D);
	const get = (n) => specimen.params().find((p) => p.name.split('.').at(-1) === n).values;
	const Wq = $derived(get('Wq'));
	const bq = $derived(get('bq'));
	const vmax = $derived(Math.max(...specimen.theta.map(Math.abs)));
	const n = 4, cell = 21, x0 = 30, y0 = 40, bx = x0 + n * cell + 7;
	const at = (i, j) => Wq[i + D * j];
	const side = (v) => cell * 0.92 * Math.sqrt(Math.min(1, Math.abs(v) / vmax));
	const cells = $derived(Array.from({ length: n * n }, (_, k) => ({ i: k % n, j: Math.floor(k / n), v: at(k % n, Math.floor(k / n)) })));
	const pos = $derived([...cells].filter((q) => q.v > 0).sort((p, q) => q.v - p.v)[0]);
	const neg = $derived([...cells].filter((q) => q.v < 0).sort((p, q) => p.v - q.v)[0]);
	const ctr = (q) => ({ x: x0 + cell * (q.j + 0.5), y: y0 + cell * (q.i + 0.5) });
	const TX = 150, LH = 13;
	const notes = $derived.by(() => {
		const out = [];
		const sorted = [pos && { q: pos, lines: ['filled: positive', `W_q[${pos.i + 1}, ${pos.j + 1}] = ${fmt(pos.v)}`] }, neg && { q: neg, lines: ['open: negative', `W_q[${neg.i + 1}, ${neg.j + 1}] = ${fmt(neg.v)}`] }].filter(Boolean).sort((a, b) => a.q.i - b.q.i);
		sorted.forEach((s, k) => out.push({ ...s, from: ctr(s.q), ty: y0 + 6 + k * 40 }));
		return out;
	});
</script>

<HowToRead width={250} height={232} notes={[]}>
	{#snippet glyph()}
		<g class="fx">
			<text class="lbl lbl--micro" x={x0} y={y0 - 24}>a corner of <tspan font-style="italic" font-family="var(--serif)" font-size="12">W</tspan><tspan font-size="8" dy="2">q</tspan><tspan dy="-2"> and </tspan><tspan font-style="italic" font-family="var(--serif)" font-size="12">b</tspan><tspan font-size="8" dy="2">q</tspan></text>
			<rect class="matrix__frame" x={x0} y={y0} width={n * cell} height={n * cell} />
			<rect class="matrix__frame" x={bx} y={y0} width={cell} height={n * cell} />
			<g class="matrix matrix--value">
				{#each cells as q (q.i * 8 + q.j)}
					{@const s = side(q.v)}
					<rect class="matrix__cell" class:matrix__cell--neg={q.v < 0} x={x0 + cell * q.j + (cell - s) / 2} y={y0 + cell * q.i + (cell - s) / 2} width={s} height={s} />
				{/each}
				{#each Array.from({ length: n }) as _, i (i)}
					{@const v = bq[i]}
					{@const s = side(v)}
					<rect class="matrix__cell" class:matrix__cell--neg={v < 0} x={bx + (cell - s) / 2} y={y0 + cell * i + (cell - s) / 2} width={s} height={s} />
				{/each}
			</g>
			<text class="lbl lbl--micro" x={x0 - 5} y={y0 + cell * 0.5 + 3.5} text-anchor="end">1</text>
			<text class="lbl lbl--micro" x={x0 + cell * 0.5} y={y0 - 6} text-anchor="middle">1</text>
		</g>
		{#each notes as n, k (k)}
			<line class="key__leader" x1={n.from.x + 4} y1={n.from.y} x2={TX - 4} y2={n.ty - 4} />
			<circle class="key__anchor" cx={n.from.x + 4} cy={n.from.y} r="1.5" />
			<text class="key__text" x={TX} y={n.ty}>{#each n.lines as l, i (i)}<tspan x={TX} dy={i ? LH : 0}>{l}</tspan>{/each}</text>
		{/each}
		<line class="key__leader" x1={bx + cell / 2} y1={y0 + n * cell + 2} x2={bx + cell / 2} y2={y0 + n * cell + 14} />
		<circle class="key__anchor" cx={bx + cell / 2} cy={y0 + n * cell + 2} r="1.5" />
		<text class="key__text" x={x0 - 2} y={y0 + n * cell + 26}><tspan x={x0 - 2}>a vector, here the bias b_q, is one column,</tspan><tspan x={x0 - 2} dy={LH}>set beside the matrix it is added to</tspan></text>
		<text class="key__text" x={x0 - 2} y={y0 + n * cell + 64}><tspan x={x0 - 2}>Rows run top to bottom and columns left</tspan><tspan x={x0 - 2} dy={LH}>to right, as written. Area ∝ |θᵢ|, on one</tspan><tspan x={x0 - 2} dy={LH}>scale for the whole plate.</tspan></text>
	{/snippet}
</HowToRead>

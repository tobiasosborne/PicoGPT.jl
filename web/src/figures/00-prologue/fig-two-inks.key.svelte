<script>
	// How to read the two inks: four marks from the figure's own arrays, enlarged —
	// a positive and a negative score z (value ink), and a positive and a negative entry
	// of its adjoint z̄ = p − e_y (covector ink). First covector marks of the book.
	import { getContext } from 'svelte';
	import HowToRead from '../primitives/HowToRead.svelte';
	import { fmt, clamp, showChar } from '../primitives/scales.js';

	let { attrs = {} } = $props();
	const specimen = $derived(getContext('specimen')());
	const b = $derived(clamp((+attrs.window || 2) - 1, 0, specimen.lab.train.x.length - 1));
	const pos = $derived(clamp((+attrs.pos || 1) - 1, 0, specimen.config.T - 1));
	const win = $derived(specimen.lab.gradTrace(specimen.theta).windows[b]);
	const z = $derived(win.val.Z[pos]);
	const p = $derived(win.val.probs[pos]);
	const y = $derived(win.y[pos]); // the engine's prepared batch holds 0-based ids
	const zbar = $derived(p.map((v, a) => v - (a === y ? 1 : 0)));
	const pick = (arr, sgn) => arr.map((v, a) => ({ v, a })).filter((o) => o.v * sgn > 0).sort((u, w) => sgn * (w.v - u.v))[0];
	const marks = $derived([
		{ o: pick(z, 1), ink: 'v', sym: 'z' },
		{ o: pick(z, -1), ink: 'v', sym: 'z' },
		{ o: pick(zbar, 1), ink: 'c', sym: 'z̄' },
		{ o: pick(zbar, -1), ink: 'c', sym: 'z̄' }
	].filter((m) => m.o));
	const S = 26, X = 28, TX = 74, LH = 13;
	const ys = [30, 74, 134, 178];
	const ch = (a) => showChar(specimen.vocab[a]);
</script>

<HowToRead width={250} height={222} notes={[]}>
	{#snippet glyph()}
		<text class="key__text" x={X - 13} y={14} font-weight="600">value ink: computed going forward</text>
		<text class="key__text" x={X - 13} y={118} font-weight="600">covector ink: pulled back from the loss</text>
		{#each marks as m, k (k)}
			{@const y0 = ys[k]}
			{@const neg = m.o.v < 0}
			<rect class="matrix__frame" x={X - S / 2} y={y0 - S / 2} width={S} height={S} />
			<rect class={m.ink === 'v' ? 'fx-key-v' : 'fx-key-c'} class:neg={neg} x={X - S * 0.42} y={y0 - S * 0.42} width={S * 0.84} height={S * 0.84} />
			<line class="key__leader" x1={X + S / 2 + 2} y1={y0} x2={TX - 4} y2={y0} />
			<circle class="key__anchor" cx={X + S / 2 + 2} cy={y0} r="1.5" />
			<text class="key__text" x={TX} y={y0 - 2}><tspan x={TX}>{m.ink === 'v' ? (neg ? 'open: negative' : 'solid: positive') : neg ? 'open: negative' : 'hatched: positive'}</tspan><tspan x={TX} dy={LH}>{m.sym} at ‘{ch(m.o.a)}’ = {fmt(m.o.v, 2)}</tspan></text>
		{/each}
		<text class="key__text" x={X - 13} y={212}>Each array has its own scale: area ∝ |entry|.</text>
	{/snippet}
</HowToRead>

<style>
	.fx-key-v { fill: var(--value); }
	.fx-key-v.neg { fill: none; stroke: var(--value); stroke-width: 0.9px; }
	.fx-key-c { fill: url(#hatch-covector); stroke: var(--covector); stroke-width: 0.9px; }
	.fx-key-c.neg { fill: none; }
</style>

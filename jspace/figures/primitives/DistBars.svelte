<script module>
	/** Width of a DistBars glyph with V bars. */
	export const distBarsWidth = (V, bw, gap) => V * bw + (V - 1) * gap;
</script>

<script>
	// A distribution over the V characters as bars from a common baseline, height ∝ p
	// (p = 1 reaches h), in vocabulary order. The bar named by `mark` (by default the
	// most probable) carries its character above it, with its probability if `pLabel`.
	// Optional: the characters under the bars (`chars`); the character that actually
	// comes next (`target`) underlined there (or just under its bar, without `chars`). `markMuted` sets the marked character in grey
	// (e.g. a lens's choice that disagrees with the model's). Value ink: a distribution is computed
	// going forward. `muted` greys the whole glyph (an unselected small multiple).
	import { fmtProb, showChar } from '#figures/primitives/scales.js';
	import { argmax } from './jdata.js';

	let {
		values = [],
		x = 0,
		y = 0,
		bw = 8,
		gap = 2,
		h = 40,
		chars = null,
		vocab = null,
		mark = null,
		markLabel = true,
		markMuted = false,
		pLabel = false,
		target = -1,
		muted = false,
		charSize = 12
	} = $props();

	const top = $derived(mark ?? argmax(values));
	const bx = (k) => x + k * (bw + gap);
	const width = $derived(values.length * bw + (values.length - 1) * gap);
</script>

<g class="dbars" class:dbars--muted={muted}>
	<line class="dbars__base" x1={x - 1.5} x2={x + width + 1.5} y1={y} y2={y} />
	{#each values as p, k (k)}
		<rect class="bar--value dbars__bar" x={bx(k)} y={y - Math.max(0, p) * h} width={bw} height={Math.max(0, p) * h} />
	{/each}
	{#if markLabel && vocab && values.length}
		{@const p = values[top]}
		<text class="tok dbars__top" class:dbars__top--muted={markMuted} x={bx(top) + bw / 2} y={y - Math.max(0, p) * h - 4} text-anchor={pLabel ? 'end' : 'middle'} dx={pLabel ? bw / 2 + 1 : 0} font-size={charSize + 2}>{showChar(vocab[top])}</text>
		{#if pLabel}
			<text class="lbl lbl--micro" x={bx(top) + bw + 3} y={y - Math.max(0, p) * h - 4}>{fmtProb(p).replace(/^0\./, '.')}</text>
		{/if}
	{/if}
	{#if chars}
		{#each chars as ch, k (k)}
			<text class="tok dbars__ch" class:dbars__ch--target={k === target} x={bx(k) + bw / 2} y={y + charSize + 1} text-anchor="middle" font-size={charSize}>{showChar(ch)}</text>
		{/each}
		{#if target >= 0}
			<line class="dbars__target" x1={bx(target) - 1} x2={bx(target) + bw + 1} y1={y + charSize + 4} y2={y + charSize + 4} />
		{/if}
	{:else if target >= 0}
		<line class="dbars__target" x1={bx(target) - 0.5} x2={bx(target) + bw + 0.5} y1={y + 3} y2={y + 3} />
	{/if}
</g>

<style>
	.dbars__base { stroke: var(--ink-3); stroke-width: var(--hair); }
	.dbars__top { fill: var(--ink); font-weight: 600; }
	.dbars__top--muted { fill: var(--ink-3); font-weight: 400; }
	.dbars__ch { fill: var(--ink-3); }
	.dbars__ch--target { fill: var(--ink); font-weight: 600; }
	.dbars__target { stroke: var(--ink); stroke-width: 1px; }
	.dbars--muted .dbars__bar { fill: var(--ink-3); opacity: 0.45; }
	.dbars--muted .dbars__top { fill: var(--ink-3); font-weight: 400; }
</style>

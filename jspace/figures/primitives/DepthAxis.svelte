<script module>
	/** Vertical room the axis takes below its line, for the options given. */
	export function depthAxisHeight({ words = true, blocks = true } = {}) {
		return 20 + (words ? 12 : 0) + (blocks ? 22 : 0);
	}
</script>

<script>
	// The depth axis of the residual stream: a hairline with one tick per depth
	// ℓ = 0, ½, 1, …, L. Depth 0 is the stream after the embedding; ℓ − ½ is after the
	// attention update of block ℓ; ℓ is after its MLP update. Optional micro labels name
	// the update that has just been added ("embed", "attn", "MLP"), and brackets beneath
	// group the two half-steps of each block. Ticks are evenly spaced, as the depths are.
	// The axis carries no data of its own; the marks placed on it do. Interactive: a hit
	// area per depth calls onpick(j) (keyboard: Enter).
	import { depthLabel, depthScale } from './jdata.js';

	let {
		depths = [],
		kinds = [],
		x = 0,
		width = 300,
		y = 0,
		sel = -1,
		words = true,
		blocks = true,
		title = 'depth',
		onpick = null
	} = $props();

	const n = $derived(depths.length);
	const pos = $derived(depthScale(n, x, x + width));
	const step = $derived(n > 1 ? width / (n - 1) : width);
	const showWords = $derived(words && step >= 30);
	const short = { embed: 'embed', attention: 'attn', mlp: 'MLP' };
	const L = $derived(Math.round(depths.at(-1) ?? 0));
	const yb = $derived(y + 20 + (showWords ? 12 : 0) + 6);
</script>

<g class="daxis">
	<line class="axis" x1={x} x2={x + width} y1={y} y2={y} />
	{#if title}<text class="lbl lbl--micro" x={x - 10} y={y + 14} text-anchor="end">{title}</text>{/if}
	{#each depths as d, j (j)}
		<line class="axis" x1={pos(j)} x2={pos(j)} y1={y} y2={y + 4} />
		<text class="lbl daxis__tick" class:lbl--strong={j === sel} x={pos(j)} y={y + 16} text-anchor="middle">{depthLabel(d)}</text>
		{#if showWords && kinds[j]}
			<text class="lbl lbl--micro" x={pos(j)} y={y + 28} text-anchor="middle">{short[kinds[j]] ?? kinds[j]}</text>
		{/if}
		{#if onpick}
			<rect class="hit" x={pos(j) - step / 2} y={y - 6} width={step} height={30} role="button" tabindex="0" aria-label="depth {depthLabel(d)}" onclick={() => onpick(j)} onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onpick(j); } }} />
		{/if}
	{/each}
	{#if blocks && L >= 1}
		{#each { length: L } as _, b (b)}
			{@const x1 = pos(2 * b) + 4}
			{@const x2 = pos(2 * b + 2) - 4}
			<path class="leader" d="M{x1} {yb - 3} v3 H{x2} v-3" />
			<text class="lbl lbl--micro" x={(x1 + x2) / 2} y={yb + 12} text-anchor="middle">block {b + 1}</text>
		{/each}
	{/if}
</g>

<style>
	.daxis__tick { font-size: 11.5px; }
</style>

<script module>
	export const meta = { size: 'wide', title: 'A character becomes a vector' };
	export { default as Key } from './fig-one-hot.key.svelte';
</script>

<script>
	// Figure: what is a character to the model?  Laid out as the product is written
	// (the "Falk scheme"): the one-hot vector e_a as a row of V slots above the columns
	// of E ∈ ℝ^{D×V}; the product E e_a = E[:, a] beside the rows; and the same column
	// as a spoke glyph — the first spoke glyph of the book (its key is exported here).
	// State: the character a (attr char=<1-based id>). Click a character or use ←/→.
	import { getContext } from 'svelte';
	import { measure } from '../primitives/measure.js';
	import MatrixGlyph from '../primitives/MatrixGlyph.svelte';
	import SpokeGlyph from '../primitives/SpokeGlyph.svelte';
	import { fmt, showChar, clamp, maxAbs, spokeScale, layoutWidth } from '../primitives/scales.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const V = $derived(specimen.config.V);
	const D = $derived(specimen.config.D);
	// svelte-ignore state_referenced_locally
	let a = $state(clamp((+attrs.char || 5) - 1, 0, specimen.config.V - 1));
	let hover = $state(null); // {i, j}
	let width = $state(868);
	const interactive = $derived(mode !== 'print');

	const Eblock = $derived(specimen.params().find((p) => p.name === 'E'));
	const Ev = $derived(Eblock.values); // column-major D×V
	const at = (i, j) => Ev[i + D * j];
	const column = $derived(Array.from({ length: D }, (_, i) => at(i, a)));
	const vmax = $derived(maxAbs(Ev));

	// geometry
	const W = $derived(layoutWidth(width, mode, 330));
	const narrow = $derived(W < 640);
	const cell = $derived(narrow ? Math.min(30, (W - 80) / (V + 2.4)) : 32);
	const mx = $derived(narrow ? 34 : Math.max(70, W * 0.1)); // matrix left edge
	const yChar = 50;
	const yHot = $derived(yChar + 30);
	const my = $derived(yHot + 34); // matrix top
	const mW = $derived(cell * V);
	const mH = $derived(cell * D);
	const cx = $derived(mx + mW + (narrow ? 26 : 58)); // the product column
	const rG = $derived(narrow ? Math.min(78, (W - 40) / 2.6) : Math.min(104, (W - cx - cell - 120) / 2.2));
	const gx = $derived(narrow ? W / 2 : Math.min(W - rG - 40, cx + cell + 120 + rG));
	const gy = $derived(narrow ? my + mH + 60 + rG : my + mH / 2);
	const sc = $derived(spokeScale(rG, vmax));
	const H = $derived(narrow ? gy + rG + 54 : my + mH + 58);

	function pick(j) { a = j; }
	function key(e) {
		if (!interactive) return;
		if (e.key === 'ArrowLeft') { a = (a + V - 1) % V; e.preventDefault(); }
		if (e.key === 'ArrowRight') { a = (a + 1) % V; e.preventDefault(); }
	}
	const sub = (k) => String(k);
	const chA = $derived(showChar(specimen.vocab[a]));
	const readout = $derived.by(() => {
		const parts = column.map((v) => fmt(v, 2)).join(', ');
		return { who: `‘${chA}’ is symbol ${a + 1} of ${V}.`, what: `column ${a + 1} of E: (${parts}).` };
	});
	const hoverText = $derived(hover ? `E[${hover.i + 1}, ${hover.j + 1}] = ${fmt(at(hover.i, hover.j), 3)}${hover.j === a ? '' : ` (column of ‘${showChar(specimen.vocab[hover.j])}’)`}` : '');
</script>

<div class="fx fx-onehot" use:measure={(w) => (width = w)}>
	<!-- svelte-ignore a11y_no_noninteractive_tabindex (composite figure supports arrow-key navigation) -->
	<svg
		viewBox="0 0 {W} {H}"
		width={W}
		height={H}
		role={interactive ? 'group' : 'img'}
		tabindex={interactive ? 0 : undefined}
		aria-label="One-hot vector of ‘{chA}’ and its embedding: {readout.what}"
		onkeydown={key}
	>
		<!-- the alphabet: one slot per symbol -->
		{#each specimen.vocab as ch, j (j)}
			{@const x = mx + cell * (j + 0.5)}
			<text class="tok fx-onehot__char" class:is-on={j === a} x={x} y={yChar} text-anchor="middle">{showChar(ch)}</text>
			<text class="lbl lbl--micro" x={x} y={yChar - 22} text-anchor="middle">{j + 1}</text>
			<!-- e_a: 1 in slot a, 0 elsewhere -->
			{#if j === a}
				<circle class="dot--value" cx={x} cy={yHot} r="5.2" />
			{:else}
				<circle class="fx-onehot__zero" cx={x} cy={yHot} r="2.2" />
			{/if}
			{#if interactive}
				<rect class="hit" x={x - cell / 2} y={yChar - 30} width={cell} height={yHot - yChar + 44} role="button" tabindex="0" aria-label="character {showChar(ch)}" onclick={() => pick(j)} onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(j); } }} />
			{/if}
		{/each}
		<text class="m" x={mx - 12} y={yHot + 5} text-anchor="end">e<tspan class="sub" dy="4">{a + 1}</tspan></text>
		<text class="lbl lbl--micro" x={mx + mW + 10} y={yHot + 4}>one-hot: a single 1</text>

		<!-- the selected column, bracketed through the matrix -->
		<rect class="fx-onehot__col" x={mx + cell * a + 1} y={yHot + 10} width={cell - 2} height={my + mH - yHot - 6} />

		<!-- E, as written -->
		<g class="fx-onehot__E">
			<MatrixGlyph get={at} m={D} n={V} x={mx} y={my} {cell} {vmax} grid />
		</g>
		<g class="fx-onehot__mute">
			{#each Array.from({ length: V }) as _, j (j)}
				{#if j !== a}<rect class="fx-onehot__veil" x={mx + cell * j} y={my} width={cell} height={mH} />{/if}
			{/each}
		</g>
		{#each Array.from({ length: D }) as _, i (i)}
			<text class="lbl lbl--micro" x={mx - 8} y={my + cell * (i + 0.5) + 3.5} text-anchor="end">{i + 1}</text>
		{/each}
		<text class="m fx-onehot__Ename" x={mx - (narrow ? 22 : 36)} y={my + mH / 2 + 6} text-anchor="end">E</text>
		{#if !narrow}
			<text class="lbl lbl--micro" x={mx - 36} y={my + mH / 2 + 22} text-anchor="end">{D} × {V}</text>
		{/if}

		<!-- the product: E e_a = E[:, a] -->
		<text class="lbl fx-onehot__eq" x={(mx + mW + cx) / 2} y={my + mH / 2 + 4} text-anchor="middle">=</text>
		<MatrixGlyph get={(i) => column[i]} m={D} n={1} x={cx} y={my} {cell} {vmax} grid />
		<text class="lbl" x={cx + cell / 2} y={my + mH + 20} text-anchor="middle"><tspan class="m m--s">E e</tspan><tspan class="m m--s sub" dy="3">{a + 1}</tspan></text>

		<!-- the same column as a spoke glyph -->
		{#if !narrow}
			<line class="leader" x1={cx + cell + 10} x2={gx - rG - 30} y1={gy} y2={gy} />
			<text class="lbl lbl--micro" x={(cx + cell + gx - rG - 20) / 2} y={gy - 7} text-anchor="middle">drawn as</text>
		{/if}
		<SpokeGlyph values={column} x={gx} y={gy} rmax={sc.rmax} scale={sc.scale} weight={1.7} tip={3.4} />
		{#each column as _, k (k)}
			{@const ang = -Math.PI / 2 + (2 * Math.PI * k) / D}
			<text class="lbl lbl--micro fx-onehot__idx" x={gx + Math.cos(ang) * (sc.rmax + 13)} y={gy + Math.sin(ang) * (sc.rmax + 13) + 3.5} text-anchor="middle">{sub(k + 1)}</text>
		{/each}
		<text class="lbl" x={gx} y={gy + sc.rmax + 36} text-anchor="middle">column {a + 1} of <tspan class="m m--s">E</tspan>: what the model means by ‘{chA}’</text>

		<!-- hover: read one entry -->
		{#if interactive}
			{#each Array.from({ length: D * V }) as _, k (k)}
				{@const i = k % D}
				{@const j = Math.floor(k / D)}
				<rect class="hit" x={mx + cell * j} y={my + cell * i} width={cell} height={cell} role="button" aria-label="read this entry; details in the live readout" tabindex="0" onpointerenter={() => (hover = { i, j })} onfocus={() => (hover = { i, j })} onpointerleave={() => (hover = null)} onblur={() => (hover = null)} onclick={() => pick(j)} onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(j); } }} />
			{/each}
		{/if}
	</svg>
	{#if interactive}
		<p class="readout" aria-live="polite">
			<span class="readout__who">{readout.who}</span>
			<span class="readout__part">{#if hover}{hoverText}.{:else}<i>E e</i><sub>{a + 1}</sub> is {readout.what}{/if}</span>
			<span class="readout__hint">Click a character (or use ←/→); point at a square to read it.</span>
		</p>
	{/if}
</div>

<style>
	.fx-onehot :global(.m) { font-family: var(--serif); font-style: italic; font-size: 17px; fill: var(--ink); }
	.fx-onehot :global(.m--s) { font-size: 14px; }
	.fx-onehot :global(.sub) { font-size: 10.5px; }
	.fx-onehot__Ename { font-size: 26px !important; }
	.fx-onehot__char { font-size: 25px; fill: var(--ink-3); transition: fill var(--t-quick); }
	.fx-onehot__char.is-on { fill: var(--ink); font-weight: 600; }
	.fx-onehot__zero { fill: none; stroke: var(--ink-3); stroke-width: var(--hair); }
	.fx-onehot__col { fill: none; stroke: var(--ink); stroke-width: 0.8px; transition: x var(--t-move) var(--ease); }
	.fx-onehot__veil { fill: var(--paper); opacity: 0.62; }
	.fx-onehot__eq { font-size: 18px; fill: var(--ink-3); }
	.fx-onehot__idx { fill: var(--ink-3); }
</style>

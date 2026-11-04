<script module>
	export const meta = { size: 'wide', title: 'The whole training set' };
</script>

<script>
	// Figure: what exactly is the training set? The corpus on top; below, one pair of rows
	// per window b: its T inputs in ink, and beneath them, shifted one place right, its T
	// targets in value ink. Every glyph sits in the column of its corpus position, so the
	// staircase of windows is visible, and every target is the corpus character after
	// its input. Each target is one prediction: one of the N = T·B columns.
	// Data: the adapter's training batch (lab.train: 1-based ids x, y per window).
	import { getContext } from 'svelte';
	import { measure } from '../primitives/measure.js';
	import { showChar, layoutWidth } from '../primitives/scales.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const T = $derived(specimen.config.T);
	const train = $derived(specimen.lab.train);
	const B = $derived(train.x.length);
	const m = $derived(specimen.corpus.length);
	const ch = (id) => showChar(specimen.vocab[id - 1]);
	let hover = $state(null); // {b, t}
	let width = $state(868);
	const interactive = $derived(mode !== 'print');

	// geometry
	const W = $derived(layoutWidth(width, mode, 330));
	const narrow = $derived(W < 600);
	const padL = $derived(narrow ? 30 : 86);
	const padR = $derived(narrow ? 8 : 96);
	const cw = $derived((W - padL - padR) / m);
	const xc = (k) => padL + cw * (k + 0.5); // corpus position k (0-based)
	const yCorpus = 30;
	const rowH = $derived(narrow ? 13 : 16);
	const pairH = $derived(2 * rowH + (narrow ? 5 : 7));
	const y0 = $derived(yCorpus + 34);
	const yIn = (b) => y0 + pairH * b;
	const yOut = (b) => yIn(b) + rowH;
	const H = $derived(y0 + pairH * B + 34);
	const fsC = $derived(narrow ? 15 : 21);
	const fs = $derived(narrow ? 11.5 : 14);

	const n = (b, t) => b * T + t + 1; // column index, 1-based
	const readout = $derived.by(() => {
		if (!hover) return null;
		const { b, t } = hover;
		const ctx = train.x[b].slice(0, t + 1).map(ch).join('');
		return `Window ${b + 1}, position ${t + 1}: from the context “${ctx}”, predict ‘${ch(train.y[b][t])}’. This is column ${n(b, t)} of ${B * T}.`;
	});
</script>

<div class="fx fx-win" use:measure={(w) => (width = w)}>
	<svg viewBox="0 0 {W} {H}" width={W} height={H} role="img" aria-label="The training set: {B} windows of {T} characters, each with its targets shifted by one: {B * T} predictions.">
		<!-- guides: one per corpus position -->
		{#each Array.from({ length: m }) as _, k (k)}
			<line class="fx-win__guide" x1={xc(k)} x2={xc(k)} y1={yCorpus + 8} y2={y0 + pairH * B - 6} />
		{/each}
		<!-- the corpus -->
		{#each Array.from(specimen.corpus) as c, k (k)}
			<text class="tok" x={xc(k)} y={yCorpus} text-anchor="middle" font-size={fsC}>{showChar(c)}</text>
		{/each}
		{#if !narrow}<text class="lbl lbl--micro" x={padL - 12} y={yCorpus - 2} text-anchor="end">corpus</text>{/if}

		{#each train.x as xs, b (b)}
			{@const ys = train.y[b]}
			{#if hover && hover.b === b}
				<rect class="fx-win__ctx" x={xc(b) - cw / 2} y={yIn(b) - rowH + 3} width={cw * (hover.t + 1)} height={rowH} />
			{/if}
			{#each xs as id, t (t)}
				<text class="tok fx-win__in" class:is-dim={hover && hover.b === b && t > hover.t} x={xc(b + t)} y={yIn(b)} text-anchor="middle" font-size={fs}>{ch(id)}</text>
			{/each}
			{#each ys as id, t (t)}
				<text class="tok fx-win__out" class:is-on={hover && hover.b === b && hover.t === t} class:is-dim={hover && hover.b === b && hover.t !== t} x={xc(b + t + 1)} y={yOut(b)} text-anchor="middle" font-size={fs}>{ch(id)}</text>
			{/each}
			<text class="lbl lbl--micro" x={xc(b) - cw * 0.9} y={yIn(b)} text-anchor="end">{narrow ? b + 1 : `window ${b + 1}`}</text>
			{#if !narrow}
				<text class="lbl lbl--micro fx-win__n" x={W - padR + 14} y={yOut(b)}>columns {n(b, 0)}–{n(b, T - 1)}</text>
			{/if}
			{#if interactive}
				{#each ys as _, t (t)}
					<rect class="hit" x={xc(b + t + 1) - cw / 2} y={yIn(b) - rowH + 2} width={cw} height={pairH} role="presentation" onpointerenter={() => (hover = { b, t })} onpointerleave={() => (hover = null)} />
				{/each}
			{/if}
		{/each}
		{#if !narrow}
			<text class="lbl lbl--micro" x={xc(0) - cw * 0.9} y={yOut(0)} text-anchor="end"><tspan class="fx-win__vlbl">targets</tspan></text>
		{/if}
		<text class="lbl" x={W - 4} y={H - 10} text-anchor="end"><tspan class="m">N</tspan> = <tspan class="m">T</tspan>·<tspan class="m">B</tspan> = {T} · {B} = {T * B} predictions</text>
	</svg>
	{#if interactive}
		<p class="readout" aria-live="polite">
			{#if readout}<span class="readout__part">{readout}</span>{:else}<span class="readout__part">{B} windows of {T}; {T * B} predictions in all.</span>{/if}
			<span class="readout__hint">Point at a target to see the context it is predicted from.</span>
		</p>
	{/if}
</div>

<style>
	.fx-win :global(.m) { font-family: var(--serif); font-style: italic; font-size: 14px; fill: var(--ink); }
	.fx-win__guide { stroke: var(--rule-2); stroke-width: var(--hair); }
	.fx-win__in { fill: var(--ink); }
	.fx-win__out { fill: var(--value); }
	.fx-win__out.is-on { font-weight: 700; fill: var(--value-deep); }
	.is-dim { opacity: 0.35; }
	.fx-win__ctx { fill: var(--value-tint); }
	.fx-win__vlbl { fill: var(--value-deep); }
	.fx-win__n { fill: var(--ink-3); }
</style>

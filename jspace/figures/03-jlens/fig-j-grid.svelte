<script module>
	export const meta = { size: 'wide', title: 'The J-lens read along a real text, depth by depth' };
	export { default as Key } from './fig-j-grid.key.svelte';
</script>

<script>
	// Figure: what does the J-lens read along a real text, depth by depth?
	// The Shakespeare model on the exchange between Gremio and Baptista (held-out text). The
	// text is set across the top, one column per position (the column is the token); under
	// each character, one cell per depth 0, ½, …, 4 (forward runs down), holding the character
	// the lens ranks first, its weight of type ∝ its probability; at the bottom, the character
	// that actually comes next. At depth 4 the J-lens is the model's own readout (J_L = I).
	// Web: the J-lens, or the logit lens (click its name); point at a cell for its top three.
	// Print: both lenses, as two plates, J-lens first.
	// Data: jspace.shakespeare.lenses.prompt.{text, ids[rank, t, j, lens], probabilities,
	// model_top1}; lens 1 = J, 2 = logit. Every number from the export.
	import { getContext } from 'svelte';
	import { measure } from '#figures/primitives/measure.js';
	import { fmtProb, layoutWidth } from '#figures/primitives/scales.js';
	import LensGrid, { lensGridLayout, gridChar } from '../primitives/LensGrid.svelte';
	import { at, depthLabel } from '../primitives/jdata.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const sh = $derived(getContext('artifacts')()?.jspace?.shakespeare ?? null);
	const interactive = $derived(mode !== 'print');
	// svelte-ignore state_referenced_locally
	let lens = $state(attrs.lens === 'logit' ? 1 : 0);
	let sel = $state(null);
	let width = $state(868);
	const W = $derived(layoutWidth(width, mode, 330));
	const narrow = $derived(W < 600);

	const pr = $derived(sh?.lenses.prompt ?? null);
	const vocab = $derived(sh?.vocab ?? []);
	const depths = $derived(sh?.depth ?? []);
	const text = $derived(pr?.text ?? '');
	const N = $derived(Array.from(text).length);
	const grid = (l) => depths.map((_, j) => Array.from({ length: N }, (_, t) => ({ ch: vocab[at(pr.ids, 0, t, j, l) - 1], p: at(pr.probabilities, 0, t, j, l) })));
	const plates = $derived(pr ? (interactive ? [lens] : [0, 1]) : []);
	const rowLabels = $derived(depths.map((d, j) => (j === depths.length - 1 ? `${depthLabel(d)} = model` : depthLabel(d))));

	// print sets both plates on one page: smaller cells, so that each plate is two blocks of lines
	const size = $derived(interactive ? (narrow ? 12 : 14) : 11.2);
	const cellH = $derived(interactive ? (narrow ? 16 : 18) : 12.5);
	const layout = $derived(lensGridLayout({ text, width: W, minCell: interactive ? (narrow ? 15 : 17) : 12.5, maxCell: 24, labelW: narrow ? 62 : 70, rows: depths.length, cellH }));
	const head = 34;
	const plateH = $derived(head + layout.height + 22);
	const H = $derived(plates.length * plateH);
	const names = ['J-lens', 'logit lens'];

	const readout = $derived.by(() => {
		if (!pr) return null;
		const s = sel ?? { r: depths.length - 4, t: 3 };
		const l = lens;
		const top = [0, 1, 2].map((k) => `‘${gridChar(vocab[at(pr.ids, k, s.t, s.r, l) - 1])}’ ${fmtProb(at(pr.probabilities, k, s.t, s.r, l), 2, true)}`).join(', ');
		const chars = Array.from(text);
		return {
			who: `‘${gridChar(chars[s.t])}’, position ${s.t + 1}, depth ${depthLabel(depths[s.r])}:`,
			body: `the ${names[l]} reads ${top}; the next character is ‘${gridChar(chars[s.t + 1] ?? '')}’, and the model’s own first choice ‘${gridChar(vocab[pr.model_top1[s.t] - 1])}’.`
		};
	});
	const aria = $derived(pr ? `The ${plates.map((l) => names[l]).join(' and the ')} along “${text.replace(/\n/g, ' ¶ ')}”, nine depths; the J-lens at depth 4 equals the model.` : '');
</script>

<div class="fx fx-grid" use:measure={(w) => (width = w)}>
	{#if pr}
		<svg viewBox="0 0 {W} {H}" width={W} height={H} role={interactive ? 'group' : 'img'} aria-label={aria}>
			{#each plates as l, i (l)}
				{@const y0 = i * plateH}
				{#if interactive}
					{#each names as nm, k (k)}
						<text class="lbl fx-grid__lens" class:is-on={k === lens} x={k ? 96 : 0} y={y0 + 14} role="button" tabindex="0" aria-pressed={k === lens} onclick={() => (lens = k)} onkeydown={(e) => (e.key === 'Enter' || e.key === ' ') && (lens = k)}>{nm}</text>
					{/each}
				{:else}
					<text class="lbl lbl--head" x={0} y={y0 + 14}>{names[l]}</text>
				{/if}
				<text class="lbl lbl--micro" x={W} y={y0 + 14} text-anchor="end">depth ↓ · weight of type ∝ probability</text>
				<LensGrid {text} cells={grid(l)} {rowLabels} {layout} x={0} y={y0 + head} {size} sel={interactive ? sel : null} onpoint={interactive ? (r, t) => (sel = { r, t }) : null} />
			{/each}
		</svg>
		{#if interactive && readout}
			<p class="readout" aria-live="polite">
				<span class="readout__who">{readout.who}</span>
				<span class="readout__part">{readout.body}</span>
				<span class="readout__hint">Point at a cell for the lens’s top three; click “logit lens” to compare.</span>
			</p>
		{/if}
	{:else}
		<p class="fig__pending">The J-lens export (jspace/artifacts/shakespeare-jlens.json) is not available.</p>
	{/if}
</div>

<style>
	.fx-grid__lens { cursor: pointer; fill: var(--ink-3); font-weight: 500; }
	.fx-grid__lens.is-on { fill: var(--ink); font-weight: 650; text-decoration: underline; text-underline-offset: 3px; }
</style>

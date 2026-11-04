<script module>
	export const meta = { size: 'wide', title: 'The attention matrices of both heads' };
	export { default as Key } from './fig-attention-triangle.key.svelte';
</script>

<script>
	// Figure: the attention matrix as a lower triangle, for both heads (small multiples),
	// and the forward computation that makes it: t ∈ [0,1] runs
	//   scores S (every pair, signed)  →  the mask removes the upper triangle  →  softmax per row.
	// Interactive: scrub t; hover/click a row to follow one reader across both heads.
	// Print: the animation becomes small multiples of its three phases.
	import { getContext } from 'svelte';
	import { measure } from '../primitives/measure.js';
	import TriangleDots from '../primitives/TriangleDots.svelte';
	import Scrubber from '../primitives/Scrubber.svelte';
	import { fmt, fmtProb, showChar, clamp, phase, maxAbs, layoutWidth } from '../primitives/scales.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const T = $derived(specimen.config.T);
	// svelte-ignore state_referenced_locally
	const start = clamp((+attrs.window || 10) - 1, 0, specimen.windows.length - 1);
	const tr = $derived(specimen.windowTrace(start));

	// svelte-ignore state_referenced_locally
	let t = $state(attrs.t != null ? +attrs.t : 1);
	// films and the gallery can drive the clock from outside
	$effect(() => { if (attrs.t != null) t = +attrs.t; });
	let row = $state(-1);
	let pinned = $state(-1);
	let width = $state(868);
	const interactive = $derived(mode !== 'print');
	const W = $derived(layoutWidth(width, mode));

	const PHASES = [
		{ at: 0, label: 'scores S' },
		{ at: 0.42, label: 'mask' },
		{ at: 0.72, label: 'softmax' },
		{ at: 1, label: 'A' }
	];
	// one scale for the scores of BOTH heads; probabilities have their own (0…1)
	const sMax = $derived(maxAbs(tr.heads.map((h) => h.S)));

	/** The matrix drawn at time t: radii interpolate from |S| to A; returns values in "A units". */
	function frame(h, t) {
		const uMask = phase(t, 0.3, 0.55);
		const uSoft = phase(t, 0.6, 0.95);
		const S = tr.heads[h].S, A = tr.heads[h].A;
		// signed score s/sMax (open ring if negative) morphs into the probability A_ij
		const M = S.map((r, i) => r.map((s, j) => (j > i ? s / sMax : s / sMax + (A[i][j] - s / sMax) * uSoft)));
		return { M, upper: 1 - uMask, uSoft };
	}

	const active = $derived(pinned >= 0 ? pinned : row);
	function onrow(i, kind) {
		if (!interactive) return;
		if (kind === 'select') pinned = pinned === i ? -1 : i;
		else row = i;
	}

	// geometry
	const narrow = $derived(W < 620);
	const cell = $derived(narrow ? Math.min(30, (W - 48) / T) : Math.min(34, (W - 200) / (2 * T)));
	const gridW = $derived(cell * T);
	const labelW = 22;
	const gap = $derived(narrow ? 0 : Math.min(120, Math.max(56, W - 2 * (gridW + labelW) - 60)));
	const x0 = $derived(narrow ? labelW + 6 : Math.max(labelW + 6, (W - (2 * (gridW + labelW) + gap)) / 2 + labelW));
	const x1 = $derived(narrow ? x0 : x0 + gridW + gap + labelW);
	const top = 54;
	const y1 = $derived(narrow ? top + gridW + 64 : top);
	const H = $derived((narrow ? y1 : top) + gridW + 36);

	const phaseName = $derived(t < 0.3 ? 'scores' : t < 0.6 ? 'masked' : t < 0.95 ? 'softmax' : 'weights');
	const heading = $derived(
		{ scores: 'scores S = QᵀK/√d', masked: 'S + M: the future set to −∞', softmax: 'softmax of each row', weights: 'weights A' }[phaseName]
	);
	const readout = $derived.by(() => {
		if (active < 0) return null;
		const i = active;
		const ch = showChar(tr.chars[i]);
		return tr.heads.map((h, k) => {
			const best = h.A[i].slice(0, i + 1).map((w, j) => ({ w, j })).sort((a, b) => b.w - a.w)[0];
			return `head ${k + 1}: ‘${ch}’ at ${i + 1} reads ‘${showChar(tr.chars[best.j])}’ at ${best.j + 1} most (${fmtProb(best.w, 2, true)})`;
		});
	});

	// print: the three phases as small multiples
	const printPhases = [
		{ t: 0, label: 'scores S, every pair' },
		{ t: 0.58, label: 'S + M: the mask' },
		{ t: 1, label: 'A = softmax(S + M)' }
	];
	const pCell = $derived(Math.min(20, (W - 3 * 34) / (3 * T)));
	const pGap = $derived((W - 3 * (pCell * T + 20)) / 2);
</script>

{#snippet panel(h, x, y, c, tt, labels = true, focus = -1)}
	{@const f = frame(h, tt)}
	<g>
		{#if labels}
			{#each tr.chars as ch, i (i)}
				<text class="tok tok--axis" class:tok--on={i === focus} x={x - 9} y={y + i * c + c / 2 + 5} text-anchor="end">{showChar(ch)}</text>
				<text class="tok tok--axis" x={x + i * c + c / 2} y={y + T * c + 17} text-anchor="middle">{showChar(ch)}</text>
			{/each}
		{/if}
		<TriangleDots M={f.M} {x} {y} cell={c} vmax={1} signed upper={f.upper} row={focus} onrow={interactive && labels ? onrow : null} />
	</g>
{/snippet}

<div class="fx fx-tri" use:measure={(w) => (width = w)}>
	{#if interactive}
		<svg viewBox="0 0 {W} {H}" width={W} height={H} role={interactive ? 'group' : 'img'} aria-label="Attention matrices of head 1 and head 2 for the window “{tr.text}”, phase: {heading}">
			<text class="lbl lbl--phase" x={x0} y={14}>{heading}</text>
			<text class="lbl lbl--head" x={x0} y={38}>head 1</text>
			<text class="lbl lbl--head" x={x1} y={narrow ? y1 - 16 : 38}>head 2</text>
			{@render panel(0, x0, top, cell, t, true, active)}
			{@render panel(1, x1, y1, cell, t, true, active)}
			{#if active >= 0 && t > 0.95}
				{#each [0, 1] as h (h)}
					{@const x = h ? x1 : x0}
					{@const y = h ? y1 : top}
					<text class="lbl lbl--micro" x={x + (active + 1) * cell + 6} y={y + active * cell + cell / 2 + 4}>Σ = 1</text>
				{/each}
			{/if}
		</svg>
		<Scrubber bind:t phases={PHASES} label="forward computation of the attention matrix" duration={3600} />
		<p class="readout" aria-live="polite">
			{#if readout}
				<span class="readout__who">Reader ‘{showChar(tr.chars[active])}’, position {active + 1}.</span>
				{#each readout as r, k (k)}<span class="readout__part">{r}{k ? '.' : ';'}</span>{/each}
			{:else}
				<span class="readout__hint">Point at a row to follow one reader through both heads; click to keep it. Press play, or drag along the line, to run the computation.</span>
			{/if}
		</p>
	{:else}
		{@const PH = 2 * (pCell * T + 46) + 10}
		<svg viewBox="0 0 {W} {PH}" width={W} height={PH} role={interactive ? 'group' : 'img'} aria-label="The attention computation for both heads in three phases: scores, mask, softmax">
			{#each [0, 1] as h (h)}
				{#each printPhases as p, k (k)}
					{@const x = 20 + k * (pCell * T + 20 + pGap)}
					{@const y = 22 + h * (pCell * T + 46)}
					{#if h === 0}<text class="lbl" x={x} y={12}>{p.label}</text>{/if}
					{#if k === 0}<text class="lbl lbl--head" x={x - 14} y={y + (pCell * T) / 2} text-anchor="end" transform="rotate(-90 {x - 14} {y + (pCell * T) / 2})">head {h + 1}</text>{/if}
					{@render panel(h, x, y, pCell, p.t, false)}
				{/each}
			{/each}
		</svg>
	{/if}
</div>

<style>
	.fx-tri :global(.tok--axis) { font-size: 15px; fill: var(--ink-2); }
	.fx-tri :global(.tok--on) { fill: var(--ink); font-weight: 650; }
</style>

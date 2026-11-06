<script module>
	export const meta = { size: 'wide', title: 'Do the lens vectors of different depths agree?' };
</script>

<script>
	// Figure: linear CKA between the J-lens vector sets of every pair of depths, both models.
	// CKA compares the V×V Gram matrices of the (centred) lens vectors, so it is blind to
	// rotations and scale of the stream. Values lie in a narrow range near 1, so the glyph
	// draws the dissimilarity: below the diagonal, a hatched square (covector ink: a statistic
	// of the J-lens covectors) of area ∝ 1 − CKA, one scale for both models; above the
	// diagonal, the CKA itself as a number. A block structure would show as small squares
	// inside blocks and large ones between them. Numbers: both exports' `cka`.
	import { getContext } from 'svelte';
	import { measure } from '#figures/primitives/measure.js';
	import { fmt, fmtDense, layoutWidth } from '#figures/primitives/scales.js';
	import { at, depthLabel } from '../primitives/jdata.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const art = $derived(getContext('artifacts')()?.jspace ?? {});
	const interactive = $derived(mode !== 'print');
	let width = $state(868);
	let focus = $state(null); // {m, i, j}

	const models = $derived(
		[
			{ key: 'specimen', name: 'the specimen', J: art.specimen },
			{ key: 'shakespeare', name: 'the Shakespeare model', J: art.shakespeare }
		].filter((m) => m.J)
	);
	const cka = (J, i, j) => at(J.cka, i, j);
	// one scale for every square of both models
	const dmax = $derived(Math.max(1e-9, ...models.flatMap((m) => m.J.depth.flatMap((_, i) => m.J.depth.map((__, j) => 1 - cka(m.J, i, j))))));

	const W = $derived(layoutWidth(width, mode, 340));
	const narrow = $derived(W < 620);
	const nMax = $derived(Math.max(1, ...models.map((m) => m.J.depth.length)));
	const cell = $derived(narrow ? Math.min(34, (W - 40) / nMax) : Math.min(40, (W - 120) / (nMax + 4)));
	const lab = 30;
	// panel origins: side by side (small first), or stacked
	const panels = $derived.by(() => {
		let x = lab + 6, y = 40;
		return models.map((m, k) => {
			const n = m.J.depth.length;
			const p = { ...m, n, x, y };
			if (narrow) y += n * cell + 70;
			else x += n * cell + lab + 70;
			return p;
		});
	});
	const H = $derived(narrow ? panels.at(-1).y + panels.at(-1).n * cell + 40 : 40 + nMax * cell + 40);
	const side = (v) => cell * 0.9 * Math.sqrt(Math.max(0, Math.min(1, v / dmax)));

	const readout = $derived.by(() => {
		if (!focus) return null;
		const m = panels[focus.m];
		return `${m.name}: CKA between depth ${depthLabel(m.J.depth[focus.i])} and depth ${depthLabel(m.J.depth[focus.j])} is ${fmt(cka(m.J, focus.i, focus.j), 3)}.`;
	});
</script>

<div class="fx fx-cka" use:measure={(w) => (width = w)}>
	{#if !models.length}
		<p class="fig__pending">The J-lens exports are missing.</p>
	{:else}
		<svg viewBox="0 0 {W} {H}" width={W} height={H} role={interactive ? 'group' : 'img'} aria-label="Linear CKA between the J-lens vectors of every pair of depths. {panels.map((m) => `${m.name}: smallest ${fmt(Math.min(...m.J.depth.flatMap((_, i) => m.J.depth.map((__, j) => cka(m.J, i, j)))), 2)}`).join('; ')}">
			{#each panels as m, k (m.key)}
				<text class="lbl lbl--head" x={m.x} y={m.y - 22}>{m.name}</text>
				{#each m.J.depth as d, i (i)}
					<text class="lbl cka__dl" x={m.x - 8} y={m.y + i * cell + cell / 2 + 4} text-anchor="end">{depthLabel(d)}</text>
					<text class="lbl cka__dl" x={m.x + i * cell + cell / 2} y={m.y + m.n * cell + 15} text-anchor="middle">{depthLabel(d)}</text>
					{#each m.J.depth as _, j (j)}
						{@const cx = m.x + j * cell + cell / 2}
						{@const cy = m.y + i * cell + cell / 2}
						{#if j < i}
							{@const s = side(1 - cka(m.J, i, j))}
							<rect class="cka__sq" x={cx - s / 2} y={cy - s / 2} width={s} height={s} />
						{:else if j > i}
							<text class="lbl lbl--micro cka__num" x={cx} y={cy + 3.5} text-anchor="middle">{fmtDense(cka(m.J, i, j), 2)}</text>
						{:else}
							<circle class="cka__diag" cx={cx} cy={cy} r="0.9" />
						{/if}
						{#if interactive}
							<rect class="hit" x={cx - cell / 2} y={cy - cell / 2} width={cell} height={cell} role="button" tabindex="-1" aria-label="depths {depthLabel(m.J.depth[i])} and {depthLabel(m.J.depth[j])}"
								onpointerenter={() => (focus = { m: k, i, j })} onpointerleave={() => (focus = null)} onkeydown={() => {}} />
						{/if}
					{/each}
				{/each}
				<line class="rule" x1={m.x} y1={m.y} x2={m.x + m.n * cell} y2={m.y + m.n * cell} />
				<text class="lbl lbl--micro" x={m.x + m.n * cell + 6} y={m.y + 10}>CKA</text>
				<text class="lbl lbl--micro" x={m.x - 8} y={m.y + m.n * cell + 30} text-anchor="start">area: 1 − CKA</text>
			{/each}
		</svg>
		{#if interactive}
			<p class="readout" aria-live="polite">
				{#if readout}<span class="readout__part">{readout}</span>{:else}<span class="readout__hint">Point at a square or a number for the pair of depths.</span>{/if}
			</p>
		{/if}
	{/if}
</div>

<style>
	.cka__sq { fill: url(#hatch-covector); stroke: var(--covector); stroke-width: var(--hair); }
	.cka__num { fill: var(--ink-2); }
	.cka__diag { fill: var(--ink-3); }
	.cka__dl { font-size: 11px; fill: var(--ink-2); }
</style>

<script module>
	export const meta = { size: 'wide', title: 'Computing the averaged Jacobian by pullbacks' };
</script>

<script>
	// Figure: how J is computed. Window 1 of the specimen, its 8 columns across and its depths
	// down (forward = down: depth 0 on top, depth 1 = L at the bottom; the value-ink stems are
	// the forward pass). A covector e_i is placed at the final depth, either in one target
	// column t′ or in every column at once, and pulled back up through the MLP (depth ½) and
	// attention (depth 0). What arrives in source column t at depth ℓ is a covector spoke glyph
	// (bar tips, covector ink):
	//   one target:     row i of K_{t′t}  (zero for t > t′: nothing is drawn)
	//   every column:   Σ_{t′≥t} row i of K_{t′t}   (linearity: one pullback for all targets)
	// The mean of the second over the sources is row i of this window's J⁽¹⁾_0; beside it the
	// grand J_0 (value ink), its row i drawn in covector ink: rows of J are pulled-back covectors.
	// Data: jspace.specimen.jacobians.blocks (window 1; divided by conventions.jacobian_factor)
	// and jacobians.averaged. The clock t runs the backward pass (covector Scrubber).
	// Interactive: click a bottom glyph (one target) or "every column"; click a row of J to
	// choose i; scrub. Print: both seeds, side by side in time (stacked), at t = 1.
	import { getContext } from 'svelte';
	import { measure } from '#figures/primitives/measure.js';
	import SpokeGlyph from '../primitives/SpokeSparse.svelte';
	import MatrixGlyph from '#figures/primitives/MatrixGlyph.svelte';
	import Scrubber from '#figures/primitives/Scrubber.svelte';
	import { fmt, showChar, clamp, phase, spokeScale, layoutWidth } from '#figures/primitives/scales.js';
	import { at, rowsOf, depthLabel } from '../primitives/jdata.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const js = $derived(getContext('artifacts')()?.jspace?.specimen ?? null);
	const interactive = $derived(mode !== 'print');
	const D = $derived(specimen.config.D);
	const T = $derived(specimen.config.T);

	// svelte-ignore state_referenced_locally
	let i = $state(clamp((+attrs.row || 1) - 1, 0, 7));
	// svelte-ignore state_referenced_locally
	let target = $state(attrs.target === 'all' ? -1 : clamp((+attrs.target || 8) - 1, 0, 7)); // −1: every column
	// svelte-ignore state_referenced_locally
	let t = $state(attrs.t != null ? +attrs.t : 1);
	$effect(() => { if (attrs.t != null) t = +attrs.t; });
	let width = $state(868);
	const W = $derived(layoutWidth(width, mode, 330));
	const narrow = $derived(W < 640);

	const depths = $derived(js?.depth ?? []);
	const nd = $derived(depths.length);
	const factor = $derived(js?.conventions?.jacobian_factor ?? 1);
	const win = $derived((js?.sampling?.block_window ?? 1) - 1);
	const chars = $derived(Array.from(specimen.corpus.slice(win, win + T)));
	const Kr = (tp, s, j, row) => Array.from({ length: D }, (_, q) => at(js.jacobians.blocks, row, q, tp, s, j) / factor);
	const add = (a, b) => a.map((v, k) => v + b[k]);
	/** covectors at [depth j][source s] for seed `tg` (−1 = every column) and coordinate `row` */
	// (a covector that is exactly zero, as the blocks K_{t′t}, t′ ≠ t, after the last attention
	// are, arrives as nothing: null, and nothing is drawn)
	const nonzero = (g) => (g && g.some((v) => Math.abs(v) > 1e-12) ? g : null);
	function pulled(tg, row) {
		return depths.map((_, j) => Array.from({ length: T }, (_, s) => {
			if (tg >= 0) return s <= tg ? nonzero(Kr(tg, s, j, row)) : null;
			let g = new Array(D).fill(0);
			for (let tp = s; tp < T; tp++) g = add(g, Kr(tp, s, j, row));
			return nonzero(g);
		}));
	}
	const Jg = $derived(js ? rowsOf((a, q) => at(js.jacobians.averaged, a, q, 0), D) : []);
	const meanRow = (G) => G[0].reduce((acc, g) => add(acc, g ?? new Array(D).fill(0)), new Array(D).fill(0)).map((v) => v / T);

	const PHASES = [
		{ at: 0, label: `e${'ᵢ'} at depth 1` },
		{ at: 0.38, label: 'through the MLP' },
		{ at: 0.66, label: 'through attention' },
		{ at: 1, label: 'the mean' }
	];
	// how far the pullback has come: depth index j is reached at time arrive(j)
	const grow = (j, tt) => (j === nd - 1 ? 1 : j === 1 ? phase(tt, 0.12, 0.38) : phase(tt, 0.42, 0.66));
	const growMean = (tt) => phase(tt, 0.72, 0.98);

	// geometry
	const padL = $derived(narrow ? 26 : 92);
	const sideW = $derived(narrow ? 0 : Math.min(250, W * 0.28));
	const gridW = $derived(W - padL - sideW - (narrow ? 4 : 16));
	const colW = $derived(gridW / T);
	const rG = $derived(Math.min(colW * 0.4, 24));
	const cx = (s) => padL + colW * (s + 0.5);
	const rowGap = $derived(2 * rG + (narrow ? 30 : 40));
	const top = 30;
	const ry = (j) => top + rG + 14 + j * rowGap; // j = depth index: 0 on top
	const panelH = $derived(ry(nd - 1) + rG + 44);
	const jCell = $derived(narrow ? 9 : Math.min(12, (sideW - 40) / D));
	const sx = $derived(narrow ? padL : W - sideW + 22);

	// one spoke scale per depth row, explicitly labelled (the seeds and the sums differ by an
	// order of magnitude); the mean over sources shares the scale of depth 0, where it lives
	function view(tg, row) {
		const G = pulled(tg, row);
		const mean = tg < 0 ? meanRow(G) : null;
		const max = G.map((r) => Math.max(1e-12, ...r.flatMap((g) => (g ? g.map(Math.abs) : [0]))));
		return { G, mean, max, sc: max.map((m) => spokeScale(rG, m)) };
	}
	const sub = (n) => String(n).split('').map((c) => '₀₁₂₃₄₅₆₇₈₉'[+c]).join('');
	const live = $derived(js ? view(target, i) : null);
	const printViews = $derived(js && !interactive ? [{ tg: T - 1, ...view(T - 1, i) }, { tg: -1, ...view(-1, i) }] : []);

	const readout = $derived.by(() => {
		if (!js) return '';
		const norm = (g) => (g ? Math.hypot(...g) : 0);
		if (target >= 0) {
			const reached = live.G[0].map((g, s) => (g ? `${fmt(norm(g))} at ‘${showChar(chars[s])}’` : null)).filter(Boolean);
			return `e${sub(i + 1)} placed in column ${target + 1} (‘${showChar(chars[target])}’) of the final depth. Pulled back to depth 0 it gives, in each source column t ≤ ${target + 1}, row ${i + 1} of the block K for target ${target + 1} and source t; its lengths: ${reached.join(', ')}. Columns after ${target + 1} receive nothing.`;
		}
		const m = live.mean, g = Jg[i];
		const cos = m.reduce((s, v, k) => s + v * g[k], 0) / Math.max(1e-12, Math.hypot(...m) * Math.hypot(...g));
		return `e${sub(i + 1)} placed in every column at once. One pullback gives, in each source column t, the sum over all targets t′ ≥ t of row ${i + 1} of the blocks K. Their mean over the ${T} sources is row ${i + 1} of this window's J⁽¹⁾₀ (length ${fmt(Math.hypot(...m))}); row ${i + 1} of J₀ over all ten windows has length ${fmt(Math.hypot(...g))}, at cosine ${fmt(cos)} to it.`;
	});
	const act = (f) => (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); f(); } };
	function key(e) {
		if (e.key === 'ArrowUp') { i = Math.max(0, i - 1); e.preventDefault(); }
		else if (e.key === 'ArrowDown') { i = Math.min(D - 1, i + 1); e.preventDefault(); }
		else if (e.key === 'ArrowLeft') { target = target < 0 ? T - 1 : Math.max(0, target - 1); e.preventDefault(); }
		else if (e.key === 'ArrowRight') { target = target < 0 ? -1 : target === T - 1 ? -1 : target + 1; e.preventDefault(); }
	}
</script>

{#snippet panel(v, tg, tt, y0, title)}
	{@const G = v.G}
	<g transform="translate(0 {y0})">
		{#if title}<text class="lbl lbl--head" x={padL - (narrow ? 20 : 80)} y={12}>{title}</text>{/if}
		<!-- forward: the stems, value ink, down -->
		{#each { length: T } as _, s (s)}
			<line class="fx-pb__stem" x1={cx(s)} x2={cx(s)} y1={ry(0) - rG - 4} y2={ry(nd - 1) + rG + 4} />
			<text class="tok fx-pb__ch" x={cx(s)} y={ry(0) - rG - 10} text-anchor="middle">{showChar(chars[s])}</text>
		{/each}
		{#each depths as d, j (j)}
			<text class="lbl" x={padL - 12} y={ry(j) + (narrow ? 4 : 0)} text-anchor="end">{narrow ? depthLabel(d) : `depth ${depthLabel(d)}`}</text>
			{#if !narrow}<text class="lbl lbl--micro" x={padL - 12} y={ry(j) + 13} text-anchor="end">spoke {fmt(v.max[j])}</text>{/if}
		{/each}
		<!-- backward: what arrives, covector ink, up -->
		{#each depths as _, j (j)}
			{@const u = grow(j, tt)}
			{#if u > 0}
				{#each G[j] as g, s (s)}
					{#if g}
						<SpokeGlyph values={g.map((x) => x * u)} x={cx(s)} y={ry(j)} rmax={rG} scale={v.sc[j].scale} ink="covector" tip={Math.max(1.5, rG / 10)} />
					{/if}
				{/each}
			{/if}
		{/each}
		<!-- the paths the covector takes: straight up through the MLP, fanning out through attention -->
		{#each G[1] as g, s (s)}
			{#if g && grow(1, tt) > 0}
				<line class="fx-pb__path" x1={cx(s)} x2={cx(s)} y1={ry(nd - 1) - rG - 3} y2={ry(nd - 1) - rG - 3 - (ry(nd - 1) - ry(1) - 2 * rG - 6) * grow(1, tt)} />
			{/if}
		{/each}
		{#if grow(0, tt) > 0}
			{#each { length: T } as _, tp (tp)}
				{#if tg < 0 || tp === tg}
					{#each { length: tp + 1 } as _, s (s)}
						{@const x1 = cx(tp)}
						{@const y1 = ry(1) - rG - 3}
						{@const x2 = cx(s)}
						{@const y2 = ry(0) + rG + 3}
						{@const u = grow(0, tt)}
						<line class="fx-pb__path" class:fx-pb__path--faint={tg < 0} x1={x1} y1={y1} x2={x1 + (x2 - x1) * u} y2={y1 + (y2 - y1) * u} />
					{/each}
				{/if}
			{/each}
		{/if}
		<!-- the seed at the final depth -->
		{#if tg >= 0}
			<text class="lbl lbl--covector" x={cx(tg)} y={ry(nd - 1) + rG + 18} text-anchor="middle">e<tspan class="fx-pb__sub" dy="3">{i + 1}</tspan></text>
		{:else}
			<text class="lbl lbl--covector" x={padL + gridW / 2} y={ry(nd - 1) + rG + 18} text-anchor="middle">e<tspan class="fx-pb__sub" dy="3">{i + 1}</tspan><tspan dy="-3"> in every column</tspan></text>
		{/if}
	</g>
{/snippet}

{#snippet side(v, tg, tt, y0)}
	{@const u = growMean(tt)}
	{@const yJ = narrow ? 30 : ry(0) - rG + 40}
	<g transform="translate(0 {y0})">
		{#if tg < 0 && v.mean}
			<text class="lbl" x={sx} y={narrow ? 10 : ry(0) - rG - 14}>mean over the {T} sources</text>
			{#if u > 0}
				<SpokeGlyph values={v.mean.map((x) => x * u)} x={sx + rG + 2} y={narrow ? 10 + rG + 10 : ry(0)} rmax={rG} scale={v.sc[0].scale} ink="covector" tip={Math.max(1.5, rG / 10)} />
			{/if}
			<text class="lbl lbl--micro" x={sx + 2 * rG + 12} y={(narrow ? 10 + rG + 10 : ry(0)) + 4}>row {i + 1} of this window’s J⁽¹⁾</text>
		{/if}
		<text class="lbl" x={sx} y={yJ + (narrow ? 2 * rG + 16 : 0) + D * jCell + 16}>J₀, all ten windows</text>
		<text class="lbl lbl--micro lbl--covector" x={sx} y={yJ + (narrow ? 2 * rG + 16 : 0) + D * jCell + 30}>row {i + 1}: e{sub(i + 1)} pulled back</text>
		<g transform="translate({sx} {yJ + (narrow ? 2 * rG + 16 : 0)})">
			<MatrixGlyph rows={Jg.map((r, a) => (a === i ? r.map(() => 0) : r))} m={D} n={D} cell={jCell} vmax={Math.max(...Jg.flat().map(Math.abs))} grid />
			<MatrixGlyph rows={[Jg[i]]} m={1} n={D} y={i * jCell} cell={jCell} vmax={Math.max(...Jg.flat().map(Math.abs))} ink="covector" />
			{#if interactive}
				{#each { length: D } as _, a (a)}
					<rect class="hit" x={-6} y={a * jCell} width={D * jCell + 12} height={jCell} role="button" tabindex="-1" aria-label="row {a + 1}" onclick={() => (i = a)} onkeydown={act(() => (i = a))} />
				{/each}
				<line class="fx-pb__rowmark" x1={-6} x2={-2} y1={i * jCell + jCell / 2} y2={i * jCell + jCell / 2} />
			{/if}
		</g>
	</g>
{/snippet}

<div class="fx fx-pb" use:measure={(w) => (width = w)}>
	{#if !js}
		<p class="fig__pending">The J-lens export (jspace/artifacts/specimen-jlens.json) is missing.</p>
	{:else if interactive}
		{@const sideH = narrow ? 2 * rG + 40 + D * jCell + 40 : 0}
		{@const H = panelH + sideH + 8}
		<!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions (arrow keys choose the coordinate and the target) -->
		<svg viewBox="0 0 {W} {H}" width={W} height={H} role="group" tabindex="0" onkeydown={key}
			aria-label="The pullback of e{i + 1} from {target < 0 ? 'every column' : `column ${target + 1}`} of the final depth to depths ½ and 0, for the window “{chars.join('')}”. {readout}">
			{@render panel(live, target, t, 0, '')}
			{@render side(live, target, t, narrow ? panelH : 0)}
			<!-- choose the seed: a bottom glyph (one target) or every column -->
			{#each { length: T } as _, s (s)}
				<rect class="hit" x={cx(s) - colW / 2} y={ry(nd - 1) - rG - 4} width={colW} height={2 * rG + 8} role="button" tabindex="-1" aria-label="seed e{i + 1} at column {s + 1}" onclick={() => (target = s)} onkeydown={act(() => (target = s))} />
			{/each}
			<rect class="hit" x={padL} y={ry(nd - 1) + rG + 4} width={gridW} height={20} role="button" tabindex="-1" aria-label="seed e{i + 1} at every column" onclick={() => (target = -1)} onkeydown={act(() => (target = -1))} />
			{#if target >= 0}
				<text class="lbl lbl--micro fx-pb__every" x={padL} y={ry(nd - 1) + rG + 34}>put e{sub(i + 1)} in every column instead</text>
				<rect class="hit" x={padL - 4} y={ry(nd - 1) + rG + 22} width={170} height={18} role="button" tabindex="-1" aria-label="seed every column" onclick={() => (target = -1)} onkeydown={act(() => (target = -1))} />
			{/if}
		</svg>
		<Scrubber bind:t phases={PHASES} label="the pullback of e{i + 1}" duration={3200} ink="covector" />
		<p class="readout" aria-live="polite">
			<span class="readout__part">{readout}</span>
			<span class="readout__hint">Click a glyph of the bottom row to seed one target, or the line under it for every column; click a row of J to choose i; scrub the pullback. Keys: ↑/↓, ←/→.</span>
		</p>
	{:else}
		{@const H = 2 * panelH + 20}
		<svg viewBox="0 0 {W} {H}" width={W} height={H} role="img" aria-label="The pullback of e{i + 1} from the last column, and from every column at once, for the window “{chars.join('')}”.">
			{#each printViews as v, k (k)}
				{@render panel(v, v.tg, 1, k * (panelH + 20), k ? `e${sub(i + 1)} in every column` : `e${sub(i + 1)} in column ${T} only`)}
			{/each}
			{@render side(printViews[1], -1, 1, panelH + 20)}
		</svg>
	{/if}
</div>

<style>
	.fx-pb__stem { stroke: var(--value); stroke-width: var(--hair); opacity: 0.5; }
	.fx-pb__ch { font-size: 15px; fill: var(--ink-2); }
	.fx-pb__path { stroke: var(--covector); stroke-width: var(--hair); stroke-dasharray: 3 2; }
	.fx-pb__path--faint { opacity: 0.45; }
	.fx-pb :global(.fx-pb__sub) { font-size: 0.75em; }
	.fx-pb__rowmark { stroke: var(--covector); stroke-width: 1.5px; }
	.fx-pb__every { cursor: pointer; fill: var(--covector-deep); }
</style>

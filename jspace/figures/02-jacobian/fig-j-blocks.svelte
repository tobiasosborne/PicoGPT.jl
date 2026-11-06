<script module>
	export const meta = { size: 'wide', title: 'The derivative of the rest of the model, block by block' };
	export { default as Key } from './fig-j-blocks.key.svelte';
</script>

<script>
	// Figure: the derivative of the rest of the model, F_ℓ: X_ℓ ↦ X_L, for window 1 of the
	// specimen ("to be or"), as its T(T+1)/2 = 36 nonzero D×D blocks
	//   K_{t′t} = ∂X_L[:, t′] / ∂X_ℓ[:, t],   t ≤ t′,
	// each a Hinton glyph (value ink: a matrix acting on values), laid out as the block
	// matrix is written: rows = targets t′, columns = sources t. Nothing above the diagonal:
	// a later column cannot move an earlier one (causality, as in attention). Beside the
	// triangle: window 1's own average J⁽¹⁾ = (1/T) Σ_t Σ_{t′≥t} K_{t′t}, and the grand average
	// J_ℓ over all ten windows. One area scale for every glyph at every depth.
	// Data: jspace.specimen.jacobians.blocks[i,q,t′,t,j] (exported as ((T+1)/2)·K; divided
	// here by conventions.jacobian_factor) and jacobians.averaged[i,q,j].
	// Interactive: choose the depth on the depth axis; point at a block for its indices.
	// Print: the depth given by the figure's attrs (default 0).
	import { getContext } from 'svelte';
	import { measure } from '#figures/primitives/measure.js';
	import MatrixGlyph from '#figures/primitives/MatrixGlyph.svelte';
	import { fmt, showChar, clamp, layoutWidth } from '#figures/primitives/scales.js';
	import DepthAxis from '../primitives/DepthAxis.svelte';
	import { at, rowsOf, frob, depthLabel, depthWords } from '../primitives/jdata.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const js = $derived(getContext('artifacts')()?.jspace?.specimen ?? null);
	const interactive = $derived(mode !== 'print');
	const D = $derived(specimen.config.D);
	const T = $derived(specimen.config.T);
	// svelte-ignore state_referenced_locally
	let j = $state(clamp(Math.round(2 * (+attrs.depth || 0)), 0, 2));
	let hover = $state(null); // {tp, t}
	let width = $state(868);
	const W = $derived(layoutWidth(width, mode, 330));
	const narrow = $derived(W < 640);

	const depths = $derived(js?.depth ?? []);
	const kinds = $derived(js?.depth_kind ?? []);
	const factor = $derived(js?.conventions?.jacobian_factor ?? 1);
	const win = $derived((js?.sampling?.block_window ?? 1) - 1);
	const chars = $derived(Array.from(specimen.corpus.slice(win, win + T)));
	const K = (tp, t, jj) => (i, q) => at(js.jacobians.blocks, i, q, tp, t, jj) / factor;
	// window 1's own average and the grand average, per depth
	const Jw = $derived(js ? depths.map((_, jj) => rowsOf((i, q) => {
		let s = 0;
		for (let t = 0; t < T; t++) for (let tp = t; tp < T; tp++) s += K(tp, t, jj)(i, q);
		return s / T;
	}, D)) : []);
	const Jg = $derived(js ? depths.map((_, jj) => rowsOf((i, q) => at(js.jacobians.averaged, i, q, jj), D)) : []);
	// one area scale for every block and average at every depth
	const vmax = $derived.by(() => {
		if (!js) return 1;
		let m = 0;
		const b = js.jacobians.blocks.data;
		for (const v of b) m = Math.max(m, Math.abs(v) / factor);
		for (const M of [...Jw, ...Jg]) for (const r of M) for (const v of r) m = Math.max(m, Math.abs(v));
		return m || 1;
	});
	const norms = $derived(js ? Array.from({ length: T }, (_, tp) => Array.from({ length: T }, (_, t) => (t <= tp ? frob(K(tp, t, j), D) : 0))) : []);

	// geometry
	const labelW = $derived(narrow ? 18 : 40);
	const sideW = $derived(narrow ? 0 : Math.min(260, W * 0.3));
	const triW = $derived(W - labelW - sideW - (narrow ? 4 : 24));
	const bs = $derived(Math.min(66, triW / T));
	const inner = $derived(bs * 0.86);
	const cell = $derived(inner / D);
	const tx0 = $derived(labelW + (narrow ? 4 : 10));
	const ty0 = 58;
	const big = $derived(narrow ? Math.min(13, (W - 60) / (2 * D)) : Math.min(17, (sideW - 30) / D));
	const sx = $derived(narrow ? 30 : W - sideW + 20);
	const sy0 = $derived(narrow ? ty0 + T * bs + 74 : ty0 + 6);
	const sy1 = $derived(narrow ? sy0 : sy0 + D * big + 58);
	const sx1 = $derived(narrow ? sx + D * big + 30 : sx);
	const H = $derived(narrow ? sy0 + D * big + 30 : Math.max(ty0 + T * bs + 36, sy1 + D * big + 30) + 34);
	const axisY = $derived(H - 26);

	const readout = $derived.by(() => {
		if (!js) return '';
		const d = depthLabel(depths[j]);
		if (hover) {
			const { tp, t } = hover;
			const k = K(tp, t, j);
			let best = { v: 0, i: 0, q: 0 };
			for (let i = 0; i < D; i++) for (let q = 0; q < D; q++) if (Math.abs(k(i, q)) > Math.abs(best.v)) best = { v: k(i, q), i, q };
			return `K at depth ${d}, target ${tp + 1} (‘${showChar(chars[tp])}’), source ${t + 1} (‘${showChar(chars[t])}’): how a nudge to column ${t + 1} at depth ${d} moves the final column ${tp + 1}. Frobenius norm ${fmt(norms[tp][t])}; largest entry ${fmt(best.v)} in row ${best.i + 1}, column ${best.q + 1}.`;
		}
		const off = norms.reduce((s, r, tp) => s + r.reduce((u, v, t) => u + (t < tp ? v * v : 0), 0), 0);
		const on = norms.reduce((s, r, tp) => s + r[tp] ** 2, 0);
		return `Depth ${d} (${depthWords(depths[j], kinds[j])}): the off-diagonal blocks carry ${fmt(100 * off / (off + on), 1)}% of the squared Frobenius norm of the window's derivative. ‖J⁽¹⁾‖ = ${fmt(frob((i, q) => Jw[j][i][q], D))}, ‖J‖ = ${fmt(frob((i, q) => Jg[j][i][q], D))}.`;
	});
</script>

<div class="fx fx-blocks" use:measure={(w) => (width = w)}>
	{#if !js}
		<p class="fig__pending">The J-lens export (jspace/artifacts/specimen-jlens.json) is missing.</p>
	{:else}
		<svg viewBox="0 0 {W} {H}" width={W} height={H} role={interactive ? 'group' : 'img'}
			aria-label="The 36 blocks of the derivative of the rest of the model at depth {depthLabel(depths[j])} for the window “{chars.join('')}”, and the averages. {readout}">
			<text class="math math--small" x={tx0} y={18}>K<tspan class="fx-blocks__sub" dy="3">t′t</tspan><tspan dy="-3"> = ∂X</tspan><tspan class="fx-blocks__sub" dy="3">L</tspan><tspan dy="-3">[:, t′] / ∂X</tspan><tspan class="fx-blocks__sub" dy="3">{depthLabel(depths[j])}</tspan><tspan dy="-3">[:, t] for “{chars.map(showChar).join('')}”</tspan></text>
			<!-- target labels (rows) and source labels (columns) -->
			{#each chars as ch, k (k)}
				<text class="tok fx-blocks__ch" class:tok--reader={hover?.tp === k} x={tx0 - 8} y={ty0 + k * bs + bs / 2 + 5} text-anchor="end">{showChar(ch)}</text>
				<text class="tok fx-blocks__ch" class:tok--reader={hover?.t === k} x={tx0 + k * bs + bs / 2} y={ty0 + T * bs + 16} text-anchor="middle">{showChar(ch)}</text>
			{/each}
			{#if !narrow}
				<text class="lbl lbl--micro" x={tx0 - 8} y={ty0 - 8} text-anchor="end">target t′</text>
				<text class="lbl lbl--micro" x={tx0 + T * bs} y={ty0 + T * bs + 30} text-anchor="end">source t</text>
			{/if}
			<!-- the triangle: nothing above the diagonal -->
			{#each { length: T } as _, tp (tp)}
				{#each { length: tp + 1 } as _, t (t)}
					{@const x = tx0 + t * bs + (bs - inner) / 2}
					{@const y = ty0 + tp * bs + (bs - inner) / 2}
					<g class:fx-blocks__dim={hover && !(hover.tp === tp && hover.t === t)}>
						<MatrixGlyph get={K(tp, t, j)} m={D} n={D} {x} {y} {cell} {vmax} grid />
					</g>
					{#if interactive}
						<rect class="hit" x={tx0 + t * bs} y={ty0 + tp * bs} width={bs} height={bs} role="button" tabindex="-1" aria-label="target {tp + 1}, source {t + 1}"
							onpointerenter={() => (hover = { tp, t })} onpointerleave={() => (hover = null)} />
					{/if}
				{/each}
			{/each}

			<!-- the averages -->
			{#if narrow}
				<text class="lbl" x={sx} y={sy0 - 26}>window {win + 1}</text>
			{:else}
				<text class="lbl" x={sx} y={sy0 - 26}>window {win + 1}: mean over t of Σ<tspan class="fx-blocks__sub" dy="3">t′≥t</tspan><tspan dy="-3"> K</tspan><tspan class="fx-blocks__sub" dy="3">t′t</tspan></text>
			{/if}
			<text class="math" x={sx} y={sy0 - 8}>J<tspan class="fx-blocks__sup" dy="-6">(1)</tspan><tspan class="fx-blocks__sub" dy="10">{depthLabel(depths[j])}</tspan></text>
			<MatrixGlyph rows={Jw[j]} x={sx} y={sy0} cell={big} {vmax} grid />
			<text class="lbl" x={sx1} y={sy1 - 26}>all ten windows</text>
			<text class="math" x={sx1} y={sy1 - 8}>J<tspan class="fx-blocks__sub" dy="4">{depthLabel(depths[j])}</tspan></text>
			<MatrixGlyph rows={Jg[j]} x={sx1} y={sy1} cell={big} {vmax} grid />

			{#if interactive}
				<DepthAxis {depths} {kinds} x={narrow ? 60 : tx0 + 40} width={narrow ? W - 100 : Math.min(260, triW - 60)} y={axisY} sel={j} blocks={false} onpick={(k) => { j = k; hover = null; }} />
			{/if}
		</svg>
		{#if interactive}
			<p class="readout" aria-live="polite">
				<span class="readout__part">{readout}</span>
				<span class="readout__hint">Choose a depth on the axis; point at a block.</span>
			</p>
		{/if}
	{/if}
</div>

<style>
	.fx-blocks__ch { font-size: 15px; fill: var(--ink-2); }
	.fx-blocks :global(.fx-blocks__sub) { font-size: 0.75em; }
	.fx-blocks :global(.fx-blocks__sup) { font-size: 0.7em; font-style: normal; }
	.fx-blocks__dim { opacity: 0.35; }
</style>

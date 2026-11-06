<script module>
	export const meta = { size: 'wide', title: 'One context against the average, and present against future' };
</script>

<script>
	// Figure: how much one context's averaged Jacobian differs from the average over all of
	// them, and how the average splits into present and future.
	// Top: at depth ½, the ten per-window averages J⁽ᵇ⁾ = (1/T) Σ_t Σ_{t′≥t} K⁽ᵇ⁾_{t′t}, one per
	// window of the specimen, each with its relative distance ‖J⁽ᵇ⁾ − J‖/‖J‖ (Frobenius) from
	// their mean J_½, drawn at the end of the row.
	// Bottom: at each depth, J_ℓ = J^self_ℓ + J^future_ℓ: the diagonal blocks K_tt (the
	// column's effect on its own output) and the strictly later ones (on later outputs),
	// each averaged over sources. Hinton glyphs in value ink, one area scale for all.
	// Data: jspace.specimen.jacobians.{per_window_attention, averaged, self, future}.
	// Interactive: point at a glyph for its numbers. Print: static.
	import { getContext } from 'svelte';
	import { measure } from '#figures/primitives/measure.js';
	import MatrixGlyph from '#figures/primitives/MatrixGlyph.svelte';
	import { fmt, showChar, layoutWidth } from '#figures/primitives/scales.js';
	import { at, rowsOf, frob, depthLabel, depthWords } from '../primitives/jdata.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const js = $derived(getContext('artifacts')()?.jspace?.specimen ?? null);
	const interactive = $derived(mode !== 'print');
	const D = $derived(specimen.config.D);
	const T = $derived(specimen.config.T);
	let hover = $state(null);
	let width = $state(868);
	const W = $derived(layoutWidth(width, mode, 330));
	const narrow = $derived(W < 640);

	const depths = $derived(js?.depth ?? []);
	const kinds = $derived(js?.depth_kind ?? []);
	const half = $derived(depths.indexOf(0.5));
	const B = $derived(js?.jacobians.per_window_attention.shape[2] ?? 0);
	const M = (t, ...rest) => rowsOf((i, q) => at(t, i, q, ...rest), D);
	const per = $derived(js ? Array.from({ length: B }, (_, b) => M(js.jacobians.per_window_attention, b)) : []);
	const J = $derived(js ? depths.map((_, j) => M(js.jacobians.averaged, j)) : []);
	const S = $derived(js ? depths.map((_, j) => M(js.jacobians.self, j)) : []);
	const F = $derived(js ? depths.map((_, j) => M(js.jacobians.future, j)) : []);
	const fr = (R) => frob((i, q) => R[i][q], D);
	const diff = (A, C) => frob((i, q) => A[i][q] - C[i][q], D);
	const dev = $derived(js ? per.map((P) => diff(P, J[half]) / fr(J[half])) : []);
	// the split is exact up to the export's rounding: check it rather than assume it
	const splitErr = $derived(js ? Math.max(...depths.map((_, j) => diff(J[j], S[j].map((r, i) => r.map((v, q) => v + F[j][i][q]))) / Math.max(fr(J[j]), 1e-12))) : 1);
	const vmax = $derived.by(() => {
		let m = 0;
		for (const A of [...per, ...J, ...S, ...F]) for (const r of A) for (const v of r) m = Math.max(m, Math.abs(v));
		return m || 1;
	});
	const winText = (b) => Array.from(specimen.corpus.slice(b, b + T)).map(showChar).join('');

	// geometry: row 1 (ten windows and their mean), row 2 (three depths × J = self + future)
	const perRow = $derived(narrow ? 5 : B);
	const gapX = $derived(narrow ? 10 : 12);
	const meanGap = $derived(narrow ? 0 : 44);
	const g1 = $derived(Math.min(9, (W - 20 - (perRow - 1) * gapX - meanGap - (narrow ? 0 : 1) * 0) / ((perRow + (narrow ? 0 : 1)) * D)));
	const s1 = $derived(g1 * D);
	const y1 = 34;
	const rows1 = $derived(Math.ceil(B / perRow));
	const rowH1 = $derived(s1 + 36);
	const x1 = (b) => 10 + (b % perRow) * (s1 + gapX);
	const yb = (b) => y1 + Math.floor(b / perRow) * rowH1;
	const meanX = $derived(narrow ? 10 : x1(perRow - 1) + s1 + gapX + meanGap);
	const meanY = $derived(narrow ? y1 + rows1 * rowH1 + 14 : y1);
	const top2 = $derived((narrow ? meanY + s1 + 40 : y1 + rows1 * rowH1) + 40);
	const g2 = $derived(Math.min(8.5, (W - (narrow ? 60 : 150) - 2 * 40) / (3 * D)));
	const s2 = $derived(g2 * D);
	const lab2 = $derived(narrow ? 44 : 130);
	const c2 = (k) => lab2 + k * (s2 + 40);
	const rowH2 = $derived(s2 + 22);
	const r2 = (j) => top2 + 22 + j * rowH2;
	const H = $derived(r2(depths.length) + 6);

	const readout = $derived.by(() => {
		if (!js) return '';
		if (hover?.kind === 'win') {
			const b = hover.b;
			return `Window ${b + 1}, “${winText(b)}”: its own average at depth ½ has norm ${fmt(fr(per[b]))} and lies at ${fmt(100 * dev[b], 0)}% of ‖J½‖ = ${fmt(fr(J[half]))} from the average of all ten.`;
		}
		if (hover?.kind === 'split') {
			const j = hover.j;
			return `Depth ${depthLabel(depths[j])} (${depthWords(depths[j], kinds[j])}): ‖J‖ = ${fmt(fr(J[j]))}, ‖Jˢᵉˡᶠ‖ = ${fmt(fr(S[j]))}, ‖Jᶠᵘᵗᵘʳᵉ‖ = ${fmt(fr(F[j]))}.`;
		}
		const lo = Math.min(...dev), hi = Math.max(...dev);
		return `The ten windows' own averages lie between ${fmt(100 * lo, 0)}% and ${fmt(100 * hi, 0)}% of ‖J½‖ from their mean. J = Jˢᵉˡᶠ + Jᶠᵘᵗᵘʳᵉ holds at every depth to within ${splitErr.toExponential(0)} of ‖J‖, the export's rounding.`;
	});
</script>

<div class="fx fx-avg" use:measure={(w) => (width = w)}>
	{#if !js}
		<p class="fig__pending">The J-lens export (jspace/artifacts/specimen-jlens.json) is missing.</p>
	{:else}
		<svg viewBox="0 0 {W} {H}" width={W} height={H} role={interactive ? 'group' : 'img'}
			aria-label="The ten per-window averaged Jacobians of the specimen at depth ½ and their mean, and the averaged Jacobian at each depth split into its present and future parts. {readout}">
			<text class="lbl lbl--head" x={10} y={14}>one window at a time, depth ½</text>
			{#each per as P, b (b)}
				<g class:fx-avg__dim={hover?.kind === 'win' && hover.b !== b}>
					<MatrixGlyph rows={P} x={x1(b)} y={yb(b)} cell={g1} {vmax} grid />
					<text class="tok fx-avg__win" x={x1(b) + s1 / 2} y={yb(b) + s1 + 14} text-anchor="middle">{winText(b)}</text>
					<text class="lbl lbl--micro" x={x1(b) + s1 / 2} y={yb(b) + s1 + 27} text-anchor="middle">{fmt(100 * dev[b], 0)}%</text>
				</g>
				{#if interactive}
					<rect class="hit" x={x1(b) - 4} y={yb(b) - 4} width={s1 + 8} height={s1 + 34} role="button" tabindex="-1" aria-label="window {b + 1}"
						onpointerenter={() => (hover = { kind: 'win', b })} onpointerleave={() => (hover = null)} />
				{/if}
			{/each}
			{#if !narrow}
				<text class="lbl" x={meanX - meanGap / 2} y={meanY + s1 / 2 + 4} text-anchor="middle">mean</text>
			{/if}
			<MatrixGlyph rows={J[half]} x={meanX} y={meanY} cell={g1} {vmax} grid />
			<text class="math math--small" x={meanX + s1 / 2} y={meanY + s1 + 15} text-anchor="middle">J<tspan class="fx-avg__sub" dy="3">½</tspan></text>
			{#if narrow}<text class="lbl" x={meanX + s1 + 10} y={meanY + s1 / 2 + 4}>the mean of the ten</text>{/if}

			<text class="lbl lbl--head" x={10} y={top2 - 6}>present and future</text>
			{#each [['J', ''], ['J', 'self'], ['J', 'future']] as [s, sup], k (k)}
				<text class="math math--small" x={c2(k) + s2 / 2} y={top2 + 12} text-anchor="middle">{s}<tspan class="fx-avg__sup" dy="-5">{sup}</tspan></text>
			{/each}
			{#each depths as d, j (j)}
				{@const y = r2(j)}
				<g class:fx-avg__dim={hover?.kind === 'split' && hover.j !== j}>
					<text class="lbl" x={lab2 - 14} y={y + s2 / 2 + 4} text-anchor="end">{narrow ? depthLabel(d) : `depth ${depthLabel(d)}`}</text>
					{#each [J[j], S[j], F[j]] as A, k (k)}
						<MatrixGlyph rows={A} x={c2(k)} {y} cell={g2} {vmax} grid />
					{/each}
					<text class="lbl fx-avg__op" x={c2(1) - 20} y={y + s2 / 2 + 5} text-anchor="middle">=</text>
					<text class="lbl fx-avg__op" x={c2(2) - 20} y={y + s2 / 2 + 5} text-anchor="middle">+</text>
				</g>
				{#if interactive}
					<rect class="hit" x={lab2 - 80} y={y - 4} width={c2(2) + s2 - lab2 + 84} height={s2 + 8} role="button" tabindex="-1" aria-label="depth {depthLabel(d)}"
						onpointerenter={() => (hover = { kind: 'split', j })} onpointerleave={() => (hover = null)} />
				{/if}
			{/each}
		</svg>
		{#if interactive}
			<p class="readout" aria-live="polite">
				<span class="readout__part">{readout}</span>
				<span class="readout__hint">Point at a window, or at a depth.</span>
			</p>
		{/if}
	{/if}
</div>

<style>
	.fx-avg__win { font-size: 11.5px; fill: var(--ink-2); white-space: pre; }
	.fx-avg__dim { opacity: 0.35; }
	.fx-avg__op { font-size: 15px; fill: var(--ink-2); }
	.fx-avg :global(.fx-avg__sub) { font-size: 0.75em; font-style: normal; }
	.fx-avg :global(.fx-avg__sup) { font-size: 0.72em; font-style: italic; }
</style>

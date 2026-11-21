<script>
	// How to read the triangle of Jacobian blocks: its top-left corner (targets and sources
	// 1–3 of window 1, depth 0), enlarged, from the figure's own data.
	import { getContext } from 'svelte';
	import HowToRead from '#figures/primitives/HowToRead.svelte';
	import MatrixGlyph from '#figures/primitives/MatrixGlyph.svelte';
	import { showChar } from '#figures/primitives/scales.js';
	import { at } from '../primitives/jdata.js';

	const specimen = $derived(getContext('specimen')());
	const js = $derived(getContext('artifacts')()?.jspace?.specimen ?? null);
	const D = $derived(specimen.config.D);
	const factor = $derived(js?.conventions?.jacobian_factor ?? 1);
	const chars = $derived(Array.from(specimen.corpus.slice(0, 3)));
	const K = (tp, t) => (i, q) => at(js.jacobians.blocks, i, q, tp, t, 0) / factor;
	const n = 3, bs = 50, inner = 44, cell = inner / 8, x0 = 40, y0 = 66;
	const vmax = $derived.by(() => {
		if (!js) return 1;
		let m = 0;
		for (let tp = 0; tp < n; tp++) for (let t = 0; t <= tp; t++) for (let i = 0; i < D; i++) for (let q = 0; q < D; q++) m = Math.max(m, Math.abs(K(tp, t)(i, q)));
		return m || 1;
	});
</script>

{#if js}
	<HowToRead width={244} height={250} notes={[
		{ ax: x0 + bs + bs / 2, ay: y0 + 10, side: 'above', len: 22, text: 'nothing above the diagonal:\na later column cannot\nmove an earlier one', dot: false },
		{ ax: x0 + bs / 2, ay: y0 + 2 * bs + bs / 2 + 8, side: 'below', len: 52, text: 'one block K, 8×8: area ∝ |entry|,\nfilled if positive, open if negative', align: 'start', dx: -28 }
	]}>
		{#snippet glyph()}
			<g class="fx">
				{#each { length: n } as _, tp (tp)}
					<text x={x0 - 6} y={y0 + tp * bs + bs / 2 + 5} text-anchor="end" font-size="14" style="font-family: var(--serif); fill: var(--ink-2)">{showChar(chars[tp])}</text>
					{#each { length: tp + 1 } as _, t (t)}
						<MatrixGlyph get={K(tp, t)} m={8} n={8} x={x0 + t * bs + (bs - inner) / 2} y={y0 + tp * bs + (bs - inner) / 2} {cell} {vmax} grid />
					{/each}
					<text x={x0 + tp * bs + bs / 2} y={y0 + n * bs + 14} text-anchor="middle" font-size="14" style="font-family: var(--serif); fill: var(--ink-2)">{showChar(chars[tp])}</text>
				{/each}
				<text class="lbl lbl--micro" x={x0 - 6} y={y0 - 6} text-anchor="end">t′</text>
				<text class="lbl lbl--micro" x={x0 + n * bs + 4} y={y0 + n * bs + 13}>t</text>
			</g>
		{/snippet}
	</HowToRead>
{/if}

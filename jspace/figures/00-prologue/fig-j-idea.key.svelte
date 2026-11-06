<script>
	// How to read one row of fig-j-idea, enlarged: the column x_ℓ at a depth (spokes),
	// the averaged Jacobian J_ℓ that replaces the rest of the model (Hinton squares), and
	// the readout of J_ℓ x_ℓ (bars). Drawn from the figure's own data: window 1, the
	// figure's column, depth 0.
	import { getContext } from 'svelte';
	import HowToRead from '#figures/primitives/HowToRead.svelte';
	import SpokeGlyph from '#figures/primitives/SpokeGlyph.svelte';
	import MatrixGlyph from '#figures/primitives/MatrixGlyph.svelte';
	import { clamp, maxAbs, spokeScale } from '#figures/primitives/scales.js';
	import DistBars from '../primitives/DistBars.svelte';
	import { fibre, matrixOf, rowsOf } from '../primitives/jdata.js';

	let { attrs = {} } = $props();
	const specimen = $derived(getContext('specimen')());
	const js = $derived(getContext('artifacts')()?.jspace?.specimen ?? null);
	const D = $derived(specimen.config.D);
	const start = $derived(clamp((+attrs.window || 1) - 1, 0, 9));
	const pos = $derived(clamp((+attrs.column || 6) - 1, 0, 7));
	const x0 = $derived(specimen.windowTrace(start).X0[pos]);
	const n = $derived(start * specimen.config.T + pos);
	const J0 = $derived(js ? rowsOf(matrixOf(js.jacobians.averaged, 0), D) : null);
	const p0 = $derived(js ? fibre(js.lenses.j_probs, n, 0) : null);
	const r = 19, gx = 30, gy = 80;
	const cell = 5, jx = 70, jw = 5 * 8;
	const bx = 136, by = 94, bw = 9, gap = 3, bh = 34;
</script>

{#if js}
	<HowToRead width={244} height={210} notes={[
		{ ax: gx, ay: gy + r + 2, side: 'below', len: 36, text: 'the column at one depth:\none spoke per entry', align: 'start', dx: -24 },
		{ ax: jx + jw / 2, ay: gy - jw / 2, side: 'above', len: 14, text: 'the averaged derivative,\none 8×8 matrix: area ∝ |entry|,\nopen if negative' },
		{ ax: bx + 3 * (bw + gap) + bw / 2, ay: by + 15, side: 'below', len: 66, text: 'the readout of J x:\nthe probability of\neach character', align: 'end', dx: 104 }
	]}>
		{#snippet glyph()}
			<g class="fx">
				<line x1={gx} x2={gx} y1={gy - r - 16} y2={gy - r - 3} style="stroke: var(--value); stroke-width: var(--line)" />
				<line x1={gx} x2={gx} y1={gy + r + 3} y2={gy + r + 14} style="stroke: var(--value); stroke-width: var(--line)" />
				<SpokeGlyph values={x0} x={gx} y={gy} rmax={r} scale={spokeScale(r, maxAbs(x0)).scale} tip={2} />
				<line x1={gx + r + 4} x2={jx - 4} y1={gy} y2={gy} style="stroke: var(--ink); stroke-width: 0.9px" />
				<MatrixGlyph rows={J0} x={jx} y={gy - jw / 2} {cell} vmax={maxAbs(J0)} grid />
				<line x1={jx + jw + 4} x2={bx - 8} y1={gy} y2={gy} style="stroke: var(--ink); stroke-width: 0.9px" />
				<DistBars values={p0} x={bx} y={by} {bw} {gap} h={bh} vocab={specimen.vocab} chars={specimen.vocab} charSize={11} />
			</g>
		{/snippet}
	</HowToRead>
{/if}

<script>
	// How to read a covector matrix: three rows and three columns of the figure's own
	// share z̄ xᵀ, enlarged, with the z̄ entries (hatched bars) beside the rows and the
	// x entries (value lollipops) above the columns. The target's row is negative, so
	// the piece shows both hatched (positive) and open (negative) squares.
	import { getContext } from 'svelte';
	import HowToRead from '../primitives/HowToRead.svelte';
	import MatrixGlyph from '../primitives/MatrixGlyph.svelte';
	import { fmt, clamp, maxAbs } from '../primitives/scales.js';

	let { attrs = {} } = $props();
	const specimen = $derived(getContext('specimen')());
	const gt = $derived(specimen.lab.gradTrace(specimen.theta0));
	const w = $derived(gt.windows[clamp((+attrs.window || 1) - 1, 0, gt.windows.length - 1)]);
	const t = $derived(clamp((+attrs.column || 8) - 1, 0, specimen.config.T - 1));
	const zbar = $derived(w.bar.Z[t]);
	const x = $derived(w.val.Xf[t]);
	const y = $derived(w.y[t]);
	// rows: the target and the two after it; columns: one positive and two others of x
	const rows = $derived([y, (y + 1) % zbar.length, (y + 2) % zbar.length]);
	const cols = $derived.by(() => {
		const idx = x.map((v, r) => ({ v, r }));
		const pos = idx.filter((o) => o.v > 0).sort((a, b) => b.v - a.v)[0];
		const neg = idx.filter((o) => o.v < 0).sort((a, b) => a.v - b.v)[0];
		const rest = idx.filter((o) => o !== pos && o !== neg).sort((a, b) => Math.abs(b.v) - Math.abs(a.v))[0];
		return [pos, neg, rest].filter(Boolean).map((o) => o.r).sort((a, b) => a - b);
	});
	const M = $derived(rows.map((a) => cols.map((r) => zbar[a] * x[r])));
	const vmax = $derived(maxAbs(M));
	const zm = $derived(maxAbs(rows.map((a) => zbar[a])));
	const xm = $derived(maxAbs(cols.map((r) => x[r])));
	const c = 27, x0 = 56, y0 = 58;
	const ch = (a) => (specimen.vocab[a] === ' ' ? '␣' : specimen.vocab[a]);
	// a negative square in the target's row and a positive one in the last row (rightmost)
	const pick = (i, sgn) => { for (let j = 2; j >= 0; j--) if (Math.sign(M[i][j]) === sgn) return [i, j]; return [i, 2]; };
	const negAt = $derived(pick(0, -1)[0] === 0 && M[0].some((v) => v < 0) ? pick(0, -1) : pick(2, -1));
	const posAt = $derived(M[2].some((v) => v > 0) ? pick(2, 1) : pick(0, 1));
</script>

<HowToRead width={244} height={262} notes={[
	{ ax: x0 + negAt[1] * c + c / 2 + 4, ay: y0 + negAt[0] * c + c / 2, side: 'right', len: x0 + 3 * c - (x0 + negAt[1] * c + c / 2 + 4) + 8, text: `open: negative,\n${fmt(M[negAt[0]][negAt[1]], 4)}` },
	{ ax: x0 + posAt[1] * c + c / 2 + 4, ay: y0 + posAt[0] * c + c / 2, side: 'right', len: x0 + 3 * c - (x0 + posAt[1] * c + c / 2 + 4) + 8, text: `hatched: positive,\n${fmt(M[posAt[0]][posAt[1]], 4)}` },
	{ ax: x0 + 3 * c + 2, ay: y0 - 24, side: 'right', len: 10, text: 'x: dots, value ink' },
	{ ax: x0 - 21, ay: y0 + 2.5 * c, side: 'below', len: 26, text: `a bar is the row’s entry of z̄, here\n${rows.map((a) => `${fmt(zbar[a], 3)}`).join(', ')}; the target’s row\nis the large negative one. A square\nis the product of its row’s bar and its\ncolumn’s dot; area ∝ |product|`, align: 'start', dx: -30 }
]}>
	{#snippet glyph()}
		<g class="fx">
			<!-- x above the columns -->
			<line class="axis" x1={x0} x2={x0 + 3 * c} y1={y0 - 24} y2={y0 - 24} />
			{#each cols as r, j (r)}
				{@const yy = y0 - 24 - (x[r] / xm) * 18}
				<line class="stem" x1={x0 + j * c + c / 2} x2={x0 + j * c + c / 2} y1={y0 - 24} y2={yy} />
				<circle class={x[r] >= 0 ? 'dot--value' : 'dot--open'} cx={x0 + j * c + c / 2} cy={yy} r="2.6" />
			{/each}
			<!-- z̄ beside the rows -->
			<line class="axis" x1={x0 - 22} x2={x0 - 22} y1={y0} y2={y0 + 3 * c} />
			{#each rows as a, i (a)}
				{@const len = (Math.abs(zbar[a]) / zm) * 18}
				<rect class="bar--covector" style:fill={zbar[a] < 0 ? 'none' : null} x={zbar[a] >= 0 ? x0 - 22 : x0 - 22 - len} y={y0 + i * c + c * 0.25} width={Math.max(0.6, len)} height={c * 0.5} />
				<text x={x0 - 44} y={y0 + i * c + c / 2 + 5} text-anchor="end" font-size="15" style="font-family: var(--serif); fill: var(--ink-2)">{ch(a)}</text>
			{/each}
			<text x={x0 - 25} y={y0 - 8} style="font-family: var(--serif); font-style: italic; font-size: 13px; fill: var(--ink)">z</text>
			<line x1={x0 - 24} x2={x0 - 18.6} y1={y0 - 17.5} y2={y0 - 17.5} style="stroke: var(--ink); stroke-width: 0.9px" />
			<MatrixGlyph rows={M} x={x0} y={y0} cell={c} {vmax} ink="covector" />
		</g>
	{/snippet}
</HowToRead>

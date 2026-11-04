<script>
	// How to read the lower-triangle dot matrix: the first four readers of head 1,
	// enlarged. Those rows are complete (a reader at position i ≤ 4 has no source
	// beyond 4), so each really sums to one.
	import { getContext } from 'svelte';
	import HowToRead from '../primitives/HowToRead.svelte';
	import TriangleDots from '../primitives/TriangleDots.svelte';
	import { fmt, showChar, clamp } from '../primitives/scales.js';

	let { attrs = {} } = $props();
	const specimen = $derived(getContext('specimen')());
	const start = $derived(clamp((+attrs.window || 11) - 1, 0, specimen.corpus.length - specimen.config.T));
	const tr = $derived(specimen.windowTrace(start));
	const n = 4;
	const A = $derived(tr.heads[0].A.slice(0, n).map((r) => r.slice(0, n)));
	const c = 28, x = 36, y = 22;
	// biggest dot in row 3 gets the area note
	const big = $derived(A[2].slice(0, 3).map((w, j) => ({ w, j })).sort((a, b) => b.w - a.w)[0]);
</script>

<HowToRead width={244} height={214} notes={[
	{ ax: x + 2.7 * c, ay: y + 0.75 * c, side: 'right', len: 12, text: 'nothing above\nthe diagonal', dot: false },
	{ ax: x + big.j * c + c / 2 + 6, ay: y + 2 * c + c / 2 - 6, side: 'right', len: 3.1 * c - big.j * c - 2, text: `area ∝ weight\n${fmt(big.w)}` },
	{ ax: x + 4 * c - 2, ay: y + 3 * c + c / 2, side: 'right', len: 10, text: 'a row\nsums to 1' },
	{ ax: x - 28, ay: y + 3 * c + c / 2 + 8, side: 'below', len: 30, text: `row: the reader ‘${showChar(tr.chars[3])}’;\ncolumns: its sources`, align: 'start', dx: -4 }
]}>
	{#snippet glyph()}
		<g class="fx">
			{#each { length: n } as _, i (i)}
				<text x={x - 10} y={y + i * c + c / 2 + 5} text-anchor="end" font-size="15" style="font-family: var(--serif); fill: var(--ink-2)">{showChar(tr.chars[i])}</text>
			{/each}
			<TriangleDots M={A} {x} {y} cell={c} />
		</g>
	{/snippet}
</HowToRead>

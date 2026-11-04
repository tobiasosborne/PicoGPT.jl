<script>
	// How to read a step-size strip: one row of the figure, enlarged — the output
	// bias b_out at the first update of the book's run (V numbers, so every mark
	// can be seen). Bars: |gᵢ|; dots: |Δθᵢ|/η.
	import { getContext } from 'svelte';
	import HowToRead from '../primitives/HowToRead.svelte';
	import { pow10 } from './paramname.js';

	const specimen = $derived(getContext('specimen')());
	const run = $derived.by(() => {
		const r = specimen.lab.run();
		r.step();
		return r;
	});
	const row = $derived(specimen.layout.find((l) => l.name === 'bout'));
	const g = $derived(Array.from(run.grad.subarray(row.offset, row.offset + row.size), Math.abs));
	const d = $derived(Array.from(run.delta.subarray(row.offset, row.offset + row.size), (x) => Math.abs(x) / run.lr[0]));
	// a local log axis 10^-3 … 10^0.3
	const x0 = 14, x1 = 186, LO = -3, HI = 0.15;
	const X = (v) => x0 + ((Math.log10(v) - LO) / (HI - LO)) * (x1 - x0);
	const y = 70;
	const big = $derived(g.map((v, k) => ({ v, k })).sort((a, b) => b.v - a.v)[0]);
</script>

<HowToRead width={244} height={170} notes={[
	{ ax: X(big.v), ay: y - 12, side: 'above', len: 22, text: 'a bar: one parameter’s |gᵢ|', align: 'end', dx: 8 },
	{ ax: X(d[0]) + 3, ay: y + 11, side: 'right', len: 6, text: 'a dot: its\nstep / η', dot: false },
	{ ax: X(1), ay: y + 26, side: 'below', len: 18, text: 'one tick per factor 100;\nat 1 the step is exactly η', align: 'end', dx: 4 }
]}>
	{#snippet glyph()}
		<g class="fx">
			{#each [-2, 0] as e (e)}
				<line class="rule" x1={X(10 ** e)} x2={X(10 ** e)} y1={y - 16} y2={y + 22} />
				<text class="lbl lbl--micro" x={X(10 ** e)} y={y - 20} text-anchor="middle">{pow10(e)}</text>
			{/each}
			<line x1={X(1)} x2={X(1)} y1={y - 16} y2={y + 26} style="stroke: var(--ink-2); stroke-width: 0.9px" />
			{#each g as v, k (k)}
				<line x1={X(v)} x2={X(v)} y1={y - 12 + (k % 3) * 2} y2={y - 2 + (k % 3) * 2} style="stroke: var(--covector); stroke-width: 1.4px; mix-blend-mode: var(--overprint)" />
			{/each}
			{#each d as v, k (k)}
				<circle cx={X(v)} cy={y + 9 + (k % 3) * 2.5} r="2.6" style="fill: var(--value); mix-blend-mode: var(--overprint)" />
			{/each}
		</g>
	{/snippet}
</HowToRead>

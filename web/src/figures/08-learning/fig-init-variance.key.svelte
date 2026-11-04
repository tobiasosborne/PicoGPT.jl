<script>
	// How to read a size profile: the first rows of the middle column (the book's
	// variances), enlarged.
	import { getContext } from 'svelte';
	import HowToRead from '../primitives/HowToRead.svelte';
	import { computed } from '../primitives/computed.svelte.js';
	import { INIT_KEY, initJob, ROWS } from './initvar.js';

	const specimen = getContext('specimen')();
	const job = computed(INIT_KEY, initJob(specimen));
	const seeds = $derived(job.value?.[1]);
	const rows = ROWS.slice(0, 4);
	const x0 = 8, x1 = 148, LO = -0.6, HI = 0.4; // log10 size, 0.25 … 2.5
	const X = (v) => x0 + ((Math.log10(v) - LO) / (HI - LO)) * (x1 - x0);
	const Yr = (k) => 30 + k * 26;
	const med = (a) => { const s = a.slice().sort((p, q) => p - q); const n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };
	const spread = $derived(seeds ? seeds.map((v) => v[rows[1][0]]) : []);
	const lo = $derived(spread.length ? Math.min(...spread) : 1);
</script>

{#if seeds}
	<HowToRead width={244} height={186} notes={[
		{ ax: X(seeds[0][rows[0][0]]) + 4, ay: Yr(0) - 4, side: 'right', len: 156 - X(seeds[0][rows[0][0]]), text: 'ring: the\nbook’s θ₀' },
		{ ax: X(Math.max(...seeds.map((v) => v[rows[1][0]]))) + 3, ay: Yr(1) + 4, side: 'right', len: 156 - X(Math.max(...seeds.map((v) => v[rows[1][0]]))), text: 'a dot: one\nseed’s rms size' },
		{ ax: X(med(seeds.map((v) => v[rows[2][0]]))) + 3, ay: Yr(2) + 9, side: 'right', len: 156 - X(med(seeds.map((v) => v[rows[2][0]]))), text: 'the line: the\nmedians' },
		{ ax: X(med(seeds.map((v) => v[rows[3][0]]))), ay: Yr(3) + 8, side: 'below', len: 18, text: 'rows top to bottom: the\nforward pass, in order', align: 'start', dx: -20 },
		{ ax: X(1), ay: 14, side: 'above', len: 4, text: 'size 1', dot: false }
	]}>
		{#snippet glyph()}
			<g class="fx">
				<line x1={X(1)} x2={X(1)} y1="16" y2={Yr(3) + 10} style="stroke: var(--ink-3); stroke-width: 0.8px" />
				<polyline points={rows.map(([k], i) => `${X(med(seeds.map((v) => v[k])))},${Yr(i)}`).join(' ')} style="fill: none; stroke: var(--value); stroke-width: 1.1px; opacity: 0.6" />
				{#each rows as [k], i (k)}
					{#each seeds as v, si (si)}
						<circle cx={X(v[k])} cy={Yr(i) + (si - 3.5) * 1.6} r={si === 0 ? 3 : 2.8} style={si === 0 ? 'fill: var(--paper); stroke: var(--ink); stroke-width: 1.3px' : 'fill: var(--value); opacity: 0.8; mix-blend-mode: var(--overprint)'} />
					{/each}
				{/each}
			</g>
		{/snippet}
	</HowToRead>
{/if}

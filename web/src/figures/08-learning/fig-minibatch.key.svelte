<script>
	// How to read a minibatch cloud: the b = 2 panel before training, enlarged.
	import { getContext } from 'svelte';
	import HowToRead from '../primitives/HowToRead.svelte';
	import { computed } from '../primitives/computed.svelte.js';
	import { MINIBATCH_KEY, minibatchJob } from './minibatch.js';

	const specimen = getContext('specimen')();
	const job = computed(MINIBATCH_KEY, minibatchJob(specimen));
	const st = $derived(job.value?.[0]);
	const c = $derived(st?.clouds[1]);
	const S = 116, ox = 4, oy = 10;
	const fr = $derived.by(() => {
		if (!st) return null;
		let r = 0;
		for (const p of c.pts) r = Math.max(r, Math.hypot(p[0] - st.g[0], p[1] - st.g[1]));
		return { r: r * 1.15 };
	});
	const X = (v) => ox + S / 2 + (v - st.g[0]) * (S / 2 / fr.r);
	const Y = (v) => oy + S / 2 - (v - st.g[1]) * (S / 2 / fr.r);
	const far = $derived(c ? c.pts.reduce((a, p) => (Math.hypot(p[0] - st.g[0], p[1] - st.g[1]) > Math.hypot(a[0] - st.g[0], a[1] - st.g[1]) ? p : a)) : null);
</script>

{#if st}
	<HowToRead width={244} height={184} notes={[
		{ ax: X(far[0]) + 4, ay: Y(far[1]), side: 'right', len: Math.max(8, 134 - X(far[0])), text: 'a bar: the\ngradient of\none minibatch' },
		{ ax: X(st.g[0]) + 7, ay: Y(st.g[1]), side: 'right', len: 134 - X(st.g[0]) - 7, text: 'the cross: the\nfull gradient,\ntheir mean' },
		{ ax: X(st.g[0]), ay: Y(st.g[1]) + c.rmsP * (S / 2 / fr.r), side: 'below', len: 132 + oy - Y(st.g[1]) - c.rmsP * (S / 2 / fr.r), text: 'dashed: their rms distance\nfrom the full gradient', align: 'start', dx: -30 }
	]}>
		{#snippet glyph()}
			<g class="fx">
				<circle cx={X(st.g[0])} cy={Y(st.g[1])} r={c.rmsP * (S / 2 / fr.r)} style="fill: none; stroke: var(--ink-3); stroke-width: 0.6px; stroke-dasharray: 2 2.5" />
				<g style="mix-blend-mode: var(--overprint)">
					{#each c.pts as p, j (j)}
						<rect x={X(p[0]) - 4} y={Y(p[1]) - 1} width="8" height="2" style="fill: var(--covector)" />
					{/each}
				</g>
				<line x1={X(st.g[0]) - 8} x2={X(st.g[0]) + 8} y1={Y(st.g[1])} y2={Y(st.g[1])} style="stroke: var(--ink); stroke-width: 1.8px" />
				<line x1={X(st.g[0])} x2={X(st.g[0])} y1={Y(st.g[1]) - 8} y2={Y(st.g[1]) + 8} style="stroke: var(--ink); stroke-width: 1.8px" />
			</g>
		{/snippet}
	</HowToRead>
{/if}

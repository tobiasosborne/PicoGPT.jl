<script>
	// The book's spoke glyph (web/src/figures/primitives/SpokeGlyph.svelte, same classes and
	// encoding: spoke k at 2πk/n from 12 o'clock, clockwise; length ∝ |x_k| on a shared
	// scale; filled tip +, open tip −; value ink ends in a dot, covector ink in a bar), for
	// vectors with exact zeros: a spoke shorter than `cut` px draws nothing at all, instead of
	// a tip stacked on the hub. Used where most entries are exactly zero (a basis covector e_i,
	// a block that causality empties). The compass ticks still mark every spoke position.
	let {
		values = [],
		x = 0,
		y = 0,
		rmax = 20,
		scale = 10,
		ink = 'value',
		tip = 2.2,
		cut = 0.6,
		compass = true
	} = $props();

	const n = $derived(values.length);
	const spokes = $derived(
		values
			.map((v, k) => {
				const a = -Math.PI / 2 + (2 * Math.PI * k) / n;
				const r = Math.min(rmax * 1.08, Math.abs(v) * scale);
				const c = Math.cos(a), s = Math.sin(a);
				return { k, v, a, r, c, s, x2: r * c, y2: r * s };
			})
	);
</script>

<g class="spoke spoke--{ink}" transform="translate({x} {y})">
	{#if compass}
		{#each spokes as s (s.k)}
			<circle class="spoke__tick" cx={s.c * rmax} cy={s.s * rmax} r="0.9" />
		{/each}
	{/if}
	<circle class="spoke__hub" r="1.1" />
	{#each spokes as s (s.k)}
		{#if s.r >= cut}
			<g class="spoke__k" class:spoke__k--neg={s.v < 0}>
				<line class="spoke__line" x1="0" y1="0" x2={s.x2} y2={s.y2} />
				{#if ink === 'covector'}
					<rect class="spoke__bar" x={-tip * 1.9} y={-tip * 0.55} width={tip * 3.8} height={tip * 1.1} transform="translate({s.x2} {s.y2}) rotate({(s.a * 180) / Math.PI + 90})" />
				{:else}
					<circle class="spoke__tip" cx={s.x2} cy={s.y2} r={tip} />
				{/if}
			</g>
		{/if}
	{/each}
</g>

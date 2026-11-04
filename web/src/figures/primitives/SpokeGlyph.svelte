<script>
	// A vector as a radial spoke glyph. Spoke k shows component x_k: it leaves the
	// centre at angle 2πk/n (k = 0 at 12 o'clock, clockwise, in index order) with
	// length ∝ |x_k| on a scale shared by every glyph it is compared with
	// (scale = rmax / max, see spokeScale). Sign is fill, as everywhere in the book:
	// a filled tip is positive, an open tip negative. Value ink ends in a dot;
	// covector ink ends in a bar (the bar of x̄). Faint compass ticks at rmax mark
	// the full scale and the spoke positions, so a short spoke is still findable.
	let {
		values = [],
		x = 0,
		y = 0,
		rmax = 20,
		scale = 10,
		ink = 'value',
		tip = 2.2,
		compass = true,
		muted = false,
		emphasis = -1,
		weight = 1,
		title = ''
	} = $props();

	const n = $derived(values.length);
	const spokes = $derived(
		values.map((v, k) => {
			const a = -Math.PI / 2 + (2 * Math.PI * k) / n;
			const r = Math.min(rmax * 1.08, Math.abs(v) * scale);
			const c = Math.cos(a), s = Math.sin(a);
			return { k, v, a, c, s, x2: r * c, y2: r * s, px: -s, py: c };
		})
	);
</script>

<g class="spoke spoke--{ink}" class:spoke--muted={muted} transform="translate({x} {y})" style:--w={weight}>
	{#if title}<title>{title}</title>{/if}
	{#if compass}
		{#each spokes as s (s.k)}
			<circle class="spoke__tick" cx={s.c * rmax} cy={s.s * rmax} r="0.9" />
		{/each}
	{/if}
	<circle class="spoke__hub" r="1.1" />
	{#each spokes as s (s.k)}
		<g class="spoke__k" class:spoke__k--em={s.k === emphasis} class:spoke__k--neg={s.v < 0}>
			<line class="spoke__line" x1="0" y1="0" x2={s.x2} y2={s.y2} />
			{#if ink === 'covector'}
				<rect class="spoke__bar" x={-tip * 1.9} y={-tip * 0.55} width={tip * 3.8} height={tip * 1.1} transform="translate({s.x2} {s.y2}) rotate({(s.a * 180) / Math.PI + 90})" />
			{:else}
				<circle class="spoke__tip" cx={s.x2} cy={s.y2} r={tip} />
			{/if}
		</g>
	{/each}
</g>

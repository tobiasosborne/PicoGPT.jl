<script>
	// The attention matrix as a lower triangle of area-encoded dots: row i is reader
	// i, column j is source j; dot area ∝ A_ij (rows sum to 1). NOTHING is drawn
	// above the diagonal — the future is not small, it is absent.
	// Signed mode (scores S, which exist for every pair): area ∝ |S_ij|/vmax, filled
	// for positive and open for negative; `upper` (0…1) shows the upper triangle,
	// which the causal mask removes.
	let {
		M = [],
		x = 0,
		y = 0,
		cell = 22,
		rmax = null,
		vmax = 1,
		signed = false,
		upper = 0,
		row = -1,
		col = -1,
		ink = 'value',
		onrow = null
	} = $props();

	const n = $derived(M.length);
	const R = $derived(rmax ?? cell * 0.47);
	const dots = $derived.by(() => {
		const out = [];
		for (let i = 0; i < n; i++)
			for (let j = 0; j < n; j++) {
				const above = j > i;
				if (above && upper <= 0) continue;
				const v = M[i][j];
				const r = R * Math.sqrt(Math.min(1, Math.abs(v) / vmax));
				out.push({ i, j, v, r, above, cx: j * cell + cell / 2, cy: i * cell + cell / 2 });
			}
		return out;
	});
</script>

<g class="tri tri--{ink}" class:tri--focus={row >= 0} transform="translate({x} {y})">
	{#each dots as d (d.i * 64 + d.j)}
		<circle
			class="tri__dot"
			class:tri__dot--neg={signed && d.v < 0}
			class:tri__dot--on={d.i === row}
			class:tri__dot--cell={d.i === row && d.j === col}
			cx={d.cx}
			cy={d.cy}
			r={d.r}
			style:opacity={d.above ? upper : null}
		/>
	{/each}
	{#if onrow}
		{#each { length: n } as _, i (i)}
			<rect
				class="tri__hit"
				x="0"
				y={i * cell}
				width={(i + 1) * cell}
				height={cell}
				role="button"
				tabindex="-1"
				aria-label="row {i + 1}"
				onpointerenter={() => onrow(i, 'hover')}
				onclick={() => onrow(i, 'select')}
				onkeydown={() => {}}
			/>
		{/each}
	{/if}
</g>

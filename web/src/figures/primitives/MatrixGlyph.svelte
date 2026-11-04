<script>
	// A matrix as a grid of area-encoded squares (a Hinton diagram in our inks).
	// Square area ∝ |w_ij| / vmax; filled = positive, open = negative. Covector ink
	// fills with hatching instead of solid ink. Rows top→bottom, columns left→right,
	// exactly as the matrix is written. `get(i,j)` or `rows` (array of rows).
	let {
		rows = null,
		get = null,
		m = rows?.length ?? 0,
		n = rows?.[0]?.length ?? 0,
		x = 0,
		y = 0,
		cell = 9,
		vmax = 1,
		ink = 'value',
		grid = false,
		highlightRow = -1,
		highlightCol = -1
	} = $props();

	const at = (i, j) => (get ? get(i, j) : rows[i][j]);
	const side = (v) => Math.max(0, cell * 0.92 * Math.sqrt(Math.min(1, Math.abs(v) / vmax)));
	const cells = $derived.by(() => {
		const out = [];
		for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) {
			const v = at(i, j), s = side(v);
			out.push({ i, j, v, s, x: j * cell + (cell - s) / 2, y: i * cell + (cell - s) / 2 });
		}
		return out;
	});
</script>

<g class="matrix matrix--{ink}" transform="translate({x} {y})">
	{#if grid}
		<rect class="matrix__frame" x="0" y="0" width={n * cell} height={m * cell} />
	{/if}
	{#each cells as c (c.i * 4096 + c.j)}
		<rect
			class="matrix__cell"
			class:matrix__cell--neg={c.v < 0}
			class:matrix__cell--hl={c.i === highlightRow || c.j === highlightCol}
			x={c.x}
			y={c.y}
			width={c.s}
			height={c.s}
		/>
	{/each}
</g>

<script module>
	/**
	 * Layout of a lens grid: the text's positions packed into blocks of whole lines (a line
	 * ends after a line break), each block at most `maxCols` positions wide; a line longer
	 * than that is broken after its last space that fits. Returns { blocks: [[t…]], cols,
	 * cellW, blockH, height } for `rows` grid rows plus the input and `next` rows.
	 */
	export function lensGridLayout({ text = '', width = 600, minCell = 17, maxCell = 26, labelW = 52, rows = 9, cellH = 18, gap = 18, extra = 2 }) {
		const chars = Array.from(text);
		const maxCols = Math.max(4, Math.floor((width - labelW) / minCell));
		// lines: positions up to and including each line break
		const lines = [];
		let cur = [];
		chars.forEach((c, t) => { cur.push(t); if (c === '\n') { lines.push(cur); cur = []; } });
		if (cur.length) lines.push(cur);
		// break long lines after a space
		const pieces = [];
		for (const ln of lines) {
			let rest = ln;
			while (rest.length > maxCols) {
				let cut = -1;
				for (let k = maxCols - 1; k > 0; k--) if (chars[rest[k]] === ' ') { cut = k; break; }
				if (cut < 0) cut = maxCols - 1;
				pieces.push(rest.slice(0, cut + 1));
				rest = rest.slice(cut + 1);
			}
			if (rest.length) pieces.push(rest);
		}
		// pack whole pieces into blocks
		const blocks = [];
		let blk = [];
		for (const pc of pieces) {
			if (blk.length + pc.length > maxCols && blk.length) { blocks.push(blk); blk = []; }
			blk = blk.concat(pc);
		}
		if (blk.length) blocks.push(blk);
		const cols = Math.max(...blocks.map((b) => b.length));
		const cellW = Math.min(maxCell, (width - labelW) / cols);
		const blockH = (rows + extra) * cellH;
		return { blocks, cols, cellW, cellH, labelW, blockH, gap, height: blocks.length * blockH + (blocks.length - 1) * gap };
	}

	/** A character as set in a lens grid: the space as ␣, a line break as ¶. */
	export const gridChar = (c) => (c === ' ' ? '␣' : c === '\n' ? '¶' : c);

	/** Type weight for a probability: 200 (p → 0) to 900 (p = 1), linear in p. */
	export const weightOf = (p) => Math.round(200 + 700 * Math.max(0, Math.min(1, p)));
</script>

<script>
	// LensGrid — a lens read along a text, set as type (chapter 3 of *Poised to speak*).
	// Columns are the positions of the text, as in the stream (the column is the token); rows
	// are the depths, top to bottom (forward runs down). Each cell holds the character the lens
	// ranks first at that position and depth, its weight of type ∝ its probability (one
	// reference for the whole grid: 200 at p → 0, 900 at p = 1). Above the rows: the text
	// itself; below them: the character that actually comes next. Long texts wrap in whole
	// lines. Pointing at a cell calls onpoint(r, t).
	// Props: text; cells[r][t] = { ch, p }; rowLabels[r]; layout (lensGridLayout); x, y;
	// sel = { r, t } (the cell pointed at); size (font size of the cells).
	let {
		text = '',
		cells = [],
		rowLabels = [],
		layout,
		x = 0,
		y = 0,
		sel = null,
		size = 14,
		inputLabel = 'text',
		nextLabel = 'next',
		onpoint = null
	} = $props();

	const chars = $derived(Array.from(text));
	const R = $derived(cells.length);
	const cx = (k) => x + layout.labelW + layout.cellW * (k + 0.5);
	const by = (b) => y + b * (layout.blockH + layout.gap);
	const ry = (b, r) => by(b) + layout.cellH * (r + 1.5) + size * 0.36; // r = −1 is the text row
</script>

<g class="lgrid">
	{#each layout.blocks as blk, b (b)}
		<!-- row labels -->
		<text class="lbl lbl--micro" x={x + layout.labelW - 10} y={ry(b, -1)} text-anchor="end">{inputLabel}</text>
		{#each rowLabels as lab, r (r)}
			<text class="lbl lbl--micro lgrid__rlab" class:lbl--strong={sel && sel.r === r} x={x + layout.labelW - 10} y={ry(b, r)} text-anchor="end">{lab}</text>
		{/each}
		<text class="lbl lbl--micro" x={x + layout.labelW - 10} y={ry(b, R)} text-anchor="end">{nextLabel}</text>
		<line class="rule" x1={x + layout.labelW - 4} x2={cx(blk.length - 1) + layout.cellW / 2} y1={ry(b, -1) + layout.cellH * 0.5 - size * 0.36} y2={ry(b, -1) + layout.cellH * 0.5 - size * 0.36} />
		<line class="rule" x1={x + layout.labelW - 4} x2={cx(blk.length - 1) + layout.cellW / 2} y1={ry(b, R - 1) + layout.cellH * 0.5 - size * 0.36} y2={ry(b, R - 1) + layout.cellH * 0.5 - size * 0.36} />
		{#each blk as t, k (t)}
			{@const on = sel && sel.t === t}
			<text class="tok lgrid__in" class:is-on={on} x={cx(k)} y={ry(b, -1)} text-anchor="middle" font-size={size}>{gridChar(chars[t])}</text>
			{#each cells as row, r (r)}
				{@const c = row[t]}
				<text class="tok lgrid__cell" x={cx(k)} y={ry(b, r)} text-anchor="middle" font-size={size} font-weight={weightOf(c.p)}>{gridChar(c.ch)}</text>
				{#if on && sel.r === r}
					<line class="lgrid__sel" x1={cx(k) - layout.cellW * 0.4} x2={cx(k) + layout.cellW * 0.4} y1={ry(b, r) + 3.5} y2={ry(b, r) + 3.5} />
				{/if}
			{/each}
			<text class="tok lgrid__next" x={cx(k)} y={ry(b, R)} text-anchor="middle" font-size={size}>{t + 1 < chars.length ? gridChar(chars[t + 1]) : ''}</text>
			{#if onpoint}
				{#each cells as _, r (r)}
					<rect class="hit" x={cx(k) - layout.cellW / 2} y={ry(b, r) - layout.cellH * 0.5 - size * 0.36} width={layout.cellW} height={layout.cellH} role="button" tabindex="-1" aria-label="position {t + 1}, row {rowLabels[r]}" onpointerenter={() => onpoint(r, t)} />
				{/each}
			{/if}
		{/each}
	{/each}
</g>

<style>
	.lgrid__in { fill: var(--ink-2); }
	.lgrid__in.is-on { fill: var(--ink); font-weight: 650; }
	.lgrid__cell { fill: var(--ink); font-variation-settings: normal; }
	.lgrid__sel { stroke: var(--ink); stroke-width: 1px; }
	.lgrid__next { fill: var(--ink-3); font-style: italic; }
	.lgrid__rlab { font-variant-numeric: tabular-nums; }
</style>

<script>
	// "How to read": one enlarged instance of a glyph, with straight hairline leaders
	// from a dot on the mark to a plain-language note. Shown the first time a glyph
	// appears in the book. The glyph is drawn by the `glyph` snippet in the key's
	// coordinates. A note: { ax, ay, side: 'left'|'right'|'above'|'below', len, text, align }
	// — the leader runs from the anchor (ax, ay) `len` px towards `side`; the text
	// continues from there. Multi-line text with '\n'.
	let { width = 240, height = 200, title = 'How to read', notes = [], glyph } = $props();

	const LH = 13.5;
	const placed = $derived(
		notes.map((n) => {
			const len = n.len ?? 18;
			const lines = String(n.text).split('\n');
			let x2 = n.ax, y2 = n.ay, tx, ty, anchor;
			if (n.side === 'left') { x2 = n.ax - len; tx = x2 - 4; ty = n.ay + 3.8 - ((lines.length - 1) * LH) / 2; anchor = 'end'; }
			else if (n.side === 'above') { y2 = n.ay - len; tx = n.ax + (n.dx ?? 0); ty = y2 - 5 - (lines.length - 1) * LH; anchor = n.align ?? 'middle'; }
			else if (n.side === 'below') { y2 = n.ay + len; tx = n.ax + (n.dx ?? 0); ty = y2 + 12; anchor = n.align ?? 'middle'; }
			else { x2 = n.ax + len; tx = x2 + 4; ty = n.ay + 3.8 - ((lines.length - 1) * LH) / 2; anchor = 'start'; }
			return { ...n, lines, x2, y2, tx, ty, anchor };
		})
	);
</script>

<div class="key" role="note" aria-label="{title}: {notes.map((n) => n.text.replace(/\n/g, ' ')).join('; ')}">
	<p class="key__title">{title}</p>
	<svg class="key__svg" viewBox="0 0 {width} {height}" width={width} height={height} aria-hidden="true">
		{@render glyph?.()}
		{#each placed as n, k (k)}
			<line class="key__leader" x1={n.ax} y1={n.ay} x2={n.x2} y2={n.y2} />
			{#if n.dot !== false}<circle class="key__anchor" cx={n.ax} cy={n.ay} r="1.5" />{/if}
			<text class="key__text" x={n.tx} y={n.ty} text-anchor={n.anchor}>
				{#each n.lines as line, i (i)}<tspan x={n.tx} dy={i ? LH : 0}>{line}</tspan>{/each}
			</text>
		{/each}
	</svg>
</div>

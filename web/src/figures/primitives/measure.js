// use:measure={(w) => (width = w)} — the element's content width, at most once a frame.
// (bind:clientWidth re-enters ResizeObserver when the figure's height follows its width.)
export function measure(node, cb) {
	let raf = 0;
	let last = -1;
	let current = cb;
	const ro = new ResizeObserver((entries) => {
		const w = Math.round(entries[0].contentRect.width);
		cancelAnimationFrame(raf);
		raf = requestAnimationFrame(() => {
			if (w !== last && w > 0) { last = w; current(w); }
		});
	});
	ro.observe(node);
	return {
		update(next) { current = next; },
		destroy() { ro.disconnect(); cancelAnimationFrame(raf); }
	};
}

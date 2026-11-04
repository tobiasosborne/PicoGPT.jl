<script>
	// The corpus as one line of type, with a window of T characters bracketed beneath it
	// and one position in the window chosen: the characters up to it are the context
	// (underlined); optionally the character after it, the target, is marked.
	// Interaction: click a character to choose it; drag the bracket to move the window;
	// keys ←/→ move the position (the window follows), shift+←/→ move the window.
	import { showChar, clamp } from './scales.js';

	let {
		corpus = '',
		T = 8,
		start = $bindable(0), // 0-based window start
		pos = $bindable(0), // 0-based position in the window (the last context character)
		maxStart = Math.max(0, corpus.length - T - 1),
		x = 0,
		y = 0, // baseline of the type
		width = 480,
		showTarget = false,
		interactive = true,
		label = 'the corpus'
	} = $props();

	const n = $derived(corpus.length);
	const step = $derived(width / n);
	const cx = (k) => x + step * (k + 0.5);
	const target = $derived(start + pos + 1);

	let dragging = $state(false);
	let grabK = 0;
	function kAt(e) {
		const svg = e.currentTarget.ownerSVGElement ?? e.currentTarget.closest('svg');
		const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
		const p = pt.matrixTransform(svg.getScreenCTM().inverse());
		return clamp(Math.floor((p.x - x) / step), 0, n - 1);
	}
	function choose(k) {
		const s = clamp(start, k - T + 1, k);
		start = clamp(s, 0, maxStart);
		pos = clamp(k - start, 0, T - 1);
	}
	function down(e) {
		dragging = true; grabK = kAt(e) - start;
		e.currentTarget.setPointerCapture(e.pointerId);
	}
	function move(e) { if (dragging) start = clamp(kAt(e) - grabK, 0, maxStart); }
	function up() { dragging = false; }
	function key(e) {
		if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
			const d = e.key === 'ArrowRight' ? 1 : -1;
			if (e.shiftKey) start = clamp(start + d, 0, maxStart);
			else choose(clamp(start + pos + d, 0, Math.min(n - 1, maxStart + T - 1 - (showTarget ? 0 : 0))));
			e.preventDefault();
		}
	}
</script>

<g class="cstrip" class:is-live={interactive}>
	{#each Array.from(corpus) as ch, k (k)}
		{@const inWin = k >= start && k < start + T}
		{@const inCtx = k >= start && k <= start + pos}
		<text class="tok cstrip__ch" class:is-win={inWin} class:is-ctx={inCtx} class:is-last={k === start + pos} class:is-target={showTarget && k === target} x={cx(k)} y={y} text-anchor="middle">{showChar(ch)}</text>
		{#if interactive}
			<rect class="hit" x={cx(k) - step / 2} y={y - 20} width={step} height="26" role="button" tabindex="0" aria-label="choose ‘{showChar(ch)}’ at {k + 1}" onclick={() => choose(k)} onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); choose(k); } }} />
		{/if}
	{/each}
	<!-- context underline, window bracket, target mark -->
	<line class="cstrip__ctx" x1={cx(start) - step * 0.42} x2={cx(start + pos) + step * 0.42} y1={y + 5} y2={y + 5} />
	<path class="cstrip__bracket" d="M{cx(start) - step / 2 + 1} {y + 10} v5 H{cx(start + T - 1) + step / 2 - 1} v-5" />
	{#if showTarget && target < n}
		<path class="cstrip__caret" d="M{cx(target) - 4} {y - 25} L{cx(target)} {y - 20} L{cx(target) + 4} {y - 25}" /><!-- clear of an ascender (b, d, t) -->
	{/if}
	{#if interactive}
		<rect class="hit hit--drag" x={cx(start) - step / 2} y={y + 8} width={step * T} height="14" role="slider" tabindex="0" aria-label="{label}: window {start + 1}, position {pos + 1}" aria-valuenow={start + 1} aria-valuemin="1" aria-valuemax={maxStart + 1} onpointerdown={down} onpointermove={move} onpointerup={up} onkeydown={key} />
	{/if}
</g>

<style>
	.cstrip__ch { font-size: 17px; fill: var(--ink-3); }
	.cstrip__ch.is-win { fill: var(--ink-2); }
	.cstrip__ch.is-ctx { fill: var(--ink); }
	.cstrip__ch.is-last { font-weight: 650; }
	.cstrip__ch.is-target { fill: var(--value-deep); font-weight: 600; }
	.cstrip__ctx { stroke: var(--ink); stroke-width: 1px; }
	.cstrip__bracket { fill: none; stroke: var(--ink-3); stroke-width: var(--hair); }
	.is-live:hover .cstrip__bracket { stroke: var(--value); }
	.cstrip__caret { fill: none; stroke: var(--value); stroke-width: 1px; }
</style>

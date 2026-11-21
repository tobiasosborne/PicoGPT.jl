<script>
	// TickTrack — a hairline track with labelled ticks, for choosing one of a few discrete
	// values by direct manipulation (α, a depth, k): click a tick or drag the handle along the
	// line (it snaps to the nearest tick); ←/→ step when the track has focus. The chosen tick
	// carries the handle and a bold label. Draw it only in interactive mode: a print figure
	// shows its states as small multiples instead. (Shared by *Poised to speak*, chapters 5–6.)
	// options: [{ value, label }]; xs (optional): the tick positions; otherwise even spacing.
	import { clamp } from '#figures/primitives/scales.js';

	let {
		options = [],
		value = $bindable(null),
		x = 0,
		y = 0,
		width = 200,
		xs = null,
		name = '',
		labelBelow = true,
		ink = 'value'
	} = $props();

	const n = $derived(options.length);
	const pos = $derived(xs ?? options.map((_, k) => x + (n > 1 ? (width * k) / (n - 1) : width / 2)));
	const index = $derived(Math.max(0, options.findIndex((o) => o.value === value)));
	let dragging = $state(false);

	function nearest(px) {
		let best = 0;
		for (let k = 1; k < n; k++) if (Math.abs(pos[k] - px) < Math.abs(pos[best] - px)) best = k;
		return best;
	}
	function svgX(e) {
		const svg = e.currentTarget.ownerSVGElement ?? e.currentTarget.closest('svg');
		const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
		return pt.matrixTransform(svg.getScreenCTM().inverse()).x;
	}
	function down(e) { dragging = true; e.currentTarget.setPointerCapture(e.pointerId); value = options[nearest(svgX(e))].value; }
	function move(e) { if (dragging) value = options[nearest(svgX(e))].value; }
	function up() { dragging = false; }
	function key(e) {
		if (e.key === 'ArrowLeft' || e.key === 'ArrowRight' || e.key === 'Home' || e.key === 'End') {
			const k = e.key === 'Home' ? 0 : e.key === 'End' ? n - 1 : clamp(index + (e.key === 'ArrowRight' ? 1 : -1), 0, n - 1);
			value = options[k].value;
			e.preventDefault();
		}
	}
</script>

<g class="ttrack ttrack--{ink}">
	{#if name}<text class="lbl lbl--micro ttrack__name" x={pos[0] - 14} y={y + 4} text-anchor="end">{name}</text>{/if}
	<line class="axis" x1={pos[0]} x2={pos[n - 1]} y1={y} y2={y} />
	{#each options as o, k (k)}
		<line class="axis" x1={pos[k]} x2={pos[k]} y1={y - 4} y2={y + 4} />
		<text class="lbl lbl--micro ttrack__lbl" class:is-on={k === index} x={pos[k]} y={labelBelow ? y + 17 : y - 9} text-anchor="middle">{o.label}</text>
	{/each}
	<circle class="handle" class:handle--active={dragging} cx={pos[index]} cy={y} r="5.5" />
	<rect
		class="hit hit--drag"
		x={pos[0] - 12}
		y={y - 12}
		width={pos[n - 1] - pos[0] + 24}
		height={labelBelow ? 34 : 24}
		role="slider"
		tabindex="0"
		aria-label={name || 'choose'}
		aria-valuenow={index + 1}
		aria-valuemin="1"
		aria-valuemax={n}
		aria-valuetext={options[index]?.label}
		onpointerdown={down}
		onpointermove={move}
		onpointerup={up}
		onkeydown={key}
	/>
</g>

<style>
	.ttrack__lbl { transition: fill var(--t-quick); }
	.ttrack__lbl.is-on { fill: var(--ink); font-weight: 650; }
	.ttrack__name { font-style: italic; font-family: var(--serif); font-size: 13px; fill: var(--ink-2); }
</style>

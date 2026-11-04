<script>
	// The loss curve (DESIGN.md §5): training step against the loss in nats, drawn as a
	// <g> inside a figure's own <svg>. Every vertex is one recorded loss.
	//
	// Props
	//   x, y, width, height   the chart's box in the parent SVG (axes, labels and the
	//                         sample margin all live inside it)
	//   series     [{ values, start = 0, label = '', style = 'main' | 'ghost' | 'dash',
	//                 band = null, labelAt = null }]
	//              values[i] is the loss after start + i updates. 'main' is value ink at
	//              --line·1.3; 'dash' is value ink dashed (a second series); 'ghost' is a
	//              hairline in --ink-3 (a reference run). A label is set at the series' last
	//              point, or at step `labelAt` if given. band = { steps, lo, hi }: a min–max envelope, one vertical hairline
	//              per bin (for long runs: every spike survives the binning).
	//   xScale     'log' (default): position ∝ log(1 + step), ticks at 0, 1, 10, 100, …;
	//              'linear': ticks at round numbers.
	//   xMax       last step on the axis (xMin is 0)
	//   yScale     'nats' (default): linear from 0 to yMax;
	//              'excess': log10 of (loss − floor), from 10^yLo to 10^yHi. In this mode
	//              `values` must already be the excess.
	//   yMax, yLo, yHi
	//   lnV        draw the hairline of uniform guessing at ln V (labelled); null: none
	//   floor      draw the entropy floor (labelled with its value); null: none.
	//              Pass specimen.lab.floor; never a typed number.
	//   samples    [{ step, loss, text, prompt = '', ghost = false }]: a staircase of text rows above the
	//              plot, one row per sample in order of step, each starting at its step's x
	//              and joined to its point on the curve by a vertical leader (rows further
	//              down start further right, so no leader crosses a row). In 'excess' mode
	//              `loss` is the excess. A ghost row (e.g. a reference run's sample at a step
	//              the live run has not reached) is set in --ink-3.
	//   rows       rows to reserve above the plot (so the plot does not move as samples
	//              arrive during a live run); default: as many as there are samples
	//   margin     width kept on the right (sample rows overhang into it; the reference
	//              lines are labelled there when it is ≥ 70)
	//   cursor     a step to mark (hairline + a dot on each main series), or null
	//   onpoint    (step | null) => void: pointer/keyboard hook over the plot area
	//   xLabel, yLabel, compact (smaller ticks, for small multiples), refLabels (false: the
//              reference lines without their words, e.g. when the figure labels them)
	//
	// Exports nothing else; the "how to read" key is LossChartKey.svelte.
	import { fmt, showChar, clamp } from './scales.js';

	let {
		x = 0,
		y = 0,
		width = 520,
		height = 260,
		series = [],
		xScale = 'log',
		xMax = 200,
		yScale = 'nats',
		yMax = null,
		yLo = -6,
		yHi = 0.5,
		lnV = null,
		floor = null,
		samples = [],
		rows = 0,
		margin = 0,
		cursor = null,
		onpoint = null,
		xLabel = 'step',
		yLabel = 'nats',
		compact = false,
		refLabels = true,
		title = ''
	} = $props();

	const L = $derived(compact ? 26 : 34);
	const R = $derived(margin + (compact ? 6 : 10));
	const rowGap = $derived(compact ? 12 : 15);
	const band = $derived(Math.max(rows, samples.length) ? Math.max(rows, samples.length) * rowGap + 8 : 0);
	const Tp = $derived((compact ? 12 : 16) + band);
	const sideRefs = $derived(margin >= 70);
	const B = $derived(compact ? 20 : 26);
	const pw = $derived(Math.max(10, width - L - R));
	const ph = $derived(Math.max(10, height - Tp - B));

	const excess = $derived(yScale === 'excess');
	const top = $derived.by(() => {
		if (yMax != null) return yMax;
		let m = lnV ?? 0;
		for (const s of series) for (const v of s.values) if (Number.isFinite(v)) m = Math.max(m, v);
		return Math.ceil(m * 2 + 0.25) / 2;
	});

	const X = (k) => {
		const u = xScale === 'log' ? Math.log1p(Math.max(0, k)) / Math.log1p(xMax) : k / xMax;
		return L + pw * clamp(u, 0, 1);
	};
	const invX = (px) => {
		const u = clamp((px - L) / pw, 0, 1);
		return Math.round(xScale === 'log' ? Math.expm1(u * Math.log1p(xMax)) : u * xMax);
	};
	const Y = (v) => {
		if (excess) {
			const lv = v > 0 ? Math.log10(v) : yLo;
			return Tp + ph * (1 - clamp((lv - yLo) / (yHi - yLo), 0, 1));
		}
		return Tp + ph * (1 - clamp(v / top, 0, 1));
	};

	const xTicks = $derived.by(() => {
		if (xScale === 'log') {
			const out = [0];
			for (let k = 1; k <= xMax; k *= 10) out.push(k);
			if (X(xMax) - X(out.at(-1)) > 26) out.push(xMax);
			return out;
		}
		const raw = xMax / 5, mag = 10 ** Math.floor(Math.log10(raw));
		const stepT = [1, 2, 2.5, 5, 10].map((f) => f * mag).find((s) => s >= raw) ?? raw;
		const out = [];
		for (let k = 0; k <= xMax + 1e-9; k += stepT) out.push(k);
		return out;
	});
	const yTicks = $derived.by(() => {
		if (excess) { const out = []; for (let d = Math.ceil(yLo); d <= Math.floor(yHi); d++) out.push(d); return out; }
		const stepT = top > 3 ? 1 : 0.5;
		const out = [];
		for (let v = 0; v <= top + 1e-9; v += stepT) out.push(+v.toFixed(2));
		return out;
	});
	const sup = (d) => String(d).replace('-', '−').replace(/[0-9]/g, (c) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+c]).replace('−', '⁻');

	function path(s) {
		let d = '', pen = false;
		for (let i = 0; i < s.values.length; i++) {
			const v = s.values[i], k = (s.start ?? 0) + i;
			if (!Number.isFinite(v) || (excess && !(v > 0))) { pen = false; continue; }
			d += `${pen ? 'L' : 'M'}${X(k).toFixed(1)} ${Y(v).toFixed(1)}`;
			pen = true;
		}
		return d;
	}
	function bandPath(b) {
		let d = '';
		for (let i = 0; i < b.steps.length; i++) {
			const x1 = X(b.steps[i]).toFixed(1);
			const y1 = Y(b.hi[i]), y2 = Y(b.lo[i]);
			d += `M${x1} ${y1.toFixed(1)}V${Math.max(y2, y1 + 0.6).toFixed(1)}`;
		}
		return d;
	}
	const lastOf = (s) => {
		for (let i = s.values.length - 1; i >= 0; i--) if (Number.isFinite(s.values[i])) return { k: (s.start ?? 0) + i, v: s.values[i] };
		return null;
	};
	const valueAt = (s, k) => {
		const i = k - (s.start ?? 0);
		return i >= 0 && i < s.values.length ? s.values[i] : null;
	};

	// samples: a staircase above the plot
	const placed = $derived(
		samples
			.filter((r) => Number.isFinite(r.loss))
			.slice()
			.sort((a, b) => a.step - b.step)
			.map((r, i) => ({ ...r, px: X(r.step), py: Y(r.loss), ty: Tp - band + 4 + (i + 1) * rowGap - 3 }))
	);

	function move(e) {
		if (!onpoint) return;
		const svg = e.currentTarget.ownerSVGElement;
		const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
		const p = pt.matrixTransform(svg.getScreenCTM().inverse());
		onpoint(invX(p.x - x));
	}
	function key(e) {
		if (!onpoint) return;
		const k = cursor ?? 0;
		const step = e.shiftKey ? 10 : 1;
		if (e.key === 'ArrowRight') { onpoint(Math.min(xMax, k + step)); e.preventDefault(); }
		if (e.key === 'ArrowLeft') { onpoint(Math.max(0, k - step)); e.preventDefault(); }
		if (e.key === 'Escape') onpoint(null);
	}
	const chars = (t) => Array.from(t ?? '');
</script>

<g class="loss" transform="translate({x} {y})">
	{#if title}<text class="lbl lbl--head" x={L} y={Tp - 4}>{title}</text>{/if}

	<!-- axes -->
	<line class="axis" x1={L} x2={L + pw} y1={Tp + ph} y2={Tp + ph} />
	<line class="axis" x1={L} x2={L} y1={Tp} y2={Tp + ph} />
	{#each xTicks as k (k)}
		<line class="axis" x1={X(k)} x2={X(k)} y1={Tp + ph} y2={Tp + ph + 3.5} />
		<text class="lbl lbl--micro" x={X(k)} y={Tp + ph + (compact ? 12 : 14)} text-anchor="middle">{k}</text>
	{/each}
	{#if xLabel && !compact}
		<text class="lbl lbl--micro loss__xl" x={L + pw} y={Tp + ph + 25} text-anchor="end">{xLabel}{xScale === 'log' ? ' (logarithmic)' : ''}</text>
	{/if}
	{#each yTicks as v (v)}
		<line class="axis" x1={L - 3.5} x2={L} y1={excess ? Y(10 ** v) : Y(v)} y2={excess ? Y(10 ** v) : Y(v)} />
		<text class="lbl lbl--micro" x={L - 6} y={(excess ? Y(10 ** v) : Y(v)) + 3.5} text-anchor="end">{excess ? `10${sup(v)}` : fmt(v, 1)}</text>
	{/each}
	{#if yLabel}
		<text class="lbl lbl--micro" x={L - 6} y={Tp - (compact ? 3 : 6)} text-anchor="end">{yLabel}</text>
	{/if}

	<!-- reference lines -->
	{#if lnV != null}
		{@const yy = excess ? Y(lnV - (floor ?? 0)) : Y(lnV)}
		<line class="loss__ref" x1={L} x2={L + pw} y1={yy} y2={yy} />
		{#if refLabels && sideRefs}
			<text class="lbl lbl--micro loss__reftext" x={L + pw + 6} y={yy + 3.5}>uniform guessing<tspan x={L + pw + 6} dy="12">ln V = {fmt(lnV, 2)}</tspan></text>
		{:else if refLabels}
			<text class="lbl lbl--micro loss__reftext" x={L + pw - 2} y={yy - 4} text-anchor="end">uniform guessing, ln V = {fmt(lnV, 2)}</text>
		{/if}
	{/if}
	{#if floor != null && !excess}
		<line class="loss__ref loss__ref--floor" x1={L} x2={L + pw} y1={Y(floor)} y2={Y(floor)} />
		{#if refLabels && sideRefs}
			<text class="lbl lbl--micro loss__reftext" x={L + pw + 6} y={Y(floor) - 9}>entropy floor<tspan x={L + pw + 6} dy="12">{fmt(floor, 4)}</tspan></text>
		{:else if refLabels}
			<text class="lbl lbl--micro loss__reftext" x={L + pw - 2} y={Y(floor) - 4} text-anchor="end">entropy floor {fmt(floor, 4)}</text>
		{/if}
	{/if}

	<!-- data -->
	<g class="loss__marks">
		{#each series as s, k (k)}
			{#if s.band}
				<path class="loss__band" d={bandPath(s.band)} />
			{/if}
			{#if s.values?.length}
				<path class="loss__line loss__line--{s.style ?? 'main'}" d={path(s)} />
			{/if}
			{#if s.label}
				{@const p = s.labelAt != null && Number.isFinite(valueAt(s, s.labelAt)) ? { k: s.labelAt, v: valueAt(s, s.labelAt) } : lastOf(s)}
				{#if p}
					<text class="lbl lbl--micro loss__label loss__label--{s.style ?? 'main'}" x={X(p.k) + 4} y={Y(p.v) - 5}>{s.label}</text>
				{/if}
			{/if}
		{/each}
	</g>

	<!-- samples: rows above the plot, a vertical leader down to the curve -->
	{#each placed as r (r.step)}
		<g class:loss__ghostrow={r.ghost}>
		<line class="leader" x1={r.px} y1={r.ty + 3} x2={r.px} y2={r.py - 2.5} />
		<circle class="loss__sdot" cx={r.px} cy={r.py} r="1.9" />
		<text class="lbl lbl--micro loss__step" x={r.px + 3} y={r.ty}>{r.step}</text>
		<text class="loss__sample" x={r.px + (compact ? 18 : 24)} y={r.ty}
			>{#each chars(r.prompt) as c, i (i)}<tspan class="loss__prompt" class:loss__sp={c === ' '}>{showChar(c)}</tspan>{/each}{#each chars(r.text) as c, i (i)}<tspan class:loss__sp={c === ' '}>{showChar(c)}</tspan>{/each}</text
		>
		</g>
	{/each}

	<!-- cursor -->
	{#if cursor != null}
		<line class="loss__cursor" x1={X(cursor)} x2={X(cursor)} y1={Tp} y2={Tp + ph} />
		{#each series as s, k (k)}
			{@const v = (s.style ?? 'main') !== 'ghost' ? valueAt(s, cursor) : null}
			{#if v != null && Number.isFinite(v)}
				<circle class="loss__cdot" cx={X(cursor)} cy={Y(v)} r="3" />
			{/if}
		{/each}
	{/if}

	{#if onpoint}
		<rect
			class="hit loss__hit"
			x={L}
			y={Tp}
			width={pw}
			height={ph}
			role="slider"
			tabindex="0"
			aria-label="training step"
			aria-valuemin="0"
			aria-valuemax={xMax}
			aria-valuenow={cursor ?? 0}
			onpointermove={move}
			onpointerdown={move}
			onpointerleave={() => onpoint(null)}
			onkeydown={key}
		/>
	{/if}
</g>

<style>
	.loss__ref { stroke: var(--ink-3); stroke-width: var(--hair); stroke-dasharray: 2 2.5; }
	.loss__ref--floor { stroke-dasharray: none; stroke: var(--ink-2); }
	.loss__reftext { fill: var(--ink-2); paint-order: stroke; stroke: var(--paper); stroke-width: 3px; stroke-linejoin: round; }
	.loss__marks { mix-blend-mode: var(--overprint); }
	.loss__line { fill: none; stroke-linejoin: round; stroke-linecap: round; }
	.loss__line--main { stroke: var(--value); stroke-width: calc(var(--line) * 1.35); }
	.loss__line--dash { stroke: var(--value); stroke-width: var(--line); stroke-dasharray: 4 3; }
	.loss__line--ghost { stroke: var(--ink-3); stroke-width: var(--hair); }
	.loss__band { stroke: var(--value); stroke-width: var(--hair); fill: none; }
	.loss__label--main, .loss__label--dash { fill: var(--value-deep); }
	.loss__label--ghost { fill: var(--ink-3); }
	.loss__sdot { fill: var(--value); }
	.loss__step { fill: var(--ink-3); }
	.loss__ghostrow .loss__sample { fill: var(--ink-3); }
	.loss__ghostrow .loss__prompt, .loss__ghostrow .loss__sp { opacity: 0.6; }
	.loss__ghostrow .loss__sdot { fill: var(--ink-3); }
	.loss__ghostrow .leader { opacity: 0.55; }
	.loss__sample { font-family: var(--serif); font-size: 13px; fill: var(--ink); font-variant-ligatures: none; }
	.loss__prompt { fill: var(--ink-3); }
	.loss__sp { fill: var(--ink-3); font-size: 0.85em; }
	.loss__cursor { stroke: var(--ink); stroke-width: var(--hair); }
	.loss__cdot { fill: var(--paper); stroke: var(--value); stroke-width: 1.4px; }
	.loss__hit { cursor: crosshair; }
</style>

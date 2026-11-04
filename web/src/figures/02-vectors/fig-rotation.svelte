<script module>
	export const meta = { size: 'wide', title: 'A shift is a rotation' };
</script>

<script>
	// Figure: what does a shift do to the positional code?
	// Each pair of rows (sin, cos) of P is drawn as a clock: position t is the point
	// (P_{2r+1,t}, P_{2r+2,t}) = (sin φ, cos φ), φ = ω_r(t−1), plotted with sin to the right
	// and cos up, so φ runs clockwise from twelve o'clock like a clock hand. The hands of
	// P_t (ink) and P_{t+k} (value ink) and the arc between them, ω_r k. Right: the Gram
	// matrix PᵀP, constant along diagonals; the diagonal of offset k is outlined.
	// State: t, k (attrs pos=, shift=). Drag a hand round any clock (t), drag along the
	// shift scale (k); ←/→ change t, ↑/↓ change k.
	import { getContext } from 'svelte';
	import { measure } from '../primitives/measure.js';
	import { fmt, clamp, layoutWidth } from '../primitives/scales.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const D = $derived(specimen.config.D);
	const T = $derived(specimen.config.T);
	const P = $derived(specimen.P);
	const R = $derived(D / 2);
	// svelte-ignore state_referenced_locally
	let t = $state(clamp((+attrs.pos || 2) - 1, 0, specimen.config.T - 1));
	// svelte-ignore state_referenced_locally
	let k = $state(clamp(+attrs.shift || 3, 0, specimen.config.T - 1));
	const kk = $derived(Math.min(k, T - 1 - t)); // t + k must be a position
	let width = $state(868);
	const interactive = $derived(mode !== 'print');

	const omega = (r) => Math.pow(10000, (-2 * r) / D);
	const fmtOmega = (w) => String(+w.toPrecision(3));
	const phi = (r, tt) => Math.atan2(P[tt][2 * r], P[tt][2 * r + 1]); // from the numbers of P
	const gram = $derived(P.map((ps) => P.map((pt) => ps.reduce((s, v, i) => s + v * pt[i], 0))));
	const gmax = $derived(R); // the diagonal, |P_t|² = D/2

	// geometry
	const W = $derived(layoutWidth(width, mode, 330));
	const narrow = $derived(W < 660);
	const gcell = $derived(narrow ? Math.min(26, (W - 60) / T) : 25);
	const gW = $derived(gcell * T);
	const clocksW = $derived(narrow ? W : W - gW - 90);
	const cols = $derived(narrow ? 2 : R);
	const rowsN = $derived(Math.ceil(R / cols));
	const pitch = $derived(clocksW / cols);
	const rc = $derived(Math.min(narrow ? 62 : 74, pitch * 0.36));
	const cy0 = 44 + 30;
	const rowH = $derived(2 * rc + 96);
	const clockAt = (r) => ({ x: pitch * ((r % cols) + 0.5), y: cy0 + rc + Math.floor(r / cols) * rowH });
	const yShift = $derived(cy0 + rowsN * rowH - 28);
	const shiftX0 = $derived(narrow ? 40 : 60);
	const shiftX1 = $derived(clocksW - 40);
	const xk = (v) => shiftX0 + ((shiftX1 - shiftX0) * v) / (T - 1);
	const gx = $derived(narrow ? (W - gW) / 2 : W - gW - 14);
	const gy = $derived(narrow ? yShift + 64 : cy0 - 6);
	const H = $derived(narrow ? gy + gW + 40 : Math.max(yShift + 34, gy + gW + 52));

	const pt = (c, ang, rr = rc) => ({ x: c.x + rr * Math.sin(ang), y: c.y - rr * Math.cos(ang) });
	function arcPath(c, a0, a1, rr) {
		const p0 = pt(c, a0, rr), p1 = pt(c, a1, rr);
		let d = a1 - a0;
		const large = Math.abs(d) % (2 * Math.PI) > Math.PI ? 1 : 0;
		if (Math.abs(d) < 1e-9) return '';
		// for turns longer than a full circle, draw a full circle plus the remainder
		if (Math.abs(d) >= 2 * Math.PI) {
			const pm = pt(c, a0 + Math.PI, rr);
			return `M${p0.x},${p0.y} A${rr},${rr} 0 1 1 ${pm.x},${pm.y} A${rr},${rr} 0 1 1 ${p0.x},${p0.y} ` + arcPath(c, a0, a0 + (d % (2 * Math.PI)), rr).replace(/^M[^A]*/, '');
		}
		return `M${p0.x},${p0.y} A${rr},${rr} 0 ${large} 1 ${p1.x},${p1.y}`;
	}

	// interaction: drag a hand (sets t), drag the shift handle (sets k)
	let dragging = $state(null);
	function svgPoint(svg, e) { const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY; return p.matrixTransform(svg.getScreenCTM().inverse()); }
	function nearestT(r, ang) {
		let best = 0, bd = Infinity;
		for (let tt = 0; tt < T; tt++) {
			const d = Math.abs(Math.atan2(Math.sin(ang - phi(r, tt)), Math.cos(ang - phi(r, tt))));
			if (d < bd) { bd = d; best = tt; }
		}
		return best;
	}
	function clockDown(r, e) { dragging = { kind: 'clock', r }; e.currentTarget.setPointerCapture(e.pointerId); clockMove(e); }
	function clockMove(e) {
		if (dragging?.kind !== 'clock') return;
		const c = clockAt(dragging.r), p = svgPoint(e.currentTarget.ownerSVGElement, e);
		t = nearestT(dragging.r, Math.atan2(p.x - c.x, c.y - p.y));
	}
	function shiftDown(e) { dragging = { kind: 'shift' }; e.currentTarget.setPointerCapture(e.pointerId); shiftMove(e); }
	function shiftMove(e) {
		if (dragging?.kind !== 'shift') return;
		const p = svgPoint(e.currentTarget.ownerSVGElement, e);
		k = clamp(Math.round(((p.x - shiftX0) / (shiftX1 - shiftX0)) * (T - 1)), 0, T - 1 - t);
	}
	function up() { dragging = null; }
	function key(e) {
		if (!interactive) return;
		if (e.key === 'ArrowLeft') { t = Math.max(0, t - 1); e.preventDefault(); }
		if (e.key === 'ArrowRight') { t = Math.min(T - 1 - Math.min(kk, T - 1), t + 1); if (t + k > T - 1) k = T - 1 - t; e.preventDefault(); }
		if (e.key === 'ArrowUp') { k = Math.min(T - 1 - t, kk + 1); e.preventDefault(); }
		if (e.key === 'ArrowDown') { k = Math.max(0, kk - 1); e.preventDefault(); }
	}
	const deg = (a) => (a * 180) / Math.PI;
	const readout = $derived.by(() => {
		const turns = Array.from({ length: R }, (_, r) => `${fmt(deg(omega(r) * kk), omega(r) * kk < 0.1 ? 2 : 0)}°`).join(', ');
		return {
			who: `t = ${t + 1}, k = ${kk}: P${'₀₁₂₃₄₅₆₇₈₉'[t + 1]} → P${t + 1 + kk < 10 ? '₀₁₂₃₄₅₆₇₈₉'[t + 1 + kk] : t + 1 + kk}.`,
			part: `The shift turns the ${R} hands by ${turns}, whatever t is. P${'₀₁₂₃₄₅₆₇₈₉'[t + 1]}·P${'₀₁₂₃₄₅₆₇₈₉'[t + 1 + kk] ?? ''} = ${fmt(gram[t][t + kk], 3)}, the same for every pair ${kk} apart.`
		};
	});
</script>

<div class="fx fx-rot" use:measure={(w) => (width = w)}>
	<svg
		viewBox="0 0 {W} {H}"
		width={W}
		height={H}
		role={interactive ? 'group' : 'img'}
		tabindex={interactive ? 0 : undefined}
		aria-label="Positions as clocks. {readout.who} {readout.part}{interactive ? ' Arrow keys: ←/→ position, ↑/↓ shift.' : ''}"
		onkeydown={key}
	>
		{#each Array.from({ length: R }) as _, r (r)}
			{@const c = clockAt(r)}
			{@const a0 = phi(r, t)}
			{@const a1 = phi(r, t + kk)}
			{@const turn = omega(r) * kk}
			{@const h0 = pt(c, a0)}
			{@const h1 = pt(c, a1)}
			<g class="fx-rot__clock">
				<circle class="fx-rot__dial" cx={c.x} cy={c.y} r={rc} />
				<line class="rule" x1={c.x} x2={c.x} y1={c.y - rc - 6} y2={c.y - rc + 4} />
				<!-- every position on this clock -->
				{#each P as _, tt (tt)}
					{@const q = pt(c, phi(r, tt))}
					<circle class="fx-rot__pos" class:is-on={tt === t || tt === t + kk} cx={q.x} cy={q.y} r="2.4" />
				{/each}
				<!-- the turn ω_r k -->
				{#if kk > 0}
					<path class="fx-rot__arc" d={arcPath(c, a0, a0 + turn, rc * 0.62)} />
				{/if}
				<!-- hands: P_t in ink, P_{t+k} in value ink -->
				<line class="fx-rot__hand" x1={c.x} y1={c.y} x2={h0.x} y2={h0.y} />
				<line class="fx-rot__hand fx-rot__hand--k" x1={c.x} y1={c.y} x2={h1.x} y2={h1.y} />
				<circle class="fx-rot__tip" cx={h0.x} cy={h0.y} r="3.6" />
				<circle class="dot--value" cx={h1.x} cy={h1.y} r="3.6" />
				<circle class="spoke__hub" cx={c.x} cy={c.y} r="1.3" />
				{#if r === 0}
					{@const l0 = pt(c, a0, rc + 15)}
					{@const l1 = pt(c, a1, rc + 15)}
					<text class="lbl fx-rot__hl" x={l0.x} y={l0.y + 4} text-anchor="middle"><tspan class="m m--s">P</tspan><tspan class="m sub" dy="3">{t + 1}</tspan></text>
					{#if kk > 0}<text class="lbl fx-rot__hl" x={l1.x} y={l1.y + 4} text-anchor="middle"><tspan class="m m--s fx-rot__v">P</tspan><tspan class="m sub fx-rot__v" dy="3">{t + 1 + kk}</tspan></text>{/if}
				{/if}
				<!-- labels -->
				<text class="lbl" x={c.x} y={c.y - rc - 14} text-anchor="middle"><tspan class="m">ω</tspan><tspan class="m sub" dy="3">{r}</tspan><tspan dy="-3">{' '}= {fmtOmega(omega(r))}</tspan></text>
				<text class="lbl lbl--micro" x={c.x} y={c.y + rc + 18} text-anchor="middle">rows {2 * r + 1}, {2 * r + 2}</text>
				<text class="lbl lbl--micro lbl--value" x={c.x} y={c.y + rc + 32} text-anchor="middle">turn <tspan class="m m--s">ω</tspan><tspan class="m sub" dy="2">{r}</tspan><tspan class="m m--s" dy="-2">k</tspan> = {fmt(deg(turn), turn < 0.1 ? 2 : 0)}°</text>
				{#if interactive}
					<circle class="handle-hit" cx={c.x} cy={c.y} r={rc + 10} role="slider" tabindex="-1" aria-label="position on clock {r}" aria-valuenow={t + 1} onpointerdown={(e) => clockDown(r, e)} onpointermove={clockMove} onpointerup={up} />
				{/if}
			</g>
		{/each}

		<!-- the shift k -->
		<g class="fx-rot__shift">
			<line class="axis" x1={shiftX0} x2={shiftX1} y1={yShift} y2={yShift} />
			{#each Array.from({ length: T }) as _, v (v)}
				<line class="axis" x1={xk(v)} x2={xk(v)} y1={yShift} y2={yShift + 4} />
				<text class="lbl lbl--micro" class:lbl--strong={v === kk} class:fx-rot__off={v > T - 1 - t} x={xk(v)} y={yShift + 16} text-anchor="middle">{v}</text>
			{/each}
			<text class="lbl" x={shiftX0 - 12} y={yShift + 4} text-anchor="end">shift <tspan class="m m--s">k</tspan></text>
			<line class="fx-rot__reach" x1={xk(0)} x2={xk(kk)} y1={yShift} y2={yShift} />
			<circle class="handle" class:handle--active={dragging?.kind === 'shift'} cx={xk(kk)} cy={yShift} r="6" />
			{#if !narrow}<text class="lbl lbl--micro" x={shiftX1 + 10} y={yShift + 4}>from <tspan class="m m--s">t</tspan> = {t + 1} to {t + 1 + kk}</text>{/if}
			{#if interactive}
				<rect class="handle-hit" x={shiftX0 - 10} y={yShift - 14} width={shiftX1 - shiftX0 + 20} height="28" role="slider" tabindex="-1" aria-label="shift k" aria-valuemin="0" aria-valuemax={T - 1 - t} aria-valuenow={kk} onpointerdown={shiftDown} onpointermove={shiftMove} onpointerup={up} />
			{/if}
		</g>

		<!-- the Gram matrix PᵀP: area ∝ P_s·P_t (all positive here), constant along diagonals -->
		<g class="fx-rot__gram">
			<text class="lbl lbl--micro" x={gx} y={gy - 26}><tspan class="m m--s">P</tspan><tspan class="m sub" dy="2">s</tspan><tspan dy="-2"> · </tspan><tspan class="m m--s">P</tspan><tspan class="m sub" dy="2">t</tspan><tspan dy="-2">, row </tspan><tspan class="m m--s">s</tspan><tspan>, column </tspan><tspan class="m m--s">t</tspan></text>
			{#each Array.from({ length: T }) as _, s (s)}
				<text class="lbl lbl--micro" x={gx - 6} y={gy + gcell * (s + 0.5) + 3.5} text-anchor="end">{s + 1}</text>
				<text class="lbl lbl--micro" x={gx + gcell * (s + 0.5)} y={gy - 6} text-anchor="middle">{s + 1}</text>
			{/each}
			<rect class="matrix__frame" x={gx} y={gy} width={gW} height={gW} />
			<g class="matrix matrix--value">
				{#each gram as row, s (s)}
					{#each row as v, u (u)}
						{@const side = gcell * 0.9 * Math.sqrt(Math.min(1, Math.abs(v) / gmax))}
						<rect class="matrix__cell" class:matrix__cell--neg={v < 0} class:fx-rot__faint={Math.abs(u - s) !== kk} x={gx + gcell * (u + 0.5) - side / 2} y={gy + gcell * (s + 0.5) - side / 2} width={side} height={side} />
					{/each}
				{/each}
			</g>
			<!-- the diagonal of offset k, and the cell (t, t+k) -->
			<path class="fx-rot__diag" d={`M${gx + gcell * kk},${gy} L${gx + gW},${gy + gcell * (T - kk)}`} />
			<rect class="fx-rot__cell" x={gx + gcell * (t + kk)} y={gy + gcell * t} width={gcell} height={gcell} />
			<text class="lbl lbl--micro" x={gx + gW / 2} y={gy + gW + 20} text-anchor="middle">every pair <tspan class="m m--s">k</tspan> = {kk} apart: {fmt(gram[0][kk], 2)}</text>
			<text class="lbl lbl--micro" x={gx + gW / 2} y={gy + gW + 34} text-anchor="middle">diagonal: <tspan class="m m--s">D</tspan>/2 = {fmt(gmax, 0)}</text>
		</g>
	</svg>
	{#if interactive}
		<p class="readout" aria-live="polite">
			<span class="readout__who">{readout.who}</span>
			<span class="readout__part">{readout.part}</span>
			<span class="readout__hint">Drag a hand round its clock to move t; drag along the scale to change the shift k (keys: ←/→ and ↑/↓).</span>
		</p>
	{/if}
</div>

<style>
	.fx-rot :global(.m) { font-family: var(--serif); font-style: italic; font-size: 14px; fill: var(--ink); }
	.fx-rot :global(.m--s) { font-size: 12.5px; }
	.fx-rot :global(.sub) { font-size: 9.5px; }
	.fx-rot__dial { fill: none; stroke: var(--rule); stroke-width: var(--hair); }
	.fx-rot__pos { fill: var(--paper); stroke: var(--ink-3); stroke-width: var(--hair); }
	.fx-rot__pos.is-on { stroke: none; fill: none; }
	.fx-rot__hand { stroke: var(--ink); stroke-width: 1.3px; }
	.fx-rot__hand--k { stroke: var(--value); stroke-width: 1.6px; }
	.fx-rot__tip { fill: var(--ink); }
	.fx-rot__arc { fill: none; stroke: var(--value); stroke-width: 1.1px; }
	.fx-rot__reach { stroke: var(--value); stroke-width: 1.6px; }
	.fx-rot__off { opacity: 0.35; }
	.fx-rot__faint { opacity: 0.32; }
	.fx-rot__diag { stroke: var(--ink-3); stroke-width: var(--hair); stroke-dasharray: 1.5 2.5; fill: none; }
	.fx-rot__cell { fill: none; stroke: var(--ink); stroke-width: 1px; }
	.fx-rot__clock .handle-hit { cursor: grab; }
	.fx-rot :global(.fx-rot__v) { fill: var(--value-deep); }
	.fx-rot__hl { paint-order: stroke; stroke: var(--paper); stroke-width: 3px; }
</style>

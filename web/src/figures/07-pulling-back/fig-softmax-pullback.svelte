<script module>
	export const meta = { size: 'wide', title: 'No gradient from the future' };
</script>

<script>
	// Figure: the softmax pullback of one attention head, at θ₀, for one training window.
	//   A   the weights (value ink; nothing above the diagonal)
	//   Ā   the adjoint of the weights, Ā = Ōᵀ V: defined on the WHOLE square, future included
	//   S̄   the adjoint of the scores: S̄ᵢⱼ = Aᵢⱼ (Āᵢⱼ − Σₖ Aᵢₖ Āᵢₖ)
	// The third panel runs that computation backwards on a clock t (covector ink):
	//   Ā  →  minus its A-weighted row mean  →  times A (the future shrinks to nothing)  =  S̄.
	// Every row of S̄ sums to zero, which is why the key bias b_k never learns.
	import { getContext } from 'svelte';
	import { measure } from '../primitives/measure.js';
	import TriangleDots from '../primitives/TriangleDots.svelte';
	import Scrubber from '../primitives/Scrubber.svelte';
	import { fmt, showChar, clamp, phase, lerp, maxAbs, layoutWidth } from '../primitives/scales.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const T = $derived(specimen.config.T);
	const gt = $derived(specimen.lab.gradTrace(specimen.theta0));
	// svelte-ignore state_referenced_locally
	const win = clamp((+attrs.window || 1) - 1, 0, 9);
	// svelte-ignore state_referenced_locally
	let head = $state(clamp((+attrs.head || 1) - 1, 0, 1));
	// svelte-ignore state_referenced_locally
	let t = $state(attrs.t != null ? +attrs.t : 1);
	$effect(() => { if (attrs.t != null) t = +attrs.t; });
	let row = $state(-1);
	let pinned = $state(-1);
	let width = $state(868);
	const interactive = $derived(mode !== 'print');
	const W = $derived(layoutWidth(width, mode));

	const w = $derived(gt.windows[win]);
	const chars = $derived(w.x.map((a) => specimen.vocab[a]));
	const hd = $derived(w.heads[head]);
	const A = $derived(hd.A);
	const Abar = $derived(hd.Abar);
	const Sbar = $derived(hd.Sbar);
	// the A-weighted mean of each row of Ā (only the allowed sources carry weight)
	const cRow = $derived(Abar.map((r, i) => r.reduce((s, v, k) => s + A[i][k] * v, 0)));
	const centred = $derived(Abar.map((r, i) => r.map((v) => v - cRow[i])));
	const aMax = $derived(maxAbs(Abar));
	const cMax = $derived(Math.max(aMax, maxAbs(centred)));
	const sMax = $derived(maxAbs(Sbar));

	const PHASES = [
		{ at: 0, label: 'Ā' },
		{ at: 0.42, label: '− mean' },
		{ at: 0.72, label: '× A' },
		{ at: 1, label: 'S̄' }
	];
	/** The covector panel at time t: values and the scale they are drawn at. */
	function frame(tt) {
		const u1 = phase(tt, 0.08, 0.4), u2 = phase(tt, 0.5, 0.95);
		const M = Abar.map((r, i) => r.map((v, j) => (v - u1 * cRow[i]) * (1 - u2 + u2 * A[i][j])));
		const vmax = u2 > 0 ? lerp(cMax, sMax, u2) : lerp(aMax, cMax, u1);
		return { M, vmax, upper: tt >= 0.999 ? 0 : 1 };
	}
	const live = $derived(frame(t));
	const active = $derived(pinned >= 0 ? pinned : row);
	function onrow(i, kind) {
		if (!interactive) return;
		if (kind === 'select') pinned = pinned === i ? -1 : i;
		else row = i;
	}

	// geometry
	const narrow = $derived(W < 640);
	const labelW = 20;
	const cell = $derived(narrow ? Math.min(30, (W - 40) / T) : Math.min(30, (W - 3 * (labelW + 6) - 2 * 44) / (3 * T)));
	const gw = $derived(cell * T);
	const gap = $derived(narrow ? 0 : (W - 3 * (gw + labelW + 6)) / 2);
	const px = $derived([0, 1, 2].map((k) => (narrow ? labelW + 10 : labelW + 6 + k * (gw + labelW + 6 + gap))));
	const top = 66;
	const py = $derived([0, 1, 2].map((k) => (narrow ? top + k * (gw + 84) : top)));
	const H = $derived((narrow ? py[2] : top) + gw + 44);

	const rowSum = (M, i) => M[i].reduce((s, v) => s + v, 0);
	const readout = $derived.by(() => {
		if (active < 0) return null;
		const i = active;
		const future = Abar[i].reduce((s, v, j) => s + (j > i ? Math.abs(v) : 0), 0);
		return {
			who: `Reader ‘${showChar(chars[i])}’, row ${i + 1}, head ${head + 1}.`,
			a: `Σⱼ Aᵢⱼ = ${fmt(rowSum(A, i), 6)}.`,
			f: i < T - 1 ? `Ā reaches ${T - 1 - i} future source${T - 2 - i ? 's' : ''} with total size ${fmt(future, 5)}; S̄ keeps none of it.` : 'This reader has no future.',
			s: `Σⱼ S̄ᵢⱼ = ${rowSum(Sbar, i).toExponential(1).replace(/-/g, '−')}: zero, up to rounding.`
		};
	});
	const GLYPH_W = { A: 9.6, S: 7.4, O: 9.4 };

	// print: four static phases
	const printPhases = $derived([
		{ M: A, vmax: 1, ink: 'value', signed: false, upper: 0, sym: 'A', bar: false, note: 'the weights' },
		{ M: Abar, vmax: aMax, ink: 'covector', signed: true, upper: 1, sym: 'A', bar: true, note: 'the whole square' },
		{ M: centred, vmax: cMax, ink: 'covector', signed: true, upper: 1, sym: 'A', bar: true, note: 'minus its row mean' },
		{ M: Sbar, vmax: sMax, ink: 'covector', signed: true, upper: 0, sym: 'S', bar: true, note: 'times A' }
	]);
	const pc = $derived(Math.min(22, (W - 4 * 26) / (4 * T + 3)));
	const pgap = $derived((W - 4 * (pc * T + 22)) / 3);
</script>

{#snippet sym(x, y, s, bar, size = 14)}
	<text class="m" {x} {y} style:font-size="{size}px">{s}</text>
	{#if bar}
		{@const k = size / 14}
		<line class="fx-spb__adj" x1={x + 2.4 * k} x2={x + (GLYPH_W[s] ?? 8) * k + 1.4 * k} y1={y - 13 * k} y2={y - 13 * k} />
	{/if}
{/snippet}

{#snippet axes(x, y, c, focus = -1, cols = true)}
	{#each chars as ch, i (i)}
		<text class="tok fx-spb__ax" class:is-on={i === focus} x={x - 7} y={y + i * c + c / 2 + 5} text-anchor="end">{showChar(ch)}</text>
		{#if cols}<text class="tok fx-spb__ax" x={x + i * c + c / 2} y={y + T * c + 16} text-anchor="middle">{showChar(ch)}</text>{/if}
	{/each}
{/snippet}

{#snippet diag(x, y, c)}
	<line class="fx-spb__diag" x1={x} y1={y} x2={x + T * c} y2={y + T * c} />
{/snippet}

<div class="fx fx-spb" use:measure={(v) => (width = v)}>
	{#if interactive}
		<svg viewBox="0 0 {W} {H}" width={W} height={H} role="img"
			aria-label="Softmax pullback, head {head + 1}, window {win + 1}: the weights A, the adjoint Ā on the whole square, and the score adjoint S̄, which is zero above the diagonal and has zero row sums.">
			<!-- head names -->
			{#each [0, 1] as h (h)}
				<text class="lbl fx-spb__head" class:is-on={h === head} x={W - (h ? 0 : 52)} y="14" text-anchor="end" role="button" tabindex="0" aria-pressed={h === head}
					onclick={() => (head = h)} onkeydown={(e) => (e.key === 'Enter' || e.key === ' ') && (head = h)}>head {h + 1}</text>
			{/each}
			<!-- A -->
			{@render sym(px[0], py[0] - 30, 'A', false)}
			<text class="lbl" x={px[0] + 14} y={py[0] - 30}>the weights</text>
			<text class="lbl lbl--micro" x={px[0]} y={py[0] - 14}>value; nothing above the diagonal</text>
			{@render axes(px[0], py[0], cell, active)}
			<TriangleDots M={A} x={px[0]} y={py[0]} {cell} vmax={1} row={active} onrow={onrow} />
			<!-- Ā -->
			{@render sym(px[1], py[1] - 30, 'A', true)}
			<text class="lbl" x={px[1] + 15} y={py[1] - 30}>=</text>
			{@render sym(px[1] + 27, py[1] - 30, 'O', true)}
			<text class="sup" x={px[1] + 38} y={py[1] - 36}>T</text>
			<text class="m" x={px[1] + 45} y={py[1] - 30}>V</text>
			<text class="lbl" x={px[1] + 58} y={py[1] - 30}>, the adjoint of the weights</text>
			<text class="lbl lbl--micro" x={px[1]} y={py[1] - 14}>covector; the future included · largest dot {fmt(aMax, 4)}</text>
			{@render axes(px[1], py[1], cell, active)}
			{@render diag(px[1], py[1], cell)}
			<TriangleDots M={Abar} x={px[1]} y={py[1]} {cell} vmax={aMax} signed upper={1} ink="covector" row={active} onrow={onrow} />
			<!-- S̄, computed on the clock -->
			{@render sym(px[2], py[2] - 30, t > 0.95 ? 'S' : 'A', true)}
			<text class="lbl" x={px[2] + 14} y={py[2] - 30}>{t > 0.95 ? '= A ⊙ (the centred rows)' : t > 0.45 ? 'centred, times A' : t > 0.08 ? 'minus its A-weighted row mean' : ''}</text>
			<text class="lbl lbl--micro" x={px[2]} y={py[2] - 14}>covector · largest dot {fmt(live.vmax, 4)}</text>
			{@render axes(px[2], py[2], cell, active)}
			{@render diag(px[2], py[2], cell)}
			<TriangleDots M={live.M} x={px[2]} y={py[2]} {cell} vmax={live.vmax} signed upper={live.upper} ink="covector" row={active} onrow={onrow} />
			{#if active >= 0 && t > 0.95}
				<text class="lbl lbl--micro fx-spb__sum" x={px[2] + (active + 1) * cell + 6} y={py[2] + active * cell + cell / 2 + 4}>Σ = 0</text>
				<text class="lbl lbl--micro fx-spb__sum" x={px[0] + (active + 1) * cell + 6} y={py[0] + active * cell + cell / 2 + 4}>Σ = 1</text>
			{/if}
		</svg>
		<Scrubber bind:t phases={PHASES} ink="covector" label="the softmax pullback, from Ā to S̄" duration={3200} />
		<p class="readout" aria-live="polite">
			{#if readout}
				<span class="readout__who">{readout.who}</span>
				<span class="readout__part">{readout.a}</span>
				<span class="readout__part">{readout.f}</span>
				<span class="readout__part">{readout.s}</span>
			{:else}
				<span class="readout__hint">Point at a row to follow one reader through the three panels; click to keep it. Press play, or drag along the line, to run the pullback from Ā to S̄. Click a head’s name to change head.</span>
			{/if}
		</p>
	{:else}
		{@const PH = pc * T + 62}
		<svg viewBox="0 0 {W} {PH}" width={W} height={PH} role="img" aria-label="The softmax pullback of head {head + 1} in four stages: A, Ā, Ā minus its row mean, and S̄.">
			{#each printPhases as ph, k (k)}
				{@const x = 20 + k * (pc * T + 22 + pgap)}
				{@const y = 34}
				{@render sym(x, 14, ph.sym, ph.bar, 13)}
				<text class="lbl" x={x + 13} y="14">{k === 3 ? '= A ⊙ (centred)' : ph.note}</text>
				<text class="lbl lbl--micro" x={x} y={y + pc * T + 18}>{k === 0 ? 'value · area ∝ weight' : `largest dot ${fmt(ph.vmax, 4)}`}</text>
				{#each chars as ch, i (i)}
					<text class="tok fx-spb__ax fx-spb__ax--s" x={x - 5} y={y + i * pc + pc / 2 + 4} text-anchor="end">{showChar(ch)}</text>
				{/each}
				{#if k > 0}{@render diag(x, y, pc)}{/if}
				<TriangleDots M={ph.M} {x} {y} cell={pc} vmax={ph.vmax} signed={ph.signed} upper={ph.upper} ink={ph.ink} />
			{/each}
		</svg>
	{/if}
</div>

<style>
	.fx-spb :global(.m) { font-family: var(--serif); font-style: italic; font-size: 14px; fill: var(--ink); }
	.fx-spb :global(.sup) { font-size: 9px; font-family: var(--serif); fill: var(--ink); }
	.fx-spb__adj { stroke: var(--ink); stroke-width: 0.9px; }
	.fx-spb__ax { font-size: 14px; fill: var(--ink-2); }
	.fx-spb__ax--s { font-size: 11px; }
	.fx-spb__ax.is-on { fill: var(--ink); font-weight: 650; }
	.fx-spb__diag { stroke: var(--rule); stroke-width: var(--hair); stroke-dasharray: 2 3; }
	.fx-spb__sum { fill: var(--ink-2); }
	.fx-spb__head { cursor: pointer; fill: var(--ink-3); font-weight: 500; }
	.fx-spb__head.is-on { fill: var(--ink); font-weight: 650; text-decoration: underline; text-underline-offset: 3px; }
	/* covector dots: hatched when positive, open when negative (DESIGN §1.5) */
	.fx-spb :global(.tri--covector .tri__dot) { fill: url(#hatch-covector); stroke: var(--covector); stroke-width: 0.75px; }
	.fx-spb :global(.tri--covector .tri__dot--neg) { fill: none; stroke: var(--covector); }
	.fx-spb :global(.tri--covector .tri__dot) { transition: none; }
</style>

<script module>
	export const meta = { size: 'wide', title: 'Why subtract the maximum' };
</script>

<script>
	// Figure: the seven exponentials e^{βz_a} of one logit column, computed in Float64
	// exactly as written (top strip) and after subtracting the maximum (bottom strip), on a
	// logarithmic axis spanning everything Float64 can represent. Beyond either end the
	// arithmetic gives up: values become ∞ or 0 and are drawn in the bins there. Below, the
	// probabilities each recipe returns. Drag β (= 1/τ) along its track.
	// Print: β = 1 and β = 200 as small multiples.
	import { getContext } from 'svelte';
	import { measure } from '../primitives/measure.js';
	import { dodge } from '../primitives/EnergyLadder.svelte';
	import { fmt, fmtProb, showChar, clamp, layoutWidth } from '../primitives/scales.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const T = $derived(specimen.config.T);
	const V = $derived(specimen.config.V);
	const interactive = $derived(mode !== 'print');
	let width = $state(868);
	// svelte-ignore state_referenced_locally
	let beta = $state(attrs.beta ? +attrs.beta : 1);

	const start = $derived(clamp((+attrs.window || 3) - 1, 0, specimen.windows.length - 1));
	const pos = $derived(clamp((+attrs.position || 1) - 1, 0, T - 1));
	const z = $derived(specimen.windowTrace(start).Z[pos]);
	const ctx = $derived(specimen.windowTrace(start).chars.slice(0, pos + 1).map(showChar).join(''));
	const m = $derived(Math.max(...z));

	// the limits of Float64, from the arithmetic itself
	const LG_MAX = Math.log10(Number.MAX_VALUE); // ≈ 308.25
	const LG_MIN = Math.log10(Number.MIN_VALUE); // ≈ −323.3 (smallest subnormal)
	const X_OVER = Math.log(Number.MAX_VALUE); // e^x = ∞ beyond this
	const betaOver = $derived(X_OVER / Math.max(...z.map(Math.abs).filter((_, a) => z[a] > 0), 1e-9));

	function compute(b) {
		const naive = z.map((v) => Math.exp(b * v));
		const S = naive.reduce((s, v) => s + v, 0);
		const stable = z.map((v) => Math.exp(b * (v - m)));
		const S2 = stable.reduce((s, v) => s + v, 0);
		return {
			naive, S, pNaive: naive.map((v) => v / S),
			stable, S2, pStable: stable.map((v) => v / S2),
			lseNaive: Math.log(S) / b, lseStable: m + Math.log(S2) / b
		};
	}

	// geometry
	const W = $derived(layoutWidth(width, mode, 340));
	const narrow = $derived(W < 640);
	const L = $derived(narrow ? 44 : 118);
	const binW = 30;
	const ax0 = $derived(L + binW + 14);
	const ax1 = $derived(W - binW - 30);
	const xLg = (lg) => ax0 + ((lg - LG_MIN) / (LG_MAX - LG_MIN)) * (ax1 - ax0);
	const xZero = $derived(L + binW / 2);
	const xInf = $derived(W - binW / 2 - 12);
	const xOf = (v) => (v === Infinity ? xInf : v === 0 ? xZero : Number.isNaN(v) ? xInf : xLg(Math.log10(v)));
	const decades = $derived(narrow ? [-300, 0, 300] : [-300, -200, -100, 0, 100, 200, 300]);

	// β track (log, 1 … 1000)
	const B_MIN = 1, B_MAX = 1000;
	const bx0 = $derived(ax0), bx1 = $derived(ax1);
	const xB = (b) => bx0 + (Math.log10(b) / 3) * (bx1 - bx0);
	const bOf = (x) => 10 ** (clamp((x - bx0) / (bx1 - bx0), 0, 1) * 3);
	let svgEl = $state();
	let dragging = $state(false);
	function px(e) { const pt = svgEl.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY; return pt.matrixTransform(svgEl.getScreenCTM().inverse()).x; }
	function bDown(e) { dragging = true; e.currentTarget.setPointerCapture(e.pointerId); bMove(e); }
	function bMove(e) {
		if (!dragging) return;
		let b = bOf(px(e));
		if (Math.abs(Math.log(b)) < 0.05) b = 1;
		beta = clamp(b, B_MIN, B_MAX);
	}
	function bKey(e) {
		const f = e.shiftKey ? 2 : 1.15;
		if (e.key === 'ArrowRight' || e.key === 'ArrowUp') { beta = clamp(beta * f, B_MIN, B_MAX); e.preventDefault(); }
		if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') { beta = clamp(beta / f, B_MIN, B_MAX); e.preventDefault(); }
		if (e.key === 'Home' || e.key === 'Enter') { beta = 1; e.preventDefault(); }
	}

	const states = $derived(interactive ? [{ b: beta }] : [{ b: 1 }, { b: 200 }]);
	const stripH = 74;
	const blockH = $derived(2 * stripH + 150);
	const top = $derived(interactive ? 84 : 36);
	const H = $derived(top + states.length * blockH + 6);
	const fmtP = (v) => (Number.isNaN(v) ? 'NaN' : fmtProb(v, 2));
	const sup = (n) => String(n).replace('-', '−');
</script>

{#snippet strip(vals, S, y, title, shifted)}
	{@const xs = vals.map(xOf)}
	{@const lx = dodge(xs, 15)}
	<text class="lbl lbl--head" x={narrow ? 4 : 0} y={y - 4}>{title}</text>
	<text class="lbl lbl--micro" x={narrow ? 4 : 0} y={y + 22}><tspan class="m">e</tspan><tspan class="m" font-size="10" dy="-6">β{shifted ? '(z − m)' : 'z'}</tspan></text>
	<line class="axis" x1={ax0} x2={ax1} y1={y + 26} y2={y + 26} />
	{#each vals as v, a (a)}
		<line class="leader" x1={xs[a]} x2={lx[a]} y1={y + 22} y2={y + 4} />
		<circle class={v === Infinity || v === 0 ? 'dot--open' : 'dot--value'} cx={xs[a]} cy={y + 26} r="3.4" />
		<text class="tok fx-lse__ch" x={lx[a]} y={y} text-anchor="middle">{showChar(specimen.vocab[a])}</text>
	{/each}
	<!-- the sum -->
	<g class="fx-lse__sum">
		<line class="fx-lse__sumtick" x1={xOf(S)} x2={xOf(S)} y1={y + 34} y2={y + 44} />
		<text class="lbl lbl--micro" x={xOf(S)} y={y + 56} text-anchor="middle">Σ{S === Infinity ? ' = ∞' : ''}</text>
	</g>
{/snippet}

<div class="fx fx-lse" use:measure={(w) => (width = w)}>
	<svg bind:this={svgEl} viewBox="0 0 {W} {H}" width={W} height={H} role={interactive ? 'group' : 'img'} aria-label="Exponentials of the logits after ‘{ctx}’ at β = {fmt(beta, 1)}, naive and shifted by the maximum">
		{#if interactive}
			<!-- β: drag along its track -->
			<text class="lbl lbl--head" x={narrow ? 4 : 0} y="34">inverse</text>
			<text class="lbl lbl--head" x={narrow ? 4 : 0} y="48">temperature</text>
			<line class="axis" x1={bx0} x2={bx1} y1="30" y2="30" />
			{#each [1, 10, 100, 1000] as b (b)}
				<line class="axis" x1={xB(b)} x2={xB(b)} y1="26" y2="34" />
				<text class="lbl lbl--micro" x={xB(b)} y="48" text-anchor="middle">{b}</text>
			{/each}
			{#if betaOver > 1 && betaOver < 1000}
				<line class="fx-lse__over" x1={xB(betaOver)} x2={xB(betaOver)} y1="22" y2="38" />
				<text class="lbl lbl--micro" x={xB(betaOver) + 5} y="20">e<tspan dy="-4" font-size="8">βz</tspan><tspan dy="4"> first overflows</tspan></text>
			{/if}
			<g transform="translate({xB(beta)} 30)">
				<circle class="handle" class:handle--active={dragging} r="6.5" />
				<text class="lbl fx-lse__blabel" y="-12" text-anchor={beta < 1.6 ? 'start' : 'middle'} x={beta < 1.6 ? -6 : 0}><tspan class="m">β</tspan> = {fmt(beta, beta < 10 ? 2 : 0)}</text>
			</g>
			<rect class="handle-hit" x={bx0 - 10} y="14" width={bx1 - bx0 + 20} height="32" role="slider" tabindex="0" aria-label="inverse temperature β" aria-valuemin={B_MIN} aria-valuemax={B_MAX} aria-valuenow={beta} aria-valuetext="β = {fmt(beta, 2)}" onpointerdown={bDown} onpointermove={bMove} onpointerup={() => (dragging = false)} onkeydown={bKey} ondblclick={() => (beta = 1)} />
		{/if}

		{#each states as st, k (k)}
			{@const c = compute(st.b)}
			{@const y0 = top + k * blockH}
			<!-- the representable range, and the two bins where Float64 gives up -->
			<g class="fx-lse__frame">
				<rect class="fx-lse__bin" x={xZero - binW / 2} y={y0 + 4} width={binW} height={2 * stripH + 16} />
				<rect class="fx-lse__bin" x={xInf - binW / 2} y={y0 + 4} width={binW} height={2 * stripH + 16} />
				<text class="lbl fx-lse__binl" x={xZero} y={y0 - 4} text-anchor="middle">0</text>
				<text class="lbl fx-lse__binl" x={xInf} y={y0 - 4} text-anchor="middle">∞</text>
				<line class="fx-lse__one" x1={xLg(0)} x2={xLg(0)} y1={y0 + 4} y2={y0 + 2 * stripH + 20} />
				{#each decades as d (d)}
					<text class="lbl lbl--micro" x={xLg(d)} y={y0 + 2 * stripH + 34} text-anchor="middle">{#if d === 0}1{:else}10<tspan dy="-5" font-size="8">{sup(d)}</tspan>{/if}</text>
				{/each}
				{#if !narrow}<text class="lbl lbl--micro" x={ax1} y={y0 + 2 * stripH + 48} text-anchor="end">largest Float64 ≈ {fmt(10 ** (LG_MAX - Math.floor(LG_MAX)), 1)} × 10<tspan dy="-5" font-size="8">{Math.floor(LG_MAX)}</tspan></text>{/if}
				{#if !narrow}<text class="lbl lbl--micro" x={ax0} y={y0 + 2 * stripH + 48}>smallest ≈ {fmt(10 ** (LG_MIN - Math.floor(LG_MIN)), 1)} × 10<tspan dy="-5" font-size="8">{sup(Math.floor(LG_MIN))}</tspan></text>{/if}
				{#if !interactive}<text class="lbl lbl--head" x={xLg(0) + 8} y={y0 + 14}><tspan class="m">β</tspan> = {st.b}</text>{/if}
			</g>
			{@render strip(c.naive, c.S, y0 + 34, 'as written', false)}
			{@render strip(c.stable, c.S2, y0 + 34 + stripH, 'max subtracted', true)}

			<!-- what each recipe returns -->
			{@const ty = y0 + 2 * stripH + 74}
			{#if !narrow}<text class="lbl lbl--micro" x={0} y={ty + 16}>probabilities</text>{/if}
			{#each specimen.vocab as ch, a (a)}
				{@const xx = narrow ? 30 + a * ((W - 40) / V) : ax0 + 20 + a * Math.min(64, (ax1 - ax0 - 200) / V)}
				<text class="tok fx-lse__pch" x={xx} y={ty} text-anchor="middle">{showChar(ch)}</text>
				<text class="lbl lbl--micro fx-lse__p" class:is-nan={Number.isNaN(c.pNaive[a])} x={xx} y={ty + 16} text-anchor="middle">{fmtP(c.pNaive[a])}</text>
				<text class="lbl lbl--micro fx-lse__p" class:lbl--strong={c.pStable[a] >= 0.1} x={xx} y={ty + 32} text-anchor="middle">{fmtP(c.pStable[a])}</text>
			{/each}
			{@const xr = ax0 + 20 + V * Math.min(64, (ax1 - ax0 - 200) / V)}
			{#if narrow}<text class="lbl lbl--micro" x="4" y={ty + 50}>row 1: as written; row 2: max subtracted</text>{:else}
			<text class="lbl lbl--micro" x={xr} y={ty + 16}>← as written{c.pNaive.some(Number.isNaN) ? ': ∞/∞ or 0/0 is not a number' : ''}</text>
			<text class="lbl lbl--micro" x={xr} y={ty + 32}>← max subtracted: lse = {fmt(c.lseStable, 3)}</text>{/if}
		{/each}
	</svg>
	{#if interactive}
		{@const c = compute(beta)}
		<p class="readout" aria-live="polite">
			<span class="readout__who">After ‘{ctx}’, at β = {fmt(beta, 2)}.</span>
			<span class="readout__part">
				{#if c.S === Infinity}As written, {c.naive.filter((v) => v === Infinity).length} of the exponentials are ∞, so the sum is ∞ and the probabilities are not numbers.{:else if c.pNaive.some(Number.isNaN)}As written, every exponential has underflowed to 0 and the probabilities are 0/0.{:else}Both recipes agree: every exponential is still a Float64.{/if}
				With the maximum subtracted the largest term is exactly 1 and the sum is {fmt(c.S2, 3)}, between 1 and {V}.
			</span>
			<span class="readout__hint">Drag β to the right (double-click: back to 1). Dividing logits by a small temperature is the same as multiplying them by a large β.</span>
		</p>
	{/if}
</div>

<style>
	.fx-lse :global(.m) { font-family: var(--serif); font-style: italic; fill: var(--ink); font-size: 14px; }
	.fx-lse__ch { font-size: 14px; }
	.fx-lse__pch { font-size: 14px; fill: var(--ink-2); }
	.fx-lse__p { fill: var(--ink-2); }
	.fx-lse__p.is-nan { fill: var(--ink); font-weight: 600; }
	.fx-lse__bin { fill: url(#hatch-ink); stroke: var(--rule); stroke-width: var(--hair); }
	.fx-lse__binl { fill: var(--ink); font-size: 13px; }
	.fx-lse__one { stroke: var(--ink-3); stroke-width: var(--hair); stroke-dasharray: 2 3; }
	.fx-lse__sumtick { stroke: var(--ink); stroke-width: 1.4px; }
	.fx-lse__over { stroke: var(--ink-2); stroke-width: 1px; }
	.fx-lse__blabel { fill: var(--ink); }
</style>

<script module>
	export const meta = { size: 'wide', title: 'The pullback of an affine map' };
	export { default as Key } from './fig-affine-pullback.key.svelte';
</script>

<script>
	// Figure: the pullback of the readout z = W_out x + b_out for one token column, at θ₀.
	// The adjoint z̄ = (p − e_y)/N arrives from the loss (shared rows, centre). Backward
	// runs up: on the left, x̄ = W_outᵀ z̄ (each column of W weighted by z̄ and summed, the
	// result drawn above the matrix); on the right, the column's share z̄ xᵀ of the weight
	// gradient, the outer product of the z̄ column and the x row. Far right: the sum of all
	// N such shares, which is W̄_out itself (read from the engine's θ̄).
	import { getContext } from 'svelte';
	import { measure } from '../primitives/measure.js';
	import MatrixGlyph from '../primitives/MatrixGlyph.svelte';
	import { fmt, fmtProb, showChar, clamp, maxAbs, layoutWidth } from '../primitives/scales.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const V = $derived(specimen.config.V);
	const D = $derived(specimen.config.D);
	const T = $derived(specimen.config.T);
	const gt = $derived(specimen.lab.gradTrace(specimen.theta0));
	const B = $derived(gt.windows.length);
	const N = $derived(B * T);

	// svelte-ignore state_referenced_locally
	let win = $state(clamp((+attrs.window || 1) - 1, 0, 9));
	// svelte-ignore state_referenced_locally
	let col = $state(clamp((+attrs.column || 8) - 1, 0, 7));
	let cell = $state(null); // {a, r} pointed-at square of the share
	let width = $state(868);
	const interactive = $derived(mode !== 'print');

	const w = $derived(gt.windows[win]);
	const chars = $derived(w.x.map((a) => specimen.vocab[a]));
	const target = $derived(w.y[col]); // 0-based id of the next character
	const x = $derived(w.val.Xf[col]);
	const p = $derived(w.val.probs[col]);
	const zbar = $derived(w.bar.Z[col]);
	const Wp = $derived(gt.params.find((q) => q.name === 'Wout'));
	const Wv = $derived(Array.from({ length: V }, (_, a) => Array.from({ length: D }, (_, r) => Wp.values[a + V * r])));
	const Wbar = $derived(Array.from({ length: V }, (_, a) => Array.from({ length: D }, (_, r) => Wp.grad[a + V * r])));
	const share = $derived(zbar.map((za) => x.map((xr) => za * xr)));
	const xbar = $derived(Array.from({ length: D }, (_, r) => zbar.reduce((s, za, a) => s + Wv[a][r] * za, 0)));

	// scales: shared across every column of every window (so clicking compares honestly)
	const zMax = $derived(maxAbs(gt.windows.map((u) => u.bar.Z)));
	const xMax = $derived(maxAbs(gt.windows.map((u) => u.val.Xf)));
	const xbMax = $derived(maxAbs(gt.windows.map((u) => u.bar.Xf)));
	const wMax = $derived(maxAbs(Wp.values));
	// one scale for the shares of every column (so columns compare honestly); the sum of
	// all N shares adds up coherently and is far larger, so it gets its own scale
	const sMax = $derived(Math.max(...gt.windows.flatMap((u) => u.bar.Z.map((z, t) => maxAbs(z) * maxAbs(u.val.Xf[t])))));
	const gMax = $derived(maxAbs(Wp.grad));

	// geometry
	const W = $derived(layoutWidth(width, mode, 340));
	const narrow = $derived(W < 640);
	const c = $derived(narrow ? Math.min(24, (W - 70) / D) : Math.min(25, (W - 330) / (3 * D)));
	const mw = $derived(c * D), mh = $derived(c * V);
	const zw = 92; // the z̄ column: bars about a zero line
	const lab = 34; // character labels
	const gap = $derived(narrow ? 0 : Math.max(26, (W - (3 * mw + zw + lab + 20)) / 3));
	// wide: [W_out]  [labels z̄]  [share]  [total]
	const xW = $derived(narrow ? lab : 8);
	const xZ = $derived(narrow ? 0 : xW + mw + gap * 0.55);
	const xS = $derived(narrow ? lab + zw : xZ + lab + zw + gap * 0.45);
	const xT = $derived(narrow ? lab + zw : xS + mw + gap);
	const top = 92; // window strip above
	const lolH = 34; // lollipop amplitude for x and x̄ rows
	const yM = $derived(top + 30 + 2 * lolH + 30); // matrices' top
	// narrow: stacked panels
	const yW = $derived(narrow ? yM : yM);
	const yS = $derived(narrow ? yM + mh + 2 * lolH + 96 : yM);
	const yT = $derived(narrow ? yS + mh + 84 : yM);
	const H = $derived((narrow ? yT : yM) + mh + 104);

	const colX = (x0, r) => x0 + r * c + c / 2;
	const rowY = (y0, a) => y0 + a * c + c / 2;
	const lol = (v, max) => (v / max) * lolH;

	// corpus strip and window bracket
	const corpusChars = $derived(Array.from(specimen.corpus));
	const cs = $derived(Math.min(15, (W - 40) / corpusChars.length));
	const cx0 = $derived(W / 2 - (cs * corpusChars.length) / 2);
	const wx = $derived(Math.min(46, (W - 40) / T));
	const wx0 = $derived(W / 2 - (wx * T) / 2);

	function key(e) {
		if (e.key === 'ArrowRight') { col = Math.min(T - 1, col + 1); e.preventDefault(); }
		if (e.key === 'ArrowLeft') { col = Math.max(0, col - 1); e.preventDefault(); }
		if (e.key === 'ArrowUp') { win = Math.max(0, win - 1); e.preventDefault(); }
		if (e.key === 'ArrowDown') { win = Math.min(B - 1, win + 1); e.preventDefault(); }
	}

	// advance widths of the italic symbols at 14 px (Literata italic), for drawn bars
	const GLYPH_W = { x: 6.6, z: 6.0, W: 12.6, p: 7.2, e: 6.0 };

	const norm = (M) => Math.sqrt(M.flat().reduce((s, v) => s + v * v, 0));
	const readout = $derived.by(() => {
		const ch = showChar(chars[col]), y = showChar(specimen.vocab[target]);
		const base = {
			who: `Column ${col + 1} of window ${win + 1}, ‘${ch}’, followed by ‘${y}’.`,
			z: `The model gives ‘${y}’ p = ${fmtProb(p[target], 3, true)}, so z̄ = (p − e_y)/${N} is ${fmt(zbar[target], 4)} there and positive elsewhere.`,
			x: `x̄ = W_outᵀ z̄ has length ${fmt(Math.hypot(...xbar), 4)}; this column’s share of W̄_out has norm ${fmt(norm(share), 4)}, the sum of all ${N} shares ${fmt(norm(Wbar), 4)}.`
		};
		if (cell) {
			const { a, r } = cell;
			base.cell = `Pointed at: W̄[‘${showChar(specimen.vocab[a])}’, ${r + 1}] = z̄ × x = ${fmt(zbar[a], 4)} × ${fmt(x[r], 3)} = ${fmt(share[a][r], 5)} from this column; ${fmt(Wbar[a][r], 4)} from all of them.`;
		}
		return base;
	});
</script>

{#snippet lollipops(vals, max, x0, y0, ink)}
	<!-- a row of lollipops above a matrix, one per column; up = positive -->
	<line class="axis" x1={x0} x2={x0 + mw} y1={y0} y2={y0} />
	{#each vals as v, r (r)}
		{@const y = y0 - lol(v, max)}
		{#if ink === 'covector'}
			<line class="fx-ap__stem--cov" x1={colX(x0, r)} x2={colX(x0, r)} y1={y0} y2={y} />
			<rect class="fx-ap__bar" class:is-neg={v < 0} x={colX(x0, r) - 4.2} y={y - 1.3} width="8.4" height="2.6" />
		{:else}
			<line class="stem" x1={colX(x0, r)} x2={colX(x0, r)} y1={y0} y2={y} />
			<circle class={v >= 0 ? 'dot--value' : 'dot--open'} cx={colX(x0, r)} cy={y} r="2.8" />
		{/if}
	{/each}
{/snippet}

{#snippet sym(x, y, s, o = {})}
	<!-- an italic symbol with an optional drawn bar (the adjoint), subscript and superscript -->
	<text class="m" {x} {y}>{s}{#if o.sub}<tspan class="sub" dy="3">{o.sub}</tspan>{/if}{#if o.sup}<tspan class="sup" dy={o.sub ? -10 : -7}>{o.sup}</tspan>{/if}</text>
	{#if o.bar}
		{@const wd = GLYPH_W[s] ?? 8}
		{@const h = s === s.toUpperCase() ? 13 : 10.2}
		<line class="fx-ap__adj" x1={x + 1.2 + (s === s.toUpperCase() ? 1.5 : 0.8)} x2={x + wd + (s === s.toUpperCase() ? 1.5 : 0.8)} y1={y - h} y2={y - h} />
	{/if}
{/snippet}

{#snippet hits(x0, y0)}
	{#if cell}
		<rect class="fx-ap__cell" x={x0 + cell.r * c + 0.5} y={y0 + cell.a * c + 0.5} width={c - 1} height={c - 1} />
	{/if}
	{#if interactive}
		{#each { length: V } as _, a (a)}
			{#each { length: D } as _, r (r)}
				<rect class="hit" x={x0 + r * c} y={y0 + a * c} width={c} height={c} role="button" aria-label="read this entry; details in the live readout" tabindex="0" onpointerenter={() => (cell = { a, r })} onfocus={() => (cell = { a, r })} onpointerleave={() => (cell = null)} onblur={() => (cell = null)} onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); (cell = { a, r }); } }} />
			{/each}
		{/each}
	{/if}
{/snippet}

<div class="fx fx-ap" use:measure={(v) => (width = v)}>
	<svg viewBox="0 0 {W} {H}" width={W} height={H} role={interactive ? 'group' : 'img'}
		aria-label="Pullback of the readout for column {col + 1} of window {win + 1}: z̄ from the loss, x̄ = W_outᵀ z̄, and the weight gradient share z̄ xᵀ. {readout.x}">
		<!-- the corpus, with the training window bracketed -->
		<g class="fx-ap__corpus">
			{#each corpusChars as ch, k (k)}
				<text class="tok fx-ap__cc" class:is-in={k >= win && k < win + T} x={cx0 + k * cs + cs / 2} y="14" text-anchor="middle">{showChar(ch)}</text>
				{#if interactive && k <= B - 1}
					<rect class="hit" x={cx0 + k * cs} y="0" width={cs} height="22" role="button" tabindex="0" aria-label="window {k + 1}" onclick={() => (win = k)} onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); (win = k); } }} />
				{/if}
			{/each}
			<path class="fx-ap__bracket" d="M {cx0 + win * cs + 1} 19 v 3 H {cx0 + (win + T) * cs - 1} v -3" />
			<text class="lbl lbl--micro" x={cx0 + (win + T / 2) * cs} y="34" text-anchor="middle">window {win + 1}</text>
		</g>
		<!-- the window: click a column (keyboard: a slider over the row) -->
		{#if interactive}
			<rect class="hit" x={wx0} y={top - 50} width={wx * T} height="38" role="slider" tabindex="0" aria-label="token column (←→), window (↑↓)"
				aria-valuemin="1" aria-valuemax={T} aria-valuenow={col + 1} aria-valuetext="column {col + 1} of window {win + 1}" onkeydown={key} />
		{/if}
		{#each chars as ch, t (t)}
			<text class="tok fx-ap__char" class:tok--reader={t === col} x={wx0 + t * wx + wx / 2} y={top - 26} text-anchor="middle">{showChar(ch)}</text>
			<text class="lbl lbl--micro" x={wx0 + t * wx + wx / 2 + 9} y={top - 22}>{t + 1}</text>
			{#if t === col}<line class="fx-ap__under" x1={wx0 + t * wx + 8} x2={wx0 + t * wx + wx - 8} y1={top - 17} y2={top - 17} />{/if}
			{#if interactive}
				<rect class="hit" x={wx0 + t * wx} y={top - 50} width={wx} height="38" role="button" tabindex="0" aria-label="column {t + 1}" onclick={() => (col = t)} onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); (col = t); } }} />
			{/if}
		{/each}
		<text class="lbl lbl--micro" x={narrow ? wx0 + T * wx : wx0 + T * wx + 6} y={narrow ? top - 2 : top - 26} text-anchor={narrow ? 'end' : 'start'}>next: <tspan class="tok fx-ap__next">{showChar(specimen.vocab[target])}</tspan></text>

		<!-- LEFT: x̄ = W_outᵀ z̄  (W in value ink; x̄ above it, covector) -->
		<g>
			<text class="lbl lbl--head" x={xW} y={yW - 2 * lolH - 30}>into the stream</text>
			{@render sym(xW, yW - 2 * lolH - 14, 'x', { bar: true })}
			<text class="lbl" x={xW + 11} y={yW - 2 * lolH - 14}>=</text>
			{@render sym(xW + 24, yW - 2 * lolH - 14, 'W', { sub: 'out', sup: 'T' })}
			{@render sym(xW + 64, yW - 2 * lolH - 14, 'z', { bar: true })}
			{@render lollipops(xbar, xbMax, xW, yW - lolH - 8, 'covector')}
			<MatrixGlyph rows={Wv} x={xW} y={yW} cell={c} vmax={wMax} highlightRow={target} />
			<text class="lbl lbl--micro" x={xW} y={yW + mh + 16}><tspan class="m m--s">W</tspan><tspan class="m m--s sub" dy="3">out</tspan><tspan dy="-3">, value ink</tspan></text>
		</g>

		<!-- CENTRE: z̄ arriving from the loss; rows shared with both matrices -->
		{#if !narrow}
			<g transform="translate({xZ} 0)">
				<text class="lbl lbl--head" x={lab + zw / 2} y={yM - 2 * lolH - 30} text-anchor="middle">from the loss</text>
				{@render sym(lab + zw / 2 - 40, yM - 2 * lolH - 14, 'z', { bar: true })}
				<text class="lbl" x={lab + zw / 2 - 29} y={yM - 2 * lolH - 14}>= (<tspan class="m">p</tspan> − <tspan class="m">e</tspan><tspan class="m sub" dy="3">y</tspan><tspan dy="-3">)/{N}</tspan></text>
			</g>
		{/if}
		{#each [narrow ? [xS - zw - lab, yS] : [xZ, yM]] as [zx, zy] (zx)}
			<g transform="translate({zx} {zy})">
				<line class="axis" x1={lab + zw / 2} x2={lab + zw / 2} y1={-4} y2={mh + 4} />
				{#each zbar as z, a (a)}
					{@const len = (Math.abs(z) / zMax) * (zw / 2 - 3)}
					<text class="tok fx-ap__row" class:is-y={a === target} class:is-pt={cell?.a === a} x={lab - 10} y={a * c + c / 2 + 5} text-anchor="end">{showChar(specimen.vocab[a])}</text>
					<rect class="bar--covector" class:fx-ap__neg={z < 0} x={z >= 0 ? lab + zw / 2 : lab + zw / 2 - len} y={a * c + c * 0.2} width={Math.max(0.6, len)} height={c * 0.6} />
				{/each}
				<text class="lbl lbl--micro" x={lab + zw / 2} y={mh + 16} text-anchor="middle">−  0  +</text>
			</g>
		{/each}

		<!-- RIGHT: this column's share z̄ xᵀ, with x (value) above -->
		<g>
			<text class="lbl lbl--head" x={xS} y={yS - 2 * lolH - 30}>into the weights</text>
			{@render sym(xS, yS - 2 * lolH - 14, 'z', { bar: true })}
			{@render sym(xS + 9, yS - 2 * lolH - 14, 'x', { sup: 'T' })}
			<text class="lbl" x={xS + 27} y={yS - 2 * lolH - 14}>: this column’s share</text>
			{@render lollipops(x, xMax, xS, yS - lolH - 8, 'value')}
			<text class="lbl lbl--micro" x={xS + mw + 6} y={yS - lolH - 5}><tspan class="m m--s">x</tspan></text>
			<MatrixGlyph rows={share} x={xS} y={yS} cell={c} vmax={sMax} ink="covector" />
			{@render hits(xS, yS)}
			<text class="lbl lbl--micro" x={xS} y={yS + mh + 16}>largest square, any column: {fmt(sMax, 3)}</text>
		</g>

		<!-- FAR RIGHT: the sum over all N columns = W̄_out -->
		<g>
			<text class="lbl lbl--head" x={xT} y={narrow ? yT - 26 : yT - 2 * lolH - 30}>all {N} columns</text>
			{@render sym(xT, narrow ? yT - 10 : yT - 2 * lolH - 14, 'W', { bar: true, sub: 'out' })}
			<text class="lbl" x={xT + 36} y={narrow ? yT - 10 : yT - 2 * lolH - 14}>= Σ<tspan class="m sub" dy="3">n</tspan></text>
			{@render sym(xT + 62, narrow ? yT - 10 : yT - 2 * lolH - 14, 'z', { bar: true, sub: 'n' })}
			{@render sym(xT + 79, narrow ? yT - 10 : yT - 2 * lolH - 14, 'x', { sub: 'n', sup: 'T' })}
			<MatrixGlyph rows={Wbar} x={xT} y={yT} cell={c} vmax={gMax} ink="covector" />
			{@render hits(xT, yT)}
			<text class="lbl lbl--micro" x={xT} y={yT + mh + 16}>largest square {fmt(gMax, 3)}</text>
		</g>
		<text class="lbl lbl--micro" x="8" y={H - 32}>full scale |entry|: W {wMax.toPrecision(3)}; x {xMax.toPrecision(3)}; z̄ {zMax.toPrecision(3)}</text>
		<text class="lbl lbl--micro" x="8" y={H - 16}>x̄ {xbMax.toPrecision(3)}; share {sMax.toPrecision(3)}; sum {gMax.toPrecision(3)}</text>
	</svg>
	{#if interactive}
		<p class="readout" aria-live="polite">
			<span class="readout__who">{readout.who}</span>
			<span class="readout__part">{readout.z}</span>
			<span class="readout__part">{cell ? readout.cell : readout.x}</span>
			<span class="readout__hint">Click a character to change the column, or a letter of the line to move the window (←→ and ↑↓ on the keyboard). Point at a square.</span>
		</p>
	{/if}
</div>

<style>
	.fx-ap :global(.m) { font-family: var(--serif); font-style: italic; font-size: 14px; fill: var(--ink); }
	.fx-ap :global(.m--s) { font-size: 12px; }
	.fx-ap :global(.sub) { font-size: 10px; }
	.fx-ap :global(.sup) { font-size: 9px; font-family: var(--serif); fill: var(--ink); }
	.fx-ap__adj { stroke: var(--ink); stroke-width: 0.9px; }
	.fx-ap__cc { font-size: 13px; fill: var(--ink-3); }
	.fx-ap__cc.is-in { fill: var(--ink); }
	.fx-ap__bracket { fill: none; stroke: var(--ink); stroke-width: 1px; }
	.fx-ap__char { font-size: 22px; }
	.fx-ap__next { font-size: 15px; fill: var(--ink); }
	.fx-ap__under { stroke: var(--ink); stroke-width: 1.2px; }
	.fx-ap__row { font-size: 15px; fill: var(--ink-2); }
	.fx-ap__row.is-y { fill: var(--ink); font-weight: 650; }
	.fx-ap__row.is-pt { text-decoration: underline; text-underline-offset: 3px; }
	.fx-ap__cell { fill: none; stroke: var(--ink); stroke-width: 1px; }
	.fx-ap__neg { fill: none; }
	.fx-ap__stem--cov { stroke: var(--covector); stroke-width: var(--line); }
	.fx-ap__bar { fill: var(--covector); stroke: var(--covector); stroke-width: 0.8px; }
	.fx-ap__bar.is-neg { fill: var(--paper); }
	.fx-ap :global(.matrix__cell--hl) { stroke-width: 1.4px; }
</style>

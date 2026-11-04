<script module>
	export const meta = { size: 'wide', title: 'The MLP of one column' };
	export { default as Key } from './fig-mlp.key.svelte';
</script>

<script>
	// Figure: the MLP of one column, top to bottom (forward = down):
	//   f̂ = LN₂(x_mid)                     the input, a spoke glyph
	//   u = W₁ f̂ + b₁ ∈ ℝᶠ                  F lollipops about zero; the rectifier keeps u_k > 0
	//   r_k W₂[:, k]                        under each firing unit: r_k times its column of W₂,
	//                                      what it adds (a unit that does not fire adds nothing)
	//   y = Σ_k r_k W₂[:, k] + b₂            the update added to the stream
	// Units are in index order k = 1…F, in bands of 16 (8 on a phone), so a unit keeps its
	// place when another column is chosen. Every mark is a number of the trained model.
	import { getContext } from 'svelte';
	import { measure } from '../primitives/measure.js';
	import SpokeGlyph from '../primitives/SpokeGlyph.svelte';
	import { fmt, showChar, clamp, maxAbs, spokeScale, layoutWidth } from '../primitives/scales.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const T = $derived(specimen.config.T);
	const D = $derived(specimen.config.D);
	const F = $derived(specimen.config.F);
	const B = $derived(specimen.lab.train.x.length);
	// svelte-ignore state_referenced_locally
	const win = clamp((+attrs.window || 1) - 1, 0, 9);
	// svelte-ignore state_referenced_locally
	let col = $state(clamp((+attrs.column || 4) - 1, 0, 7));
	let unit = $state(-1);
	let width = $state(868);
	const interactive = $derived(mode !== 'print');
	const W = $derived(layoutWidth(width, mode));
	const narrow = $derived(W < 640);

	const b = $derived(Math.min(win, B - 1));
	const w = $derived(specimen.lab.gradTrace(specimen.theta).windows[b]);
	const chars = $derived(Array.from(specimen.corpus.slice(b, b + T)));
	const prm = $derived(Object.fromEntries(specimen.params().map((p) => [p.name.split('.').at(-1), p.values])));
	/** column k of W₂ (D×F, column-major) */
	const W2col = (k) => prm.W2.slice(k * D, (k + 1) * D);
	const W1row = (k) => Array.from({ length: D }, (_, i) => prm.W1[k + F * i]);

	const fh = $derived(w.val.ln2[col]);
	const u = $derived(w.val.U[col]);
	const y = $derived(w.val.Y[col]);
	const parts = $derived(u.map((uk, k) => W2col(k).map((v) => Math.max(0, uk) * v)));
	const fire = $derived(u.filter((x) => x > 0).length);

	// scales shared over the whole window, so switching columns does not rescale
	const uMax = $derived(maxAbs(w.val.U));
	const partMax = $derived(maxAbs(w.val.U.map((uu) => uu.map((uk, k) => W2col(k).map((v) => Math.max(0, uk) * v)))));

	// ── geometry ──
	const per = $derived(narrow ? 8 : 16);
	const bands = $derived(Math.ceil(F / per));
	const L = $derived(narrow ? 8 : 86);
	const R = $derived(narrow ? 6 : 24);
	const slot = $derived((W - L - R) / per);
	const xs = (k) => L + slot * ((k % per) + 0.5);
	const rU = $derived(Math.min(34, slot * 0.6));
	const sc = $derived(spokeScale(rU, partMax));
	const uAmp = 40; // px for the largest |u_k| in the window
	const IN = 98; // the input glyph
	const TOP = 196; // first band's zero line
	const bandH = $derived(uAmp * 2 + rU * 2 + 46);
	const zeroY = (band) => TOP + band * bandH;
	const glyphY = (band) => zeroY(band) + uAmp + 16 + rU;
	const yMax = $derived(maxAbs(y, w.val.Y));
	const rY = $derived(Math.max(rU, yMax * sc.scale));
	const outY = $derived(glyphY(bands - 1) + rU + 46 + rY);
	const H = $derived(outY + rY + (narrow ? 40 : 52));
	const inR = 32;
	const inSc = $derived(spokeScale(inR, maxAbs(w.val.ln2)));

	const contributions = $derived(
		parts.map((p, k) => ({ k, n: Math.hypot(...p), r: Math.max(0, u[k]) })).filter((c) => c.r > 0).sort((a, b) => b.n - a.n)
	);
	const readout = $derived.by(() => {
		const ch = showChar(chars[col]);
		const top = contributions.slice(0, 3).map((c) => `unit ${c.k + 1} (r = ${fmt(c.r, 2)})`).join(', ');
		const one = unit >= 0
			? (() => {
				const k = unit, dot = W1row(k).reduce((s, v, i) => s + v * fh[i], 0);
				return u[k] > 0
					? `Unit ${k + 1}: u = w·f̂ + b₁ = ${fmt(dot, 2)} + ${fmt(prm.b1[k], 2)} = ${fmt(u[k], 2)} > 0, so it fires and adds ${fmt(u[k], 2)} × W₂[:, ${k + 1}], of length ${fmt(Math.hypot(...parts[k]), 2)}.`
					: `Unit ${k + 1}: u = w·f̂ + b₁ = ${fmt(dot, 2)} + ${fmt(prm.b1[k], 2)} = ${fmt(u[k], 2)} ≤ 0, so the rectifier sets r = 0 and its column of W₂ adds nothing.`;
			})()
			: null;
		return {
			who: `Column ‘${ch}’, position ${col + 1} of “${chars.join('')}”.`,
			fire: `${fire} of ${F} units fire; the largest shares come from ${top}. The update has length ‖y‖ = ${fmt(Math.hypot(...y), 2)}.`,
			one
		};
	});
	function key(e) {
		if (e.key === 'ArrowRight') { col = Math.min(T - 1, col + 1); e.preventDefault(); }
		if (e.key === 'ArrowLeft') { col = Math.max(0, col - 1); e.preventDefault(); }
		if (e.key === 'ArrowDown') { unit = Math.min(F - 1, unit + 1); e.preventDefault(); }
		if (e.key === 'ArrowUp') { unit = unit <= 0 ? -1 : unit - 1; e.preventDefault(); }
	}
	const charX = (j) => { const cw = Math.min(46, (W - L - R) / T); return W / 2 + cw * (j - (T - 1) / 2); };
</script>

<div class="fx fx-mlp" use:measure={(w) => (width = w)}>
	<!-- svelte-ignore a11y_no_noninteractive_element_interactions, a11y_no_noninteractive_tabindex -->
	<svg viewBox="0 0 {W} {H}" width={W} height={H} role="img" tabindex={interactive ? 0 : -1} onkeydown={key}
		aria-label="The MLP of column {col + 1}: {fire} of {F} hidden units fire; the update y has length {fmt(Math.hypot(...y), 2)}.">
		<!-- the window (click a character) and the input f̂ -->
		<g>
			{#each chars as ch, j (j)}
				<text class="tok fx-mlp__char" class:tok--reader={j === col} x={charX(j)} y={34} text-anchor="middle">{showChar(ch)}</text>
				{#if interactive}
					<rect class="hit" x={charX(j) - 14} y={12} width="28" height="30" role="button" tabindex="-1" aria-label="column {j + 1}" onclick={() => (col = j)} onkeydown={() => {}} />
				{/if}
			{/each}
			{#if !narrow}
				<text class="lbl" x={L - 14} y={IN - 3} text-anchor="end">input</text>
				<text class="lbl" x={L - 14} y={IN + 13} text-anchor="end"><tspan class="m">f̂</tspan></text>
			{/if}
			<SpokeGlyph values={fh} x={W / 2} y={IN} rmax={inR} scale={inSc.scale} weight={1.35} tip={2.6} />
			<text class="lbl" x={W / 2 + inR + 16} y={IN + 4}><tspan class="m">f̂</tspan> = LN₂(x){narrow ? '' : ', the normalised column'}</text>
		</g>

		{#each { length: bands } as _, band (band)}
			{@const z = zeroY(band)}
			{@const gy = glyphY(band)}
			{@const k0 = band * per}
			<!-- row labels -->
			{#if !narrow}
				<text class="fx-mlp__eq" x={L - 14} y={z - 3} text-anchor="end">u = W<tspan class="sub" dy="3">1</tspan><tspan dy="-3">f̂ + b</tspan><tspan class="sub" dy="3">1</tspan></text>
				<text class="lbl lbl--micro" x={L - 14} y={z + 11} text-anchor="end">units {k0 + 1}–{Math.min(F, k0 + per)}</text>
				<text class="fx-mlp__eq" x={L - 14} y={gy + 4} text-anchor="end">r<tspan class="sub" dy="3">k</tspan><tspan dy="-3"> W</tspan><tspan class="sub" dy="3">2</tspan><tspan dy="-3">[:, k]</tspan></text>
			{/if}
			{#if narrow}
				<text class="lbl lbl--micro" x={L} y={z - uAmp - 8}>u<tspan class="sub" dy="3">k</tspan><tspan dy="-3">, units {k0 + 1}–{Math.min(F, k0 + per)}</tspan></text>
			{/if}
			<line class="fx-mlp__zero" x1={L + 2} x2={W - R - 2} y1={z} y2={z} />
			<!-- lollipops: u_k; the rectifier keeps the positive ones -->
			{#each { length: Math.min(per, F - k0) } as _, i (i)}
				{@const k = k0 + i}
				{@const v = u[k]}
				{@const yy = z - (v / uMax) * uAmp}
				<g class="fx-mlp__unit" class:is-off={v <= 0} class:is-on={k === unit}>
					<line class="fx-mlp__stem" x1={xs(k)} x2={xs(k)} y1={z} y2={yy} />
					<circle class="fx-mlp__head" cx={xs(k)} cy={yy} r="3" />
					{#if k === unit}
						<SpokeGlyph values={W2col(k)} x={xs(k)} y={gy} rmax={rU} scale={sc.scale} muted compass={false} />
					{/if}
					<circle class="fx-mlp__hub" cx={xs(k)} cy={gy} r="1.1" />
					{#if v > 0}
						<SpokeGlyph values={parts[k]} x={xs(k)} y={gy} rmax={rU} scale={sc.scale} compass={false} tip={1.4 + 0.5 * Math.sqrt(v)} />
					{/if}
					{#if k === unit || (!narrow && slot > 34)}
						<text class="lbl lbl--micro fx-mlp__idx" x={xs(k)} y={gy + rU + 13} text-anchor="middle">{k + 1}</text>
					{/if}
				</g>
				{#if interactive}
					<rect class="hit" x={xs(k) - slot / 2} y={z - uAmp - 6} width={slot} height={gy + rU + 16 - (z - uAmp - 6)} role="button" tabindex="-1"
						aria-label="unit {k + 1}" onpointerenter={() => (unit = k)} onpointerleave={() => (unit = -1)} onclick={() => (unit = k)} onkeydown={() => {}} />
				{/if}
			{/each}
		{/each}

		<!-- the sum: y = Σ r_k W₂[:, k] + b₂ -->
		<g>
			{#if !narrow}
				<text class="lbl" x={L - 14} y={outY - 3} text-anchor="end">update</text>
				<text class="lbl" x={L - 14} y={outY + 13} text-anchor="end"><tspan class="m">y</tspan></text>
			{/if}
			<line class="rule" x1={L + 2} x2={W - R - 2} y1={glyphY(bands - 1) + rU + 24} y2={glyphY(bands - 1) + rU + 24} />
			<text class="lbl fx-mlp__sum" x={W / 2 - rY - 26} y={outY + 5} text-anchor="end">Σ<tspan class="sub" dy="3">k</tspan><tspan dy="-3"> r</tspan><tspan class="sub" dy="3">k</tspan><tspan dy="-3"> W₂[:, k] + b₂ =</tspan></text>
			<SpokeGlyph values={y} x={W / 2} y={outY} rmax={rY} scale={sc.scale} weight={1.4} tip={2.6} />
			{#if narrow}
				<text class="lbl" x={W / 2} y={outY + rY + 22} text-anchor="middle">‖y‖ = {fmt(Math.hypot(...y), 2)}, added to the stream</text>
			{:else}
				<text class="lbl" x={W / 2 + rY + 16} y={outY + 5}>‖y‖ = {fmt(Math.hypot(...y), 2)}, added to the stream</text>
			{/if}
		</g>
	</svg>
	{#if interactive}
		<p class="readout" aria-live="polite">
			<span class="readout__who">{readout.who}</span>
			<span class="readout__part">{readout.fire}</span>
			{#if readout.one}<span class="readout__part">{readout.one}</span>{/if}
			<span class="readout__hint">Click a character (or ← →) to change the column; point at a unit (or ↑ ↓) to read it.</span>
		</p>
	{/if}
</div>

<style>
	.fx-mlp :global(.m) { font-family: var(--serif); font-style: italic; font-size: 14px; fill: var(--ink); }
	.fx-mlp :global(.sub) { font-size: 9px; }
	.fx-mlp__char { font-size: 20px; }
	.fx-mlp__zero { stroke: var(--ink); stroke-width: var(--hair); }
	.fx-mlp__stem { stroke: var(--value); stroke-width: var(--line); }
	.fx-mlp__head { fill: var(--value); stroke: var(--value); stroke-width: 1px; }
	.fx-mlp__unit.is-off .fx-mlp__stem { stroke: var(--ink-3); opacity: 0.55; }
	.fx-mlp__unit.is-off .fx-mlp__head { fill: var(--paper); stroke: var(--ink-3); }
	.fx-mlp__unit.is-on .fx-mlp__stem { stroke-width: var(--heavy); }
	.fx-mlp__idx { fill: var(--ink-3); }
	.fx-mlp__hub { fill: var(--ink-3); }
	.fx-mlp__unit.is-on .fx-mlp__idx { fill: var(--ink); font-weight: 600; }
	.fx-mlp__sum { fill: var(--ink-2); font-size: 13px; }
	.fx-mlp__eq { font-family: var(--serif); font-style: italic; font-size: 13.5px; fill: var(--ink-2); }
</style>

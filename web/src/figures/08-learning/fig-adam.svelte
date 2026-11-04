<script module>
	export const meta = { size: 'wide', title: 'What Adam changes' };
	export { default as Key } from './fig-adam.key.svelte';

	// The book's run, replayed step by step and remembered (per specimen): for update t,
	// |gᵢ| (the gradient it used) and |Δθᵢ|/η_t (the step it took, in units of η).
	const runs = new WeakMap();
	function replay(specimen) {
		if (!runs.has(specimen)) runs.set(specimen, { run: specimen.lab.run(), g: [], d: [] });
		return runs.get(specimen);
	}
	function snapshot(specimen, t) {
		const r = replay(specimen);
		while (r.g.length < t) {
			const run = r.run;
			run.step();
			const k = run.t - 1;
			const scale = run.clipped[k] ? run.opts.clip / run.gradNorm[k] : 1;
			const eta = run.lr[k];
			r.g.push(Float64Array.from(run.grad, (x) => Math.abs(x * scale)));
			r.d.push(Float64Array.from(run.delta, (x) => Math.abs(x) / eta));
		}
		return { g: r.g[t - 1], d: r.d[t - 1], eta: r.run.lr[t - 1] };
	}
</script>

<script>
	// Figure: for every parameter θᵢ, grouped by array (rows, in the order of θ):
	//   a bar   at |gᵢ|       — covector ink: the gradient, what plain gradient descent
	//                           would move θᵢ by per unit of learning rate;
	//   a dot   at |Δθᵢ|/η_t  — value ink: what AdamW actually moved it by, same unit.
	// One shared log axis. Values below 10⁻⁹ (the key bias, whose true gradient is zero)
	// are set in a gutter on the left, labelled as rounding.
	// Clock: the update number t of the book's run (scrubbable). Print: t = 1 and t = last.
	import { getContext } from 'svelte';
	import { measure } from '../primitives/measure.js';
	import Scrubber from '../primitives/Scrubber.svelte';
	import { clamp, layoutWidth } from '../primitives/scales.js';
	import { paramName, pow10, sci, jitter } from './paramname.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const interactive = $derived(mode !== 'print');
	const STEPS = $derived(specimen.lab.steps);
	const warmup = $derived(specimen.lab.recipe.schedule?.warmup ?? 0);

	// svelte-ignore state_referenced_locally
	let clock = $state(attrs.t != null ? +attrs.t : 0);
	$effect(() => { if (attrs.t != null) clock = +attrs.t; });
	// the clock is logarithmic in the update number: the early steps are where Adam is most unlike GD
	const stepOf = (u) => Math.round(Math.exp(clamp(u, 0, 1) * Math.log(STEPS)));
	const clockOf = (t) => Math.log(t) / Math.log(STEPS);
	const step = $derived(stepOf(clock));
	const PHASES = $derived([
		{ at: 0, label: 'step 1' },
		...(narrow ? [] : [{ at: clockOf(5), label: '5' }]),
		...(warmup > 1 ? [{ at: clockOf(warmup), label: narrow ? `${warmup}` : `${warmup}, warmup ends` }] : []),
		...(narrow ? [] : [{ at: clockOf(100), label: '100' }]),
		{ at: 1, label: `${STEPS}` }
	]);

	let width = $state(868);
	const W = $derived(layoutWidth(width, mode, 340));
	const narrow = $derived(W < 600);

	const rows = $derived(specimen.layout.map((l) => ({ ...l, label: paramName(l.name) })));

	// axis: 10^-9 … 10^0.5, gutter for anything smaller
	const LO = -9, HI = 0.5, GLO = -22;
	const labelW = $derived(narrow ? 40 : 58);
	const countW = $derived(narrow ? 0 : 30);
	const gutterW = $derived(narrow ? 30 : 46);
	const x0 = $derived(labelW + gutterW + 10); // where 10^LO sits
	const x1 = $derived(W - countW - 8);
	const xOf = (v) => {
		if (!(v > 0)) return labelW + 4;
		const e = Math.log10(v);
		if (e < LO) return labelW + 4 + ((clamp(e, GLO, LO) - GLO) / (LO - GLO)) * (gutterW - 8);
		return x0 + ((Math.min(e, HI) - LO) / (HI - LO)) * (x1 - x0);
	};
	const ticks = [-8, -6, -4, -2, 0];
	/** x of a value in a print panel of width pw whose names column is L wide. */
	function pxIn(v, L, pw) {
		const g0 = L + 4, a0 = L + gutterW + 10, a1 = pw - 10;
		if (!(v > 0)) return g0;
		const e = Math.log10(v);
		if (e < LO) return g0 + ((clamp(e, GLO, LO) - GLO) / (LO - GLO)) * (gutterW - 8);
		return a0 + ((Math.min(e, HI) - LO) / (HI - LO)) * (a1 - a0);
	}

	const rowH = 23, band = 8;
	const top = $derived(narrow ? 64 : 46);
	const panelH = $derived(top + rows.length * rowH + 30);

	/** One row's marks as two paths (fast to redraw while scrubbing): bars |gᵢ|, dots |Δθᵢ|/η. */
	function rowMarks(snap, r, y0, x) {
		let bars = '', dots = '';
		for (let k = 0; k < r.size; k++) {
			const i = r.offset + k, j = jitter(i);
			bars += `M${x(snap.g[i]).toFixed(1)} ${(y0 + 1 + j * (band - 1)).toFixed(1)}v3.6`;
			const xd = x(snap.d[i]), yd = y0 + rowH - 2.5 - (1 - j) * (band - 1);
			dots += `M${(xd - 1.45).toFixed(2)} ${yd.toFixed(2)}a1.45 1.45 0 1 0 2.9 0a1.45 1.45 0 1 0 -2.9 0`;
		}
		return { bars, dots };
	}
	const marks = (snap) => rows.map((r) => rowMarks(snap, r, 0, xOf));

	function spread(a) {
		const s = Array.from(a).filter((v) => v >= 10 ** LO).sort((x, y) => x - y);
		const q = (f) => s[Math.floor(f * (s.length - 1))];
		return { lo: q(0.05), hi: q(0.95), dec: Math.log10(q(0.95) / q(0.05)) };
	}

	const snap = $derived(snapshot(specimen, step));
	const drawn = $derived(marks(snap));
	let hoverRow = $state(-1);

	const summary = $derived.by(() => {
		const g = spread(snap.g), d = spread(snap.d);
		return { g, d };
	});
	const rowInfo = $derived.by(() => {
		if (hoverRow < 0) return null;
		const r = rows[hoverRow];
		const g = Array.from(snap.g.subarray(r.offset, r.offset + r.size));
		const d = Array.from(snap.d.subarray(r.offset, r.offset + r.size));
		return { r, gmin: Math.min(...g), gmax: Math.max(...g), dmin: Math.min(...d), dmax: Math.max(...d) };
	});

	const printSteps = $derived([1, STEPS]);
	const printSnaps = $derived(mode === 'print' ? printSteps.map((t) => ({ t, snap: snapshot(specimen, t) })) : []);
</script>

{#snippet axis(y, label = true)}
	<g class="fx-adam__axis">
		{#each ticks as e (e)}
			<line class="rule" x1={xOf(10 ** e)} x2={xOf(10 ** e)} y1={y} y2={y + rows.length * rowH} />
			{#if label}<text class="lbl lbl--micro" x={xOf(10 ** e)} y={y - 7} text-anchor="middle">{pow10(e)}</text>{/if}
		{/each}
		<line class="fx-adam__one" x1={xOf(1)} x2={xOf(1)} y1={y - 2} y2={y + rows.length * rowH} />
		<!-- the gutter: below 10⁻⁹, rounding -->
		<line class="rule" x1={labelW + gutterW + 2} x2={labelW + gutterW + 2} y1={y} y2={y + rows.length * rowH} stroke-dasharray="2 3" />
		{#if label}<text class="lbl lbl--micro" x={labelW + gutterW / 2 + 1} y={y - 7} text-anchor="middle">&lt;{pow10(LO)}</text>{/if}
	</g>
{/snippet}

{#snippet strips(y, d, labels = true, hover = -1)}
	<g>
		{#each rows as r, k (r.name)}
			{@const yy = y + k * rowH}
			<g class="fx-adam__row" class:is-dim={hover >= 0 && hover !== k} transform="translate(0 {yy})">
				{#if labels}
					<text class="fx-adam__name" x={labelW - 8} y={rowH / 2 + 4} text-anchor="end"><tspan class="m">{r.label.sym}</tspan>{#if r.label.sub}<tspan class="m sub" dy="3">{r.label.sub}</tspan>{/if}</text>
					{#if countW}<text class="lbl lbl--micro" x={W - 4} y={rowH / 2 + 4} text-anchor="end">{r.size}</text>{/if}
				{/if}
				<path class="fx-adam__bars" d={d[k].bars} />
				<path class="fx-adam__dots" d={d[k].dots} />
			</g>
		{/each}
	</g>
{/snippet}

<div class="fx fx-adam" use:measure={(w) => (width = w)}>
	{#if interactive}
		<svg viewBox="0 0 {W} {panelH}" width={W} height={panelH} role={interactive ? 'group' : 'img'}
			aria-label="Step sizes at update {step} of {STEPS}: gradients span {summary.g.dec.toFixed(1)} decades, Adam's steps {summary.d.dec.toFixed(1)} decades">
			<text class="lbl lbl--head" x="0" y="14">step {step}</text>
			<text class="lbl" x={narrow ? 0 : x0} y={narrow ? 32 : 14}><tspan class="fx-adam__cov">bars: |<tspan class="m">g</tspan><tspan class="m sub" dy="3">i</tspan><tspan dy="-3">|, the gradient</tspan></tspan>{#if !narrow}{" · "}{/if}<tspan class="fx-adam__val" x={narrow ? 0 : null} dy={narrow ? 15 : 0}>dots: |Δ<tspan class="m">θ</tspan><tspan class="m sub" dy="3">i</tspan><tspan dy="-3">| / </tspan><tspan class="m">η</tspan>, Adam’s step</tspan></text>
			{#if !narrow}<text class="lbl lbl--micro" x={xOf(1) + 5} y={top - 18}>step = <tspan class="m">η</tspan></text>{/if}
			{#if countW}<text class="lbl lbl--micro" x={W - 4} y={top - 7} text-anchor="end">count</text>{/if}
			{@render axis(top)}
			{@render strips(top, drawn, true, hoverRow)}
			{#each rows as r, k (r.name)}
				<rect class="hit" x="0" y={top + k * rowH} width={W} height={rowH} role="button" tabindex="0"
					aria-label={r.name} onpointerenter={() => (hoverRow = k)} onfocus={() => (hoverRow = k)} onpointerleave={() => (hoverRow = -1)} onblur={() => (hoverRow = -1)} onclick={() => (hoverRow = hoverRow === k ? -1 : k)} onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); (hoverRow = hoverRow === k ? -1 : k); } }} />
			{/each}
			<text class="lbl lbl--micro" x={labelW + gutterW / 2 + 1} y={top + rows.length * rowH + 14} text-anchor="middle">rounding</text>
			<text class="lbl lbl--micro" x={x1} y={top + rows.length * rowH + 14} text-anchor="end">log scale, per unit of learning rate</text>
		</svg>
		<Scrubber bind:t={clock} phases={PHASES} label="update number in the book’s training run" duration={7000} />
		<p class="readout" aria-live="polite">
			{#if rowInfo}
				<span class="readout__who"><i>{rowInfo.r.label.sym}</i>{#if rowInfo.r.label.sub}<sub>{rowInfo.r.label.sub}</sub>{/if}, {rowInfo.r.size} numbers, step {step}.</span>
				<span class="readout__part">Gradients from {sci(rowInfo.gmin)} to {sci(rowInfo.gmax)}; Adam’s steps from {sci(rowInfo.dmin)} to {sci(rowInfo.dmax)} times <i>η</i>.</span>
			{:else}
				<span class="readout__who">Step {step} of {STEPS}, <i>η</i> = {sci(snap.eta, 3)}.</span>
				<span class="readout__part">The middle 90% of the gradients span {summary.g.dec.toFixed(1)} decades ({sci(summary.g.lo)} to {sci(summary.g.hi)}); of Adam’s steps, {summary.d.dec.toFixed(1)} ({sci(summary.d.lo)} to {sci(summary.d.hi)} times <i>η</i>).</span>
			{/if}
			<span class="readout__hint">Drag along the line to move through training; point at a row to read it.</span>
		</p>
	{:else}
		{@const half = (W - labelW - 24) / 2}
		<svg viewBox="0 0 {W} {panelH + 18}" width={W} height={panelH + 18} role={interactive ? 'group' : 'img'} aria-label="Gradient sizes and Adam step sizes for every parameter at the first and the last update of the book’s run">
			<text class="lbl" x={labelW + gutterW + 10} y="12"><tspan class="fx-adam__cov">bars: |<tspan class="m">g</tspan><tspan class="m sub" dy="3">i</tspan><tspan dy="-3">|</tspan>, the gradient</tspan> · <tspan class="fx-adam__val">dots: |Δ<tspan class="m">θ</tspan><tspan class="m sub" dy="3">i</tspan><tspan dy="-3">| / </tspan><tspan class="m">η</tspan>, Adam’s step</tspan> · per unit of learning rate, log scale</text>
			{#each printSnaps as ps, c (ps.t)}
				<g transform="translate({c ? labelW + half + 24 : 0} 18)">
					{@render printPanel(ps, c ? half : labelW + half, c === 0)}
				</g>
			{/each}
		</svg>
	{/if}
</div>

{#snippet printPanel(ps, pw, names)}
	{@const L = names ? labelW : 0}
	{@const px = (v) => pxIn(v, L, pw)}
	{@const ax0 = L + gutterW + 10}
	<text class="lbl lbl--head" x={ax0} y="14">step {ps.t}</text>
	{#each ticks as e (e)}
		<line class="rule" x1={px(10 ** e)} x2={px(10 ** e)} y1={top} y2={top + rows.length * rowH} />
		<text class="lbl lbl--micro" x={px(10 ** e)} y={top - 7} text-anchor="middle">{pow10(e)}</text>
	{/each}
	<line class="fx-adam__one" x1={px(1)} x2={px(1)} y1={top - 2} y2={top + rows.length * rowH} />
	<line class="rule" x1={L + gutterW + 2} x2={L + gutterW + 2} y1={top} y2={top + rows.length * rowH} stroke-dasharray="2 3" />
	<text class="lbl lbl--micro" x={L + gutterW / 2 + 1} y={top - 7} text-anchor="middle">&lt;{pow10(LO)}</text>
	{#each rows as r, k (r.name)}
		{@const yy = top + k * rowH}
		{#if names}
			<text class="fx-adam__name" x={L - 8} y={yy + rowH / 2 + 4} text-anchor="end"><tspan class="m">{r.label.sym}</tspan>{#if r.label.sub}<tspan class="m sub" dy="3">{r.label.sub}</tspan>{/if}</text>
		{/if}
		{@const pm = rowMarks(ps.snap, r, yy, (v) => pxIn(v, L, pw))}
		<path class="fx-adam__bars" d={pm.bars} />
		<path class="fx-adam__dots" d={pm.dots} />
	{/each}
	<text class="lbl lbl--micro" x={L + gutterW / 2 + 1} y={top + rows.length * rowH + 14} text-anchor="middle">rounding</text>
{/snippet}

<style>
	.fx-adam :global(.m) { font-family: var(--serif); font-style: italic; }
	.fx-adam :global(.sub) { font-size: 0.72em; }
	.fx-adam__name { font-size: 15px; fill: var(--ink); }
	.fx-adam__bars { stroke: var(--covector); stroke-width: 1px; fill: none; mix-blend-mode: var(--overprint); }
	.fx-adam__dots { fill: var(--value); stroke: none; mix-blend-mode: var(--overprint); opacity: 0.85; }
	.fx-adam__one { stroke: var(--ink-3); stroke-width: 0.8px; }
	.fx-adam__cov { fill: var(--covector-deep); }
	.fx-adam__val { fill: var(--value-deep); }
	.fx-adam__row { transition: opacity var(--t-quick); }
	.fx-adam__row.is-dim { opacity: 0.25; }
</style>

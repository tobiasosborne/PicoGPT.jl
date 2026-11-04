<script module>
	export const meta = { size: 'wide', title: 'The surprise of every prediction' };
	export { default as Key } from './fig-surprise.key.svelte';
</script>

<script>
	// Figure: all N = T·B predictions of the training set. Row b is window b, set under the
	// corpus where it lies; the bar in column k is the surprise −ln q(y) of the prediction of
	// corpus character k from that window's prefix. Outline: the untrained model (θ₀).
	// Solid: the model after `step` steps of the recorded run (replayed and verified by the
	// adapter; without a verified replay, only θ₀ and the trained θ). Hairline: ln V.
	// Interactive: scrub the training step; point at a bar for its context and numbers.
	// Print: outline = untrained, solid = trained.
	import { getContext, onMount } from 'svelte';
	import { measure } from '../primitives/measure.js';
	import Scrubber from '../primitives/Scrubber.svelte';
	import { fmt, showChar, clamp, layoutWidth } from '../primitives/scales.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const V = $derived(specimen.config.V);
	const T = $derived(specimen.config.T);
	const interactive = $derived(mode !== 'print');
	const lnV = $derived(Math.log(specimen.config.V));
	const steps = $derived(specimen.provenance?.steps ?? 0);
	let width = $state(868);

	// the training clock: t ∈ [0, 1] ↦ step on a logarithmic scale (learning is fast early)
	// svelte-ignore state_referenced_locally
	let t = $state(attrs.t != null ? +attrs.t : 1);
	$effect(() => { if (attrs.t != null) t = +attrs.t; });
	const stepAt = (u) => Math.round((steps + 1) ** u - 1);
	const atStep = (s) => Math.log(s + 1) / Math.log(steps + 1);
	let traj = $state(null);
	let trajTried = $state(false);
	function ensureTraj() { if (!trajTried) { trajTried = true; traj = specimen.trajectory(); } }
	onMount(() => {
		if (!interactive) return;
		const idle = window.requestIdleCallback ?? ((f) => setTimeout(f, 300));
		idle(() => ensureTraj());
	});
	const step = $derived(stepAt(t));
	$effect(() => { if (interactive && step !== steps && step !== 0) ensureTraj(); });
	const thetaNow = $derived.by(() => {
		if (step >= steps || !steps) return specimen.theta;
		if (step === 0) return specimen.theta0;
		return traj ? traj.thetas[step] : (t < 0.5 ? specimen.theta0 : specimen.theta);
	});
	const shownStep = $derived(step === 0 || step >= steps || traj ? step : t < 0.5 ? 0 : steps);

	const P0 = $derived(specimen.predictions(specimen.theta0));
	const P = $derived(specimen.predictions(thetaNow));
	const surprise = (pr) => pr.probs.map((p, n) => -Math.log(p[pr.y[Math.floor(n / T)][n % T] - 1]));
	const s0 = $derived(surprise(P0));
	const s = $derived(surprise(P));
	const floor = $derived(specimen.lab.floor);
	const sMax = $derived(Math.max(lnV, ...s0, ...surprise(specimen.predictions(specimen.theta))) * 1.08);

	// geometry
	const W = $derived(layoutWidth(width, mode, 340));
	const narrow = $derived(W < 640);
	const n = $derived(specimen.corpus.length);
	const L = $derived(narrow ? 6 : 104);
	const R = $derived(narrow ? 6 : 64);
	const cw = $derived((W - L - R) / n);
	const cx = (k) => L + cw * (k + 0.5);
	const rowH = $derived(narrow ? 26 : 32);
	const top = 66;
	const B = $derived(P.B);
	const H = $derived(top + B * rowH + 64);
	const barW = $derived(Math.min(15, cw * 0.42));
	const yb = $derived(top + B * rowH + 30);
	const hOf = (v) => (Math.min(v, sMax) / sMax) * (rowH - 5);

	let hover = $state(-1);
	const sel = $derived(hover >= 0 ? { b: Math.floor(hover / T), t: hover % T } : null);
	const loss = $derived(s.reduce((a, v) => a + v, 0) / s.length);
	const readout = $derived.by(() => {
		if (!sel) return null;
		const nn = sel.b * T + sel.t;
		const ctx = specimen.corpus.slice(sel.b, sel.b + sel.t + 1);
		const y = specimen.corpus[sel.b + sel.t + 1];
		const q = Math.exp(-s[nn]);
		return { ctx: Array.from(ctx).map(showChar).join(''), y: showChar(y), q, s: s[nn], s0: s0[nn], b: sel.b, t: sel.t };
	});
	const PHASES = $derived((narrow ? [0, 10] : [0, 1, 10, 100]).filter((x) => x < steps).map((x) => ({ at: atStep(x), label: x === 0 ? 'untrained' : `step ${x}` })).concat([{ at: 1, label: `trained (${steps})` }]));
</script>

<div class="fx fx-surp" use:measure={(w) => (width = w)}>
	<svg viewBox="0 0 {W} {H}" width={W} height={H} role="img" aria-label="Surprise of each of the {s.length} predictions at step {shownStep}; mean {fmt(loss, 4)} nats, uniform guessing {fmt(lnV, 2)}">
		<!-- the corpus: one column per character -->
		{#each Array.from(specimen.corpus) as ch, k (k)}
			{@const inCtx = sel && k >= sel.b && k <= sel.b + sel.t}
			{@const isY = sel && k === sel.b + sel.t + 1}
			<text class="tok fx-surp__ch" class:is-ctx={inCtx} class:is-y={isY} x={cx(k)} y={top - 22} text-anchor="middle">{showChar(ch)}</text>
		{/each}
		{#if sel}
			<line class="fx-surp__ctxline" x1={cx(sel.b) - cw * 0.4} x2={cx(sel.b + sel.t) + cw * 0.4} y1={top - 16} y2={top - 16} />
		{/if}
		{#if !narrow}<text class="lbl lbl--micro" x={L - 10} y={top - 22} text-anchor="end">corpus</text>{/if}

		{#each { length: B } as _, b (b)}
			{@const y0 = top + (b + 1) * rowH}
			<g class="fx-surp__row" class:is-on={sel?.b === b}>
				{#if !narrow}
					<text class="lbl lbl--micro" x={L - 10} y={y0 - 4} text-anchor="end">window {b + 1}</text>
				{/if}
				<line class="fx-surp__base" x1={cx(b + 1) - cw * 0.5} x2={cx(b + T) + cw * 0.5} y1={y0} y2={y0} />
				<line class="fx-surp__lnv" x1={cx(b + 1) - cw * 0.5} x2={cx(b + T) + cw * 0.5} y1={y0 - hOf(lnV)} y2={y0 - hOf(lnV)} />
				{#each { length: T } as _, tt (tt)}
					{@const nn = b * T + tt}
					{@const x = cx(b + 1 + tt)}
					<rect class="fx-surp__was" x={x - barW / 2} y={y0 - hOf(s0[nn])} width={barW} height={hOf(s0[nn])} />
					<rect class="bar--value fx-surp__bar" class:is-on={hover === nn} x={x - barW / 2 + 1.5} y={y0 - hOf(s[nn])} width={barW - 3} height={hOf(s[nn])} />
					{#if s[nn] > sMax}<path class="fx-surp__cap" d="M{x - 3} {y0 - rowH + 7} l3 -3 l3 3" />{/if}
					{#if interactive}
						<rect class="hit" x={x - cw / 2} y={y0 - rowH + 2} width={cw} height={rowH} role="button" tabindex="-1" aria-label="window {b + 1}, position {tt + 1}" onpointerenter={() => (hover = nn)} onpointerleave={() => (hover = -1)} onkeydown={() => {}} />
					{/if}
				{/each}
			</g>
		{/each}
		<!-- direct labels on the first row -->
		{#if !narrow}
			<text class="lbl lbl--micro fx-surp__lnvl" x={cx(T) + cw * 0.5 + 6} y={top + rowH - hOf(lnV) + 3.5}>ln {V}: uniform guessing</text>
		{/if}

		<!-- the mean of all bars is the loss -->
		<text class="lbl" x={L} y={yb}>mean of the {s.length} bars, the loss: <tspan class="fx-surp__num">{fmt(loss, 4)}</tspan> nats<tspan class="fx-surp__dim">{shownStep === 0 ? '' : `  (untrained: ${fmt(P0.loss, 4)})`}</tspan></text>
		<text class="lbl lbl--micro" x={L} y={yb + 18}>uniform guessing: ln {V} = {fmt(lnV, 4)} · the floor no model can beat: {fmt(floor, 4)}</text>
		<text class="lbl lbl--head" x={W - 4} y={14} text-anchor="end">{shownStep === 0 ? 'untrained' : shownStep >= steps ? `trained, step ${steps}` : `step ${shownStep}`}</text>
	</svg>
	{#if interactive}
		<Scrubber bind:t phases={PHASES} label="training step" duration={4200} />
		<p class="readout" aria-live="polite">
			{#if readout}
				<span class="readout__who">Window {readout.b + 1}, position {readout.t + 1}.</span>
				<span class="readout__part">After ‘{readout.ctx}’ the next character is ‘{readout.y}’. At step {shownStep} the model gives it {fmt(readout.q, 3)}, a surprise of {fmt(readout.s, 3)} nats (untrained: {fmt(readout.s0, 2)}).</span>
			{:else}
				<span class="readout__hint">Point at a bar to see its context and its numbers. Press play or drag along the line to train the model again, step by step, as it was trained.</span>
			{/if}
		</p>
	{/if}
</div>

<style>
	.fx-surp__ch { font-size: 17px; fill: var(--ink-2); }
	.fx-surp__ch.is-ctx { fill: var(--ink); font-weight: 600; }
	.fx-surp__ch.is-y { fill: var(--value-deep); font-weight: 650; }
	.fx-surp__ctxline { stroke: var(--ink); stroke-width: 1px; }
	.fx-surp__base { stroke: var(--ink-3); stroke-width: var(--hair); }
	.fx-surp__lnv { stroke: var(--rule); stroke-width: var(--hair); stroke-dasharray: 2 2; }
	.fx-surp__was { fill: none; stroke: var(--ink-3); stroke-width: var(--hair); }
	.fx-surp__bar.is-on { fill: var(--value-deep); }
	.fx-surp__cap { fill: none; stroke: var(--value); stroke-width: 1px; }
	.fx-surp__row.is-on .fx-surp__base { stroke: var(--ink); }
	.fx-surp__num { fill: var(--ink); font-weight: 600; }
	.fx-surp__dim { fill: var(--ink-3); }
	.fx-surp__lnvl { fill: var(--ink-3); }
</style>

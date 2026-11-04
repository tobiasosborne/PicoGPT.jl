<script module>
	export const meta = { size: 'wide', title: 'The learning rate and the length of the gradient' };
</script>

<script>
	// Figure: the book's training run, step by step (k = 1 … 200, linear axis):
	//   above   the learning rate η_k of update k: linear warmup to η, then half a cosine
	//           down to η_min (value ink: a setting of the forward computation)
	//   below   the length ‖g_k‖ of the gradient before clipping, on a log scale, and the
	//           radius c of the ball the clip projects onto (covector ink: a gradient)
	// Point at a step to read both. All numbers from specimen.lab.run() (the artifact's
	// recipe; it reproduces the artifact's loss trajectory).
	import { getContext } from 'svelte';
	import { measure } from '../primitives/measure.js';
	import { computed } from '../primitives/computed.svelte.js';
	import { fmt, clamp, layoutWidth } from '../primitives/scales.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = getContext('specimen')();
	const lab = specimen.lab;
	const R = lab.recipe;
	const steps = lab.steps;
	const interactive = $derived(mode !== 'print');

	const job = computed(`fig-schedule:${steps}:${R.eta}:${R.schedule.warmup}:${R.schedule.etaMin}:${R.clip}`, function* () {
		const run = lab.run();
		for (let k = 0; k < steps; k++) { run.step(); if (k % 25 === 0) yield k / steps; }
		return { lr: run.lr, g: run.gradNorm, clipped: run.clipped.map((c) => (c ? 1 : 0)) };
	});

	let width = $state(868);
	let cursor = $state(null);
	const W = $derived(layoutWidth(width, mode));
	const narrow = $derived(W < 560);
	const Lm = $derived(narrow ? 40 : 52);
	const Rm = $derived(narrow ? 12 : 132);
	const pw = $derived(W - Lm - Rm);
	const X = (k) => Lm + (pw * k) / steps;
	const top1 = 40, h1 = 150;
	const top2 = top1 + h1 + 58, h2 = 150;
	const H = top2 + h2 + 34;

	const etaY = (v) => top1 + h1 * (1 - v / R.eta);
	// gradient lengths: log axis spanning the run's values and the radius c
	const gRange = $derived.by(() => {
		const g = job.value?.g ?? [R.clip];
		const lo = Math.floor(Math.log10(Math.min(...g, R.clip)));
		const hi = Math.ceil(Math.log10(Math.max(...g, R.clip)) + 0.05);
		return [lo, Math.max(hi, lo + 1)];
	});
	const gY = (v) => top2 + h2 * (1 - (Math.log10(v) - gRange[0]) / (gRange[1] - gRange[0]));
	const gTicks = $derived(Array.from({ length: gRange[1] - gRange[0] + 1 }, (_, i) => gRange[0] + i));
	const sup = (d) => String(d).replace(/-/, '⁻').replace(/[0-9]/g, (c) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+c]);

	const etaPath = $derived(job.value ? job.value.lr.map((v, i) => `${i ? 'L' : 'M'}${X(i + 1).toFixed(1)} ${etaY(v).toFixed(1)}`).join('') : '');
	const gPath = $derived(job.value ? job.value.g.map((v, i) => `${i ? 'L' : 'M'}${X(i + 1).toFixed(1)} ${gY(v).toFixed(1)}`).join('') : '');
	const nClipped = $derived(job.value ? job.value.clipped.reduce((a, b) => a + b, 0) : 0);
	const gMax = $derived(job.value ? Math.max(...job.value.g) : NaN);
	const kTicks = $derived(Array.from({ length: Math.floor(steps / 50) + 1 }, (_, i) => i * 50));

	function point(e) {
		const svg = e.currentTarget.ownerSVGElement;
		const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
		const p = pt.matrixTransform(svg.getScreenCTM().inverse());
		cursor = clamp(Math.round(((p.x - Lm) / pw) * steps), 1, steps);
	}
	function key(e) {
		const d = e.shiftKey ? 10 : 1;
		if (e.key === 'ArrowRight') { cursor = clamp((cursor ?? 0) + d, 1, steps); e.preventDefault(); }
		if (e.key === 'ArrowLeft') { cursor = clamp((cursor ?? 2) - d, 1, steps); e.preventDefault(); }
		if (e.key === 'Escape') cursor = null;
	}
	const at = $derived(cursor != null && job.value ? { k: cursor, eta: job.value.lr[cursor - 1], g: job.value.g[cursor - 1], clipped: job.value.clipped[cursor - 1] } : null);
	const sci = (x) => {
		const e = Math.floor(Math.log10(Math.abs(x)));
		return e >= -2 ? fmt(x, e >= -1 ? 3 : 4) : `${fmt(x / 10 ** e, 2)} × 10${sup(e)}`;
	};
</script>

<div class="fx fx-sched" use:measure={(w) => (width = w)}>
	{@html job.tag}
	<svg viewBox="0 0 {W} {H}" width={W} height={H} role="img"
		aria-label="The learning rate rises linearly to {fmt(R.eta, 3)} over {R.schedule.warmup} steps, then falls along a cosine to {fmt(R.schedule.etaMin, 4)} at step {steps}. The gradient's length stays below the clip radius {fmt(R.clip, 0)} at every step{job.value ? `; its largest value is ${fmt(gMax, 3)}` : ''}.">
		<!-- the learning rate -->
		<text class="lbl lbl--head" x={Lm} y={top1 - 26}>learning rate <tspan class="m">η</tspan><tspan class="m sub" dy="3">k</tspan></text>
		<line class="axis" x1={Lm} x2={Lm} y1={top1} y2={top1 + h1} />
		<line class="axis" x1={Lm} x2={Lm + pw} y1={top1 + h1} y2={top1 + h1} />
		{#each [0, 0.5, 1] as f (f)}
			<line class="axis" x1={Lm - 3} x2={Lm} y1={etaY(f * R.eta)} y2={etaY(f * R.eta)} />
			<text class="lbl lbl--micro" x={Lm - 6} y={etaY(f * R.eta) + 3.5} text-anchor="end">{fmt(f * R.eta, f ? 3 : 0)}</text>
		{/each}
		<line class="rule" x1={X(R.schedule.warmup)} x2={X(R.schedule.warmup)} y1={top1} y2={top1 + h1} />
		<text class="lbl lbl--micro" x={X(R.schedule.warmup) + 5} y={top1 + h1 - 6}>end of warmup, step {R.schedule.warmup}</text>
		<path class="fx-sched__eta" d={etaPath} />
		{#if !narrow}
			<text class="lbl lbl--micro fx-sched__dl" x={X(R.schedule.warmup)} y={etaY(R.eta) - 8} text-anchor="middle"><tspan class="m">η</tspan> = {fmt(R.eta, 3)}</text>
			<text class="lbl lbl--micro fx-sched__dl" x={X(steps) + 6} y={etaY(R.schedule.etaMin) + 3}><tspan class="m">η</tspan><tspan class="sub" dy="2">min</tspan><tspan dy="-2"> = {fmt(R.schedule.etaMin, 4)}</tspan></text>
			<text class="lbl lbl--micro" x={X(steps) + 6} y={etaY(R.schedule.etaMin) + 17}>at step {steps}</text>
		{/if}

		<!-- the gradient's length, and the ball it is clipped to -->
		<text class="lbl lbl--head" x={Lm} y={top2 - 12}>length of the gradient <tspan class="fx-sched__cov">‖<tspan class="m">g</tspan><tspan class="m sub" dy="3">k</tspan><tspan dy="-3">‖</tspan></tspan>, before clipping</text>
		<line class="axis" x1={Lm} x2={Lm} y1={top2} y2={top2 + h2} />
		<line class="axis" x1={Lm} x2={Lm + pw} y1={top2 + h2} y2={top2 + h2} />
		{#each gTicks as d (d)}
			<line class="axis" x1={Lm - 3} x2={Lm} y1={gY(10 ** d)} y2={gY(10 ** d)} />
			<text class="lbl lbl--micro" x={Lm - 6} y={gY(10 ** d) + 3.5} text-anchor="end">{d === 0 ? '1' : `10${sup(d)}`}</text>
		{/each}
		<rect class="fx-sched__outside" x={Lm} y={top2} width={pw} height={Math.max(0, gY(R.clip) - top2)} />
		<line class="fx-sched__clip" x1={Lm} x2={Lm + pw} y1={gY(R.clip)} y2={gY(R.clip)} />
		{#if !narrow}
			<text class="lbl lbl--micro fx-sched__dl" x={X(steps) + 6} y={gY(R.clip) + 3.5}>clip radius <tspan class="m">c</tspan> = {fmt(R.clip, 0)}</text>
			<text class="lbl lbl--micro" x={X(steps) + 6} y={gY(R.clip) - 10}>clipped above</text>
		{/if}
		<path class="fx-sched__g" d={gPath} />
		{#if job.value}
			{#each job.value.g as v, i (i)}
				{#if job.value.clipped[i]}<circle class="fx-sched__gc" cx={X(i + 1)} cy={gY(v)} r="2.6" />{/if}
			{/each}
			{#if !narrow}
				<text class="lbl lbl--micro fx-sched__dl fx-sched__covtext" x={X(steps) + 6} y={gY(job.value.g.at(-1)) + 3.5}>{fmt(job.value.g.at(-1), 3)}</text>
			{/if}
		{:else}
			<text class="lbl lbl--micro" x={Lm + 8} y={top2 + h2 / 2}>computing the run… {Math.round(job.progress * 100)} %</text>
		{/if}
		{#each kTicks as k (k)}
			<line class="axis" x1={X(k)} x2={X(k)} y1={top2 + h2} y2={top2 + h2 + 3.5} />
			<text class="lbl lbl--micro" x={X(k)} y={top2 + h2 + 15} text-anchor="middle">{k}</text>
			<line class="axis" x1={X(k)} x2={X(k)} y1={top1 + h1} y2={top1 + h1 + 3.5} />
			<text class="lbl lbl--micro" x={X(k)} y={top1 + h1 + 15} text-anchor="middle">{k}</text>
		{/each}
		<text class="lbl lbl--micro" x={Lm + pw} y={top2 + h2 + 29} text-anchor="end">update k</text>

		{#if at}
			<line class="fx-sched__cursor" x1={X(at.k)} x2={X(at.k)} y1={top1} y2={top2 + h2} />
			<circle class="fx-sched__cdot" cx={X(at.k)} cy={etaY(at.eta)} r="3.2" />
			<circle class="fx-sched__cdot fx-sched__cdot--cov" cx={X(at.k)} cy={gY(at.g)} r="3.2" />
		{/if}
		{#if interactive}
			<rect class="hit fx-sched__hit" x={Lm} y={top1} width={pw} height={top2 + h2 - top1} role="slider" tabindex="0"
				aria-label="update number" aria-valuemin="1" aria-valuemax={steps} aria-valuenow={cursor ?? 1}
				onpointermove={point} onpointerdown={point} onpointerleave={() => (cursor = null)} onkeydown={key} />
		{/if}
	</svg>
	{#if interactive}
		<p class="readout" aria-live="polite">
			{#if at}
				<span class="readout__who">Update {at.k}.</span>
				<span class="readout__part">η<sub>k</sub> = {sci(at.eta)}; the gradient has length {sci(at.g)}, {at.clipped ? `longer than c = ${fmt(R.clip, 0)}, so it is shortened to c` : `inside the ball of radius ${fmt(R.clip, 0)}, so the clip leaves it alone`}.</span>
			{:else}
				<span class="readout__part">{job.value ? `${nClipped === 0 ? 'No update' : `${nClipped} of the ${steps} updates`} of the book’s run ${nClipped === 1 ? 'was' : nClipped === 0 ? 'was' : 'were'} clipped: the longest gradient has length ${fmt(gMax, 3)}.` : 'Computing the run…'}</span>
				<span class="readout__hint">Point at a step, or use ← and →.</span>
			{/if}
		</p>
	{/if}
</div>

<style>
	.fx-sched :global(.m) { font-family: var(--serif); font-style: italic; font-size: 13.5px; fill: var(--ink); }
	.fx-sched :global(.sub) { font-size: 9.5px; }
	.fx-sched__eta { fill: none; stroke: var(--value); stroke-width: calc(var(--line) * 1.3); stroke-linejoin: round; }
	.fx-sched__g { fill: none; stroke: var(--covector); stroke-width: var(--line); stroke-linejoin: round; mix-blend-mode: var(--overprint); }
	.fx-sched__gc { fill: url(#hatch-covector); stroke: var(--covector); stroke-width: 0.8px; }
	.fx-sched__clip { stroke: var(--ink); stroke-width: var(--hair); }
	.fx-sched__outside { fill: url(#hatch-ink); opacity: 0.35; }
	.fx-sched__dl { fill: var(--ink-2); }
	.fx-sched__covtext { fill: var(--covector-deep); }
	.fx-sched__cov { fill: var(--covector-deep); }
	.fx-sched__cov :global(.m) { fill: var(--covector-deep); }
	.fx-sched__cursor { stroke: var(--ink); stroke-width: var(--hair); }
	.fx-sched__cdot { fill: var(--paper); stroke: var(--value); stroke-width: 1.4px; }
	.fx-sched__cdot--cov { stroke: var(--covector); }
	.fx-sched__hit { cursor: crosshair; }
</style>

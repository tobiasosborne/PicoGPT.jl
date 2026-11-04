<script module>
	export const meta = { size: 'wide', title: 'When more training makes prediction worse' };
</script>

<script>
	import { getContext } from 'svelte';
	import LossChart from '../primitives/LossChart.svelte';
	import { measure } from '../primitives/measure.js';
	import { fmt, layoutWidth } from '../primitives/scales.js';

	let { mode = 'interactive' } = $props();
	const data = $derived(getContext('artifacts')?.()?.overfitting);
	const runs = $derived([data?.small, data?.large].filter(Boolean));
	let width = $state(868);
	let picked = $state(null);
	const W = $derived(layoutWidth(width, mode, 320));
	const stacked = $derived(W < 700);
	const gap = 36;
	const cw = $derived(stacked ? W : (W - gap) / 2);
	const ch = 260, heading = 42, foot = 32;
	const panelH = heading + ch + foot;
	const H = $derived(stacked ? panelH * runs.length + gap : panelH);
	const xMax = $derived(Math.max(...runs.map((r) => r.final.step)));
	const yMax = $derived(Math.ceil(Math.max(...runs.flatMap((r) => [...r.train_loss, ...r.val_loss])) * 2) / 2);
	// LossChart's standard insets, with room on the right for direct labels.
	const margin = 86;
	const X = (k) => 34 + (cw - 34 - 10 - margin) * k / xMax;
	const Y = (v) => heading + 16 + (ch - 16 - 26) * (1 - v / yMax);
	const integer = (n) => n.toLocaleString('en-GB').replaceAll(',', ' ');
	const px = (j) => stacked ? 0 : j * (cw + gap);
	const py = (j) => stacked ? j * (panelH + gap) : 0;
	const nearest = (r, k) => r.step.reduce((a, b) => Math.abs(b - k) < Math.abs(a - k) ? b : a);
	function key(e, r, j) {
		const current = picked?.runIndex === j ? picked.step : r.best.step;
		const i = r.step.indexOf(current);
		if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
			e.preventDefault();
			picked = { runIndex: j, step: r.step[Math.max(0, Math.min(r.step.length - 1, i + (e.key === 'ArrowRight' ? 1 : -1)))] };
		} else if (e.key === 'Escape') picked = null;
	}
</script>

<div class="fx fx-overfit" use:measure={(w) => width = w}>
	{#if runs.length !== 2}
		<p class="fig__pending">The two Shakespeare loss records are not present in this build.</p>
	{:else}
		<svg viewBox="0 0 {W} {H}" width={W} height={H} role={mode === 'print' ? 'img' : 'group'} aria-label="Training and validation losses of both models on shared axes. Large model: best validation {fmt(data.large.best.val_loss, 3)}, final validation {fmt(data.large.final.val_loss, 3)}, final training {fmt(data.large.final.train_loss, 3)} nats.">
			{#each runs as r, j (r.model.param_count)}
				<g transform="translate({px(j)} {py(j)})">
					<text class="lbl lbl--head" x="34" y="12">{j === 0 ? 'The book’s model' : 'The larger model'}</text>
					<text class="lbl lbl--micro" x="34" y="29">{integer(r.model.param_count)} parameters · {integer(r.final.step)} steps</text>
					<LossChart y={heading} width={cw} height={ch} series={[
						{ steps: r.step, values: r.train_loss },
						{ steps: r.step, values: r.val_loss, style: 'dash' }
					]} {xMax} {yMax} {margin} lnV={r.uniform} xScale="linear" xLabel="training step" refLabels={false} />
					<text class="lbl lbl--micro overfit__ref" x={X(xMax) - 3} y={Y(r.uniform) - 7} text-anchor="end">uniform guessing · ln {r.model.V}</text>
					<!-- The final gap is a measured interval, not a coloured category. -->
					<path class="overfit__gap" d="M{X(r.final.step) + 5} {Y(r.final.val_loss)}h5V{Y(r.final.train_loss)}h-5" />
					<text class="lbl lbl--micro" x={X(r.final.step) + 16} y={Y(r.final.val_loss) - 3}>held-out<tspan x={X(r.final.step) + 16} dy="13">{fmt(r.final.val_loss, 3)}</tspan></text>
					<text class="lbl lbl--micro lbl--value" x={X(r.final.step) + 16} y={Y(r.final.train_loss) + 8}>training<tspan x={X(r.final.step) + 16} dy="13">{fmt(r.final.train_loss, 3)}</tspan></text>
					<circle class="overfit__best" cx={X(r.best.step)} cy={Y(r.best.val_loss)} r="4" />
					<path class="leader" d="M{X(r.best.step)} {Y(r.best.val_loss) - 6}v-74h18" />
					<text class="lbl lbl--micro" x={X(r.best.step) + 22} y={Y(r.best.val_loss) - 84}>best held-out {fmt(r.best.val_loss, 3)}<tspan x={X(r.best.step) + 22} dy="13">step {integer(r.best.step)}</tspan></text>
					<text class="lbl lbl--micro" x="34" y={panelH - 3}>Solid: training · dashed: held-out · bracket: final gap</text>
					{#if mode !== 'print'}
						{#if picked?.runIndex === j}
							<line class="overfit__cursor" x1={X(picked.step)} x2={X(picked.step)} y1={heading + 16} y2={heading + ch - 26} />
						{/if}
						<rect class="hit" x="34" y={heading + 16} width={X(xMax) - 34} height={ch - 16 - 26} role="slider" tabindex="0" aria-label="{j === 0 ? 'Book' : 'Large'} model checkpoint" aria-valuemin={r.step[0]} aria-valuemax={r.final.step} aria-valuenow={picked?.runIndex === j ? picked.step : r.best.step}
							onpointermove={(e) => {
								const pt = e.currentTarget.ownerSVGElement.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
								const p = pt.matrixTransform(e.currentTarget.getScreenCTM().inverse());
								picked = { runIndex: j, step: nearest(r, (p.x - 34) / (X(xMax) - 34) * xMax) };
							}} onpointerleave={() => picked = null} onfocus={() => picked = { runIndex: j, step: r.best.step }} onblur={() => picked = null} onkeydown={(e) => key(e, r, j)} />
					{/if}
				</g>
			{/each}
		</svg>
		{#if mode !== 'print'}
			<p class="readout" aria-live="polite">
				{#if picked}
					{@const r = runs[picked.runIndex]}
					{@const k = r.step.indexOf(picked.step)}
					<span class="readout__who">{picked.runIndex === 0 ? 'Book’s model' : 'Larger model'}, step {integer(picked.step)}.</span>
					Training {fmt(r.train_loss[k], 3)}, held-out {fmt(r.val_loss[k], 3)} nats per character.
				{:else}<span class="readout__hint">Point along either plot; focus and use ← → to read recorded checkpoints. Both axes have the same scale.</span>{/if}
			</p>
		{/if}
	{/if}
</div>

<style>
	.overfit__best { fill: var(--paper); stroke: var(--ink); stroke-width: 1.4px; }
	.overfit__gap, .overfit__cursor { fill: none; stroke: var(--ink-2); stroke-width: var(--hair); }
	.overfit__ref { fill: var(--ink-2); paint-order: stroke; stroke: var(--paper); stroke-width: 3px; }
	.hit { cursor: crosshair; }
</style>

<script module>
	export const meta = { size: 'wide', title: 'What the Shakespeare model learned, and when' };
</script>

<script>
	// Figure: the Tiny Shakespeare model's loss during training (step on a log axis; step 0
	// set apart at the left), on its training text (solid line, filled dots) and on held-out
	// text it never trained on (open dots), against ln V, uniform guessing. In the margin, the
	// sample the model wrote at τ = 1 at each saved checkpoint, joined to its step by a leader,
	// with the share of its words that occur in the training text. Point at a checkpoint for
	// the whole sample. Data: artifacts/shakespeare/{loss,samples}.json via the server.
	import { getContext } from 'svelte';
	import { measure } from '../primitives/measure.js';
	import { fmt, layoutWidth } from '../primitives/scales.js';

	let { mode = 'interactive' } = $props();
	const data = $derived(getContext('artifacts')?.()?.shakespeare ?? null);
	const loss = $derived(data?.loss ?? null);
	const interactive = $derived(mode !== 'print');
	let width = $state(868);
	let pick = $state(-1);

	const tauShown = 1;
	const samples = $derived(
		data?.samples ? data.samples.steps.map((s) => data.samples.samples.find((x) => x.step === s && x.temperature === tauShown)).filter(Boolean) : []
	);

	// geometry
	const W = $derived(layoutWidth(width, mode, 340));
	const narrow = $derived(W < 720);
	const L = 34;
	const cx1 = $derived(narrow ? W - 20 : Math.round(W * 0.42));
	const plotTop = 22;
	const blockH = 74;
	const xS = $derived(narrow ? 0 : cx1 + 56);
	const plotH = $derived(narrow ? 260 : Math.max(300, samples.length * blockH - 30));
	const yBase = $derived(plotTop + plotH);
	const H = $derived(narrow ? yBase + 50 + samples.length * (blockH + 6) : yBase + 52);

	const steps = $derived(loss?.step ?? []);
	const sMin = $derived(Math.min(...steps.filter((s) => s > 0)));
	const sMax = $derived(Math.max(...steps));
	const gap0 = 26; // step 0 is set apart
	const xOf = (s) => (s <= 0 ? L : L + gap0 + ((Math.log10(s) - Math.log10(sMin)) / (Math.log10(sMax) - Math.log10(sMin) || 1)) * (cx1 - L - gap0));
	const yMax = $derived(loss ? Math.ceil(Math.max(loss.uniform, ...loss.train, ...loss.val) * 2) / 2 : 5);
	const yOf = (v) => yBase - (v / yMax) * plotH;
	const decades = $derived([100, 1000, 10000, 100000].filter((d) => d > sMin && d < sMax));
	const linePath = (ys) => 'M' + steps.map((s, k) => `${xOf(s).toFixed(1)} ${yOf(ys[k]).toFixed(1)}`).join('L');
	const kEnd = $derived(steps.length - 1);
	const valAt = (s) => { const k = steps.indexOf(s); return k >= 0 ? loss.val[k] : null; };
	const yBlock = (k) => (narrow ? yBase + 50 + k * (blockH + 6) : plotTop + k * blockH);
	const lineChars = $derived(Math.max(24, Math.floor((W - xS - 6) / 7.6)));
	const lines = (text, n = 3, w = lineChars) => text.replace(/^\n+/, '').split('\n').filter((l) => l.length).slice(0, n).map((l) => (l.length > w ? l.slice(0, w - 1) + '…' : l));
	const pct = (w) => (w && w.total ? Math.round((100 * w.known) / w.total) : null);
	const active = $derived(pick >= 0 ? samples[pick] : null);
</script>

<div class="fx fx-ckpt" use:measure={(w) => (width = w)}>
	{#if !loss}
		<p class="fig__pending">The Shakespeare model’s training record (artifacts/shakespeare/loss.json) is not present in this build.</p>
	{:else}
		<svg viewBox="0 0 {W} {H}" width={W} height={H} role="img" aria-label="Loss of the Shakespeare model from {fmt(loss.val[0], 2)} to {fmt(loss.val.at(-1), 2)} nats on held-out text over {sMax} steps; uniform guessing {fmt(loss.uniform, 2)}">
			<!-- axes -->
			<line class="axis" x1={L} x2={L} y1={plotTop} y2={yBase} />
			{#each Array.from({ length: Math.floor(yMax) + 1 }, (_, k) => k) as v (v)}
				<line class="axis" x1={L - 4} x2={L} y1={yOf(v)} y2={yOf(v)} />
				<text class="lbl lbl--micro" x={L - 7} y={yOf(v) + 3.5} text-anchor="end">{v}</text>
			{/each}
			<text class="lbl lbl--micro" x={L - 7} y={plotTop - 8} text-anchor="end">nats</text>
			<line class="axis" x1={L + gap0 - 6} x2={cx1} y1={yBase} y2={yBase} />
			<line class="axis" x1={L} x2={L + 8} y1={yBase} y2={yBase} />
			{#each [0, sMin, ...decades, sMax] as s (s)}
				<line class="axis" x1={xOf(s)} x2={xOf(s)} y1={yBase} y2={yBase + 4} />
				<text class="lbl lbl--micro" x={xOf(s)} y={yBase + 16} text-anchor="middle">{s.toLocaleString('en').replace(/,/g, ' ')}</text>
			{/each}
			<text class="lbl lbl--micro" x={cx1} y={yBase + 32} text-anchor="end">training step (logarithmic)</text>

			<!-- uniform guessing -->
			<line class="fx-ckpt__uni" x1={L} x2={cx1} y1={yOf(loss.uniform)} y2={yOf(loss.uniform)} />
			<text class="lbl lbl--micro" x={cx1} y={yOf(loss.uniform) - 6} text-anchor="end">ln {loss.model.V} = {fmt(loss.uniform, 2)}: uniform guessing</text>

			<!-- the two curves -->
			<path class="fx-ckpt__train" d={linePath(loss.train)} />
			<path class="fx-ckpt__val" d={linePath(loss.val)} />
			{#each steps as s, k (s)}
				<circle class="dot--value" cx={xOf(s)} cy={yOf(loss.train[k])} r="2.6" />
				<circle class="dot--open" cx={xOf(s)} cy={yOf(loss.val[k])} r="2.8" />
			{/each}
			<text class="lbl lbl--value" x={xOf(sMax)} y={yOf(loss.train[kEnd]) + 20} text-anchor="end">training text {fmt(loss.train[kEnd], 2)}</text>
			<text class="lbl" x={xOf(sMax) - 4} y={yOf(loss.val[kEnd]) + 17} text-anchor="end">held-out text {fmt(loss.val[kEnd], 2)}</text>

			<!-- samples in the margin -->
			{#each samples as smp, k (smp.step)}
				{@const v = valAt(smp.step)}
				{@const yb = yBlock(k)}
				{@const pc = pct(smp.words)}
				<g class="fx-ckpt__smp" class:is-on={pick === k} onpointerenter={() => interactive && (pick = k)} onpointerleave={() => interactive && (pick = -1)} role="presentation">
					{#if v != null && !narrow}
						<line class="leader" x1={xOf(smp.step) + 3} y1={yOf(v)} x2={xS - 6} y2={yb + 4} />
						<circle class="fx-ckpt__mark" cx={xOf(smp.step)} cy={yOf(v)} r="4.5" />
					{/if}
					<text class="lbl lbl--micro" x={xS} y={yb + 8}><tspan class="fx-ckpt__step">step {smp.step.toLocaleString('en').replace(/,/g, ' ')}</tspan>{v != null ? ` · held-out ${fmt(v, 2)} nats` : ''}{pc != null ? ` · ${pc} % of words real` : ''}</text>
					{#each lines(smp.text) as ln, j (j)}
						<text class="fx-ckpt__text" x={xS} y={yb + 24 + j * 14.5} xml:space="preserve">{ln}</text>
					{/each}
					{#if interactive}<rect class="hit" x={xS - 4} y={yb - 4} width={W - xS + 4} height={blockH - 6} />{/if}
				</g>
			{/each}
		</svg>
		{#if interactive}
			<div class="readout fx-ckpt__readout" aria-live="polite">
				{#if active}
					<span class="readout__who">Step {active.step}, τ = {active.temperature}{pct(active.words) != null ? `, ${active.words.known} of ${active.words.total} words real` : ''}.</span>
					<pre class="fx-ckpt__full">{active.text.replace(/^\n+/, '')}</pre>
				{:else}
					<span class="readout__hint">Point at a sample for all of it. Each starts from a newline and runs {data.samples.length} characters; every character drawn from softmax(z/τ) at τ = {tauShown}.</span>
				{/if}
			</div>
		{/if}
	{/if}
</div>

<style>
	.fx-ckpt__train { fill: none; stroke: var(--value); stroke-width: 1.6px; }
	.fx-ckpt__val { fill: none; stroke: var(--value); stroke-width: var(--hair); }
	.fx-ckpt__uni { stroke: var(--ink-3); stroke-width: var(--hair); stroke-dasharray: 3 3; }
	.fx-ckpt__mark { fill: none; stroke: var(--ink); stroke-width: 1px; }
	.fx-ckpt__text { font-family: var(--serif); font-size: 12.5px; fill: var(--ink); white-space: pre; }
	.fx-ckpt__step { fill: var(--ink); font-weight: 600; }
	.fx-ckpt__smp.is-on .fx-ckpt__mark { stroke-width: 2px; }
	.fx-ckpt__smp.is-on .leader { stroke: var(--ink); }
	.fx-ckpt__readout { min-height: 8em; }
	.fx-ckpt__full { font-family: var(--serif); font-size: 13px; line-height: 1.35; color: var(--ink); white-space: pre-wrap; margin: 0.4em 0 0; max-height: 16em; overflow: auto; }
</style>

<script module>
	export const meta = { size: 'wide', title: 'How reliable the logit lens is, depth by depth, in the Shakespeare model' };
	export { default as Key } from './fig-j-logit-lens-depth.key.svelte';
</script>

<script>
	// Figure: the logit lens of the Shakespeare model at its nine depths, on held-out text.
	// x: the depth axis (DepthAxis). y: a fraction of the sampled positions, 0…1.
	// Filled dots (joined by a line): how often the lens's top character is the model's own
	// top character. Open rings (joined by a hairline): how often it is the character that
	// actually comes next. Two references: uniform guessing (1/V) and the model's own
	// accuracy (the ring at the last depth, where the lens is the model).
	// Data: jspace.shakespeare.lenses.metrics[j].logit_lens.{model_agreement, top1_accuracy,
	// top3_accuracy, kl_model_lens, positions}; depth, depth_kind, model.config.V.
	// Interactive: point at a depth for its numbers. Print: static.
	import { getContext } from 'svelte';
	import { measure } from '#figures/primitives/measure.js';
	import { fmt, layoutWidth } from '#figures/primitives/scales.js';
	import DepthAxis, { depthAxisHeight } from '../primitives/DepthAxis.svelte';
	import { depthLabel, depthWords, depthScale } from '../primitives/jdata.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const sh = $derived(getContext('artifacts')()?.jspace?.shakespeare ?? null);
	const interactive = $derived(mode !== 'print');
	let width = $state(868);
	const W = $derived(layoutWidth(width, mode, 330));
	const narrow = $derived(W < 600);
	// svelte-ignore state_referenced_locally
	let sel = $state(attrs.depth != null ? Math.round(2 * +attrs.depth) : -1);

	const depths = $derived(sh?.depth ?? []);
	const kinds = $derived(sh?.depth_kind ?? []);
	const m = $derived(sh ? sh.lenses.metrics.map((r) => r.logit_lens) : []);
	const V = $derived(sh?.model.config.V ?? 1);
	const uniform = $derived(1 / V);
	const own = $derived(m.at(-1)?.top1_accuracy ?? 0);
	const pct = (x) => `${Math.round(100 * x)}%`;

	// geometry
	const padL = $derived(narrow ? 34 : 64);
	const padR = $derived(narrow ? 12 : 168);
	const plotT = 18;
	const plotH = $derived(narrow ? 190 : 250);
	const yAxis = $derived(plotT + plotH + 10);
	const H = $derived(yAxis + depthAxisHeight() + 6);
	const xOf = $derived(depthScale(depths.length, padL + 10, W - padR));
	const yOf = (f) => plotT + plotH * (1 - f);
	const line = (vals) => vals.map((v, j) => `${j ? 'L' : 'M'}${xOf(j)} ${yOf(v)}`).join(' ');
	const ticks = [0, 0.25, 0.5, 0.75, 1];

	const words = $derived.by(() => {
		if (!sh) return null;
		const j = sel >= 0 ? sel : depths.length - 1, r = m[j];
		return {
			who: `Depth ${depthLabel(depths[j])}, ${depthWords(depths[j], kinds[j])}.`,
			text: `Over ${r.positions.toLocaleString('en-GB')} held-out positions the logit lens names the model's own top character at ${pct(r.model_agreement)} of them and the character that actually comes next at ${pct(r.top1_accuracy)} (among its top three: ${pct(r.top3_accuracy)}). The relative entropy of its distribution from the model's is ${fmt(r.kl_model_lens, 2)} nats on average.`
		};
	});
</script>

<div class="fx fx-lld" use:measure={(w) => (width = w)}>
	{#if !sh}
		<p class="fig__pending">The J-lens export (jspace/artifacts/shakespeare-jlens.json) is missing.</p>
	{:else}
		<svg viewBox="0 0 {W} {H}" width={W} height={H} role={interactive ? 'group' : 'img'}
			aria-label="The logit lens of the Shakespeare model by depth: agreement with the model {m.map((r) => pct(r.model_agreement)).join(', ')}; accuracy {m.map((r) => pct(r.top1_accuracy)).join(', ')} at depths {depths.map(depthLabel).join(', ')}.">
			<!-- the frame: fractions -->
			{#each ticks as f (f)}
				<line class="rule" x1={padL} x2={W - padR + 8} y1={yOf(f)} y2={yOf(f)} />
				<text class="lbl lbl--micro" x={padL - 6} y={yOf(f) + 3.5} text-anchor="end">{f === 0 ? '0' : f === 1 ? '1' : fmt(f, 2).replace(/^0/, '')}</text>
			{/each}
			<line class="fx-lld__ref" x1={padL} x2={W - padR + 8} y1={yOf(uniform)} y2={yOf(uniform)} />
			<text class="lbl lbl--micro" x={padL + 4} y={yOf(uniform) - 4}>uniform guessing, 1/{V}</text>
			<line class="fx-lld__ref fx-lld__ref--own" x1={padL} x2={W - padR + 8} y1={yOf(own)} y2={yOf(own)} />
			<text class="lbl lbl--micro" x={padL + 4} y={yOf(own) - 4}>the model’s own accuracy, {pct(own)}</text>

			{#if sel >= 0}<line class="leader" x1={xOf(sel)} x2={xOf(sel)} y1={plotT - 6} y2={yAxis} />{/if}

			<path class="fx-lld__agree" d={line(m.map((r) => r.model_agreement))} />
			<path class="fx-lld__acc" d={line(m.map((r) => r.top1_accuracy))} />
			{#each m as r, j (j)}
				<circle class="dot--value" cx={xOf(j)} cy={yOf(r.model_agreement)} r={sel === j ? 4.6 : 3.6} />
				<circle class="dot--open fx-lld__ring" cx={xOf(j)} cy={yOf(r.top1_accuracy)} r={sel === j ? 4.8 : 3.8} />
			{/each}
			{#if !narrow}
				{@const L = m.length - 1}
				<text class="lbl lbl--value" x={xOf(L) + 10} y={yOf(m[L].model_agreement) + 4}>agrees with the model</text>
				<text class="lbl" x={xOf(L) + 10} y={yOf(m[L].top1_accuracy) + 16}>names the next character</text>
			{:else}
				<text class="lbl lbl--micro lbl--value" x={xOf(0)} y={yOf(0.92)}>● agrees with the model</text>
				<text class="lbl lbl--micro" x={xOf(0)} y={yOf(0.92) + 14}>○ names the next character</text>
			{/if}

			<DepthAxis {depths} {kinds} x={xOf(0)} width={xOf(depths.length - 1) - xOf(0)} y={yAxis} sel={sel} onpick={interactive ? (j) => (sel = j) : null} />
			{#if interactive}
				{#each depths as _, j (j)}
					{@const step = xOf(1) - xOf(0)}
					<rect class="hit" x={xOf(j) - step / 2} y={plotT - 6} width={step} height={plotH + 14} role="button" tabindex="-1" aria-label="depth {depthLabel(depths[j])}" onpointerenter={() => (sel = j)} onclick={() => (sel = j)} onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); sel = j; } }} />
				{/each}
			{/if}
		</svg>
		{#if interactive}
			<p class="readout" aria-live="polite">
				<span class="readout__who">{words.who}</span>
				<span class="readout__part">{words.text}</span>
				<span class="readout__hint">Point at a depth.</span>
			</p>
		{/if}
	{/if}
</div>

<style>
	.fx-lld__agree { fill: none; stroke: var(--value); stroke-width: var(--line); }
	.fx-lld__acc { fill: none; stroke: var(--value); stroke-width: var(--hair); }
	.fx-lld__ring { stroke-width: 1.2px; }
	.fx-lld__ref { stroke: var(--ink-3); stroke-width: var(--hair); stroke-dasharray: 2 3; }
	.fx-lld__ref--own { stroke-dasharray: 6 3; }
</style>

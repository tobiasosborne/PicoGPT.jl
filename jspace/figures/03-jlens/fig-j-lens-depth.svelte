<script module>
	export const meta = { size: 'wide', title: 'The two lenses against the model and the text, depth by depth' };
</script>

<script>
	// Figure: how do the two lenses compare with the model and with the text, depth by depth?
	// The Shakespeare model at its nine depths, over the held-out positions of the export.
	// Three panels on one depth axis each: how often the lens's top character is the model's
	// own top character; how often it is the character that actually comes next; and the
	// relative entropy KL(model ‖ lens) in nats. J-lens: filled dots on a line; logit lens:
	// open dots on a hairline (both value ink: readouts are computed going forward). Crosses:
	// the J-lens with the pair-weighted average of the paper's Methods, J/((T+1)/2), which
	// LayerNorm's ε does not let cancel; at the last depth it is the only one off the model.
	// References: the model itself (agreement 1, KL 0), its own accuracy, and uniform
	// guessing 1/V. Data: jspace.shakespeare.lenses.metrics[j].{j_lens, logit_lens} and
	// lenses.literal.{metrics[j].j_lens, final_max_probability_error}.
	// Interactive: point at a depth. Print: the same.
	import { getContext } from 'svelte';
	import { measure } from '#figures/primitives/measure.js';
	import { fmt, layoutWidth } from '#figures/primitives/scales.js';
	import DepthAxis, { depthAxisHeight } from '../primitives/DepthAxis.svelte';
	import { depthLabel, depthWords, depthScale } from '../primitives/jdata.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const sh = $derived(getContext('artifacts')()?.jspace?.shakespeare ?? null);
	const interactive = $derived(mode !== 'print');
	let sel = $state(-1);
	let width = $state(868);
	const W = $derived(layoutWidth(width, mode, 330));
	const narrow = $derived(W < 700);

	const depths = $derived(sh?.depth ?? []);
	const kinds = $derived(sh?.depth_kind ?? []);
	const M = $derived(sh?.lenses.metrics ?? []);
	const lit = $derived(sh?.lenses.literal ?? null);
	const V = $derived(sh?.model.config.V ?? 1);
	const panels = $derived(sh ? [
		{ key: 'model_agreement', title: 'agrees with the model', max: 1, fmtv: (v) => `${fmt(100 * v, 0)}%`, refs: [{ v: 1, label: 'the model' }] },
		{ key: 'top1_accuracy', title: 'is the next character', max: 1, fmtv: (v) => `${fmt(100 * v, 0)}%`, refs: [{ v: M.at(-1).logit_lens.top1_accuracy, label: 'the model' }, { v: 1 / V, label: 'uniform guessing' }] },
		{ key: 'kl_model_lens', title: 'KL(model ‖ lens), nats', max: Math.ceil(Math.max(...M.flatMap((m) => [m.j_lens.kl_model_lens, m.logit_lens.kl_model_lens]))), fmtv: (v) => fmt(v, 2), refs: [] }
	] : []);

	// layout: three panels in a row, or stacked
	const cols = $derived(narrow ? 1 : 3);
	const gap = 34;
	const padL = 30;
	const panelW = $derived((W - padL - gap * (cols - 1)) / cols);
	const plotH = $derived(narrow ? 150 : 180);
	const panelH = $derived(26 + plotH + 12 + depthAxisHeight({ words: false, blocks: false }) + 16);
	const px = (i) => padL + (narrow ? 0 : i * (panelW + gap));
	const py = (i) => (narrow ? i * panelH : 0);
	const H = $derived(narrow ? panels.length * panelH : panelH);
	const xOf = (i) => depthScale(depths.length, px(i) + 8, px(i) + panelW - 8);
	const yOf = (i, P, v) => py(i) + 26 + plotH * (1 - v / P.max);
	const line = (i, P, vals) => vals.map((v, j) => `${j ? 'L' : 'M'}${xOf(i)(j).toFixed(1)} ${yOf(i, P, v).toFixed(1)}`).join(' ');

	const pct = (v) => `${fmt(100 * v, 0)}%`;
	const readout = $derived.by(() => {
		if (!sh) return null;
		const j = sel >= 0 ? sel : Math.max(0, depths.length - 3), m = M[j], l = lit?.metrics?.[j]?.j_lens;
		return {
			who: `Depth ${depthLabel(depths[j])}, ${depthWords(depths[j], kinds[j])}:`,
			body: `the J-lens agrees with the model at ${pct(m.j_lens.model_agreement)} of ${m.j_lens.positions.toLocaleString('en-GB')} positions, the logit lens at ${pct(m.logit_lens.model_agreement)}; they name the next character at ${pct(m.j_lens.top1_accuracy)} and ${pct(m.logit_lens.top1_accuracy)}; KL ${fmt(m.j_lens.kl_model_lens)} and ${fmt(m.logit_lens.kl_model_lens)} nats.` +
				(l ? ` The pair-weighted J-lens: ${pct(l.model_agreement)} agreement, KL ${fmt(l.kl_model_lens, 3)} nats.` : '')
		};
	});
	const aria = $derived(sh ? `J-lens and logit lens of the Shakespeare model by depth: agreement with the model ${M.map((m) => `${pct(m.j_lens.model_agreement)}/${pct(m.logit_lens.model_agreement)}`).join(', ')}.` : '');
</script>

<div class="fx fx-ld" use:measure={(w) => (width = w)}>
	{#if sh}
		<svg viewBox="0 0 {W} {H}" width={W} height={H} role={interactive ? 'group' : 'img'} aria-label={aria}>
			{#each panels as P, i (P.key)}
				{@const X = xOf(i)}
				{@const jv = M.map((m) => m.j_lens[P.key])}
				{@const lv = M.map((m) => m.logit_lens[P.key])}
				{@const tv = lit?.metrics ? lit.metrics.map((m) => m.j_lens[P.key]) : null}
				<g class="fx-ld__panel">
					<text class="lbl lbl--head" x={px(i)} y={py(i) + 12}>{P.title}</text>
					{#each [0, P.max / 2, P.max] as v (v)}
						<line class="rule" x1={px(i)} x2={px(i) + panelW} y1={yOf(i, P, v)} y2={yOf(i, P, v)} />
						<text class="lbl lbl--micro" x={px(i) - 5} y={yOf(i, P, v) + 3.5} text-anchor="end">{P.max === 1 ? (v === 0.5 ? '½' : v) : fmt(v, v % 1 ? 1 : 0)}</text>
					{/each}
					{#each P.refs as ref (ref.label)}
						<line class="fx-ld__ref" x1={px(i)} x2={px(i) + panelW} y1={yOf(i, P, ref.v)} y2={yOf(i, P, ref.v)} />
						<text class="lbl lbl--micro" x={px(i) + panelW} y={yOf(i, P, ref.v) + (ref.v === 1 ? 12 : -4)} text-anchor="end">{ref.label}</text>
					{/each}
					{#if sel >= 0}<line class="fx-ld__sel" x1={X(sel)} x2={X(sel)} y1={yOf(i, P, P.max)} y2={yOf(i, P, 0)} />{/if}
					<path class="fx-ld__logit" d={line(i, P, lv)} />
					<path class="fx-ld__j" d={line(i, P, jv)} />
					{#each lv as v, j (j)}<circle class="fx-ld__ring" cx={X(j)} cy={yOf(i, P, v)} r="3.2" />{/each}
					{#each jv as v, j (j)}<circle class="dot--value" cx={X(j)} cy={yOf(i, P, v)} r="3.2" />{/each}
					{#if tv}
						{#each tv as v, j (j)}
							{@const cx = X(j) + 7}
							{@const cy = yOf(i, P, v)}
							<path class="fx-ld__lit" d="M{cx - 2.6} {cy - 2.6} L{cx + 2.6} {cy + 2.6} M{cx - 2.6} {cy + 2.6} L{cx + 2.6} {cy - 2.6}" />
						{/each}
					{/if}
					{#if i === 0}
						{@const jl = Math.max(0, depths.length - 3)}
						<text class="lbl lbl--micro lbl--value" x={X(jl) + 6} y={yOf(i, P, jv[jl]) + 16}>J-lens</text>
						<text class="lbl lbl--micro lbl--value" x={X(jl) - 8} y={yOf(i, P, lv[jl]) - 4} text-anchor="end">logit lens</text>
						{#if tv}
							<text class="lbl lbl--micro" x={X(2)} y={yOf(i, P, 0.62)}>× the pair-weighted J-lens</text>
						{/if}
					{/if}
					<DepthAxis depths={depths} kinds={kinds} x={px(i) + 8} width={panelW - 16} y={yOf(i, P, 0) + 10} sel={sel} words={false} blocks={false} title="" onpick={interactive ? (j) => (sel = j) : null} />
				</g>
			{/each}
		</svg>
		{#if interactive && readout}
			<p class="readout" aria-live="polite">
				<span class="readout__who">{readout.who}</span>
				<span class="readout__part">{readout.body}</span>
				<span class="readout__hint">Click a depth on any axis for its numbers.</span>
			</p>
		{/if}
	{:else}
		<p class="fig__pending">The J-lens export (jspace/artifacts/shakespeare-jlens.json) is not available.</p>
	{/if}
</div>

<style>
	.fx-ld__j { fill: none; stroke: var(--value); stroke-width: var(--line); }
	.fx-ld__logit { fill: none; stroke: var(--value); stroke-width: var(--hair); }
	.fx-ld__ring { fill: var(--paper); stroke: var(--value); stroke-width: 1px; }
	.fx-ld__lit { stroke: var(--value-deep); stroke-width: 1px; fill: none; }
	.fx-ld__ref { stroke: var(--ink-3); stroke-width: var(--hair); stroke-dasharray: 3 3; }
	.fx-ld__sel { stroke: var(--rule); stroke-width: 1px; }
</style>

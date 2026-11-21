<script module>
	export const meta = { size: 'wide', title: 'One swap in the Shakespeare model' };
</script>

<script>
	// Figure: one lens-coordinate swap in the Tiny Shakespeare model. The pair (s, u) and the
	// depth were chosen by the analysis: every one of the C(65, 2) pairs was tried at α = 1,
	// at every position of the prompt, and the pair whose swap changed the final top character
	// and moved the final distribution furthest (total variation) was kept. The small multiples
	// show the model's distribution for the character after the prompt at α = 0 … 2 of
	// x′ = x + αV(σ(c) − c): the characters that reach the top four at any α, the rest summed.
	// Numbers: J-lens export (Shakespeare) interventions.swap, lenses.prompt.text, vocab.
	import { getContext } from 'svelte';
	import { measure } from '#figures/primitives/measure.js';
	import { fmt, fmtProb, fmtDense, layoutWidth } from '#figures/primitives/scales.js';
	import { at, depthLabel, tv } from './jdata.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const S = $derived(getContext('artifacts')()?.jspace?.shakespeare ?? null);
	const interactive = $derived(mode !== 'print');
	let width = $state(868);
	let focus = $state(-1);

	const glyph = (ch) => (ch === ' ' ? '␣' : ch === '\n' ? '↵' : ch);
	const sw = $derived(S?.interventions.swap ?? null);
	const V = $derived(S?.vocab.length ?? 0);
	const dists = $derived(sw ? sw.alphas.map((_, k) => Array.from({ length: V }, (_, c) => at(sw.probabilities, c, k))) : []);
	const pair = $derived(sw ? sw.pair.map((i) => i - 1) : []);
	// the characters shown: every character in the top four at some α, plus the pair
	const shown = $derived.by(() => {
		if (!sw) return [];
		const set = new Set();
		for (const p of dists) p.map((x, c) => [x, c]).sort((x, y) => y[0] - x[0]).slice(0, 4).forEach(([, c]) => set.add(c));
		pair.forEach((c) => set.add(c));
		const firstTop = (c) => { const k = dists.findIndex((p) => p.map((x, i) => [x, i]).sort((x, y) => y[0] - x[0]).slice(0, 4).some(([, i]) => i === c)); return k < 0 ? 99 : k; };
		return [...set].sort((x, y) => firstTop(x) - firstTop(y) || dists[0][y] - dists[0][x]);
	});
	const rest = (p) => 1 - shown.reduce((t, c) => t + p[c], 0);
	const text = $derived(S?.lenses.prompt.text ?? '');
	const lastLine = $derived(text.slice(text.slice(0, -1).lastIndexOf('\n') + 1));

	// geometry: the multiples side by side, or stacked on a narrow screen
	const W = $derived(layoutWidth(width, mode, 340));
	const narrow = $derived(W < 620);
	const K = $derived((sw?.alphas.length ?? 1));
	const gap = $derived(narrow ? 0 : 18);
	const mw = $derived(narrow ? W - 8 : (W - 8 - gap * (K - 1)) / K);
	const xm = (k) => (narrow ? 4 : 4 + k * (mw + gap));
	const nb = $derived(shown.length + 1);
	const bw = $derived((mw * 0.96) / nb);
	const hBar = $derived(narrow ? 58 : 96);
	const yText = 20;
	const top = $derived(yText + (narrow ? 50 : 64));
	const step = $derived(hBar + 92);
	const base = (k) => top + 22 + hBar + (narrow ? k * step : 0);
	const H = $derived(base(narrow ? K - 1 : 0) + 44);
	// label a bar if it is the top of its multiple, or one of the pair and visible
	const labelled = (p, c) => p[c] >= 0.05 && (c === p.indexOf(Math.max(...p)) || pair.includes(c));

	const readout = $derived.by(() => {
		if (!sw) return null;
		const k = focus >= 0 ? focus : sw.alphas.indexOf(1);
		const p = dists[k];
		const top = p.map((x, c) => [x, c]).sort((x, y) => y[0] - x[0]).slice(0, 3);
		return `α = ${fmt(sw.alphas[k], 2)}: ${top.map(([x, c]) => `‘${glyph(S.vocab[c])}’ ${fmtProb(x, 2, true)}`).join(', ')}; total variation from the unedited model ${fmt(tv(p, sw.baseline), 3)}.`;
	});
</script>

<div class="fx fx-gremio" use:measure={(w) => (width = w)}>
	{#if !sw}
		<p class="fig__pending">The Shakespeare J-lens export (jspace/artifacts/shakespeare-jlens.json) is missing.</p>
	{:else}
		<svg viewBox="0 0 {W} {H}" width={W} height={H} role={interactive ? 'group' : 'img'} aria-label="Swapping ‘{glyph(S.vocab[pair[0]])}’ and ‘{glyph(S.vocab[pair[1]])}’ at depth {depthLabel(sw.depth)} at every position of the prompt: {sw.alphas.map((al, k) => `α ${fmt(al, 2)}: top ‘${glyph(S.vocab[dists[k].indexOf(Math.max(...dists[k]))])}’`).join('; ')}">
			<text class="lbl" x={4} y={yText - 4}>swap <tspan class="gr__tok">‘{glyph(S.vocab[pair[0]])}’</tspan> and <tspan class="gr__tok">‘{glyph(S.vocab[pair[1]])}’</tspan> at depth {depthLabel(sw.depth)}, {narrow ? 'every column; the end of the prompt:' : 'in every column of the prompt:'}</text>
			<text class="gr__prompt" x={4} y={yText + (narrow ? 20 : 22)}>{#each Array.from(narrow ? lastLine : text) as ch, i (i)}<tspan class:gr__last={i === (narrow ? lastLine : text).length - 1}>{glyph(ch)}</tspan>{/each}</text>
			{#each sw.alphas as al, k (k)}
				{@const p = dists[k]}
				{@const b0 = base(k)}
				<text class="lbl lbl--head" x={xm(k)} y={b0 - hBar - 22}>α = {fmt(al, al % 1 ? 2 : 0)}{al === 0 ? ': the model' : ''}</text>
				<line class="axis" x1={xm(k)} x2={xm(k) + mw} y1={b0} y2={b0} />
				{#each shown as c, i (c)}
					<rect class="bar--value" x={xm(k) + i * bw + bw * 0.14} y={b0 - p[c] * hBar} width={bw * 0.72} height={p[c] * hBar} />
					{#if labelled(p, c)}<text class="lbl lbl--micro lbl--strong" x={xm(k) + i * bw + bw / 2} y={b0 - p[c] * hBar - 4} text-anchor="middle">{fmtDense(p[c], 2)}</text>{/if}
					<text class="tok gr__ch" class:is-pair={pair.includes(c)} x={xm(k) + i * bw + bw / 2} y={b0 + 15} text-anchor="middle">{glyph(S.vocab[c])}</text>
				{/each}
				<rect class="gr__rest" x={xm(k) + shown.length * bw + bw * 0.14} y={b0 - rest(p) * hBar} width={bw * 0.72} height={Math.max(0, rest(p) * hBar)} />
				<text class="tok gr__ch" x={xm(k) + shown.length * bw + bw / 2} y={b0 + 15} text-anchor="middle">…</text>
				<text class="lbl lbl--micro" x={xm(k)} y={b0 + 32}>TV {fmt(tv(p, sw.baseline), 3)}</text>
				{#if interactive}
					<rect class="hit" x={xm(k)} y={b0 - hBar - 34} width={mw} height={hBar + 70} role="button" tabindex="0" aria-label="α = {al}"
						onpointerenter={() => (focus = k)} onfocus={() => (focus = k)} onpointerleave={() => (focus = -1)} onblur={() => (focus = -1)} onkeydown={() => {}} />
				{/if}
			{/each}
		</svg>
		{#if interactive && readout}
			<p class="readout" aria-live="polite"><span class="readout__part">{readout}</span><span class="readout__hint">Point at a panel to read it.</span></p>
		{/if}
	{/if}
</div>

<style>
	.gr__rest { fill: none; stroke: var(--ink-3); stroke-width: var(--hair); stroke-dasharray: 2 2; }
	.gr__ch { font-size: 13px; fill: var(--ink-2); }
	.gr__ch.is-pair { fill: var(--ink); font-weight: 700; text-decoration: underline; }
	.gr__tok { font-family: var(--serif); font-size: 14px; fill: var(--ink); }
	.gr__prompt { font-family: var(--serif); font-size: 13px; fill: var(--ink-2); letter-spacing: 0.01em; }
	.gr__last { fill: var(--ink); font-weight: 700; }
</style>

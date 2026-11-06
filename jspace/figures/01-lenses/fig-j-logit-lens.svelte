<script module>
	export const meta = { size: 'wide', title: 'The logit lens on one window of the specimen' };
</script>

<script>
	// Figure: the logit lens on one window of the specimen. Small multiples: rows = depths
	// 0, ½, 1; columns = positions. In each cell the V probabilities of softmax(Wout LN(x) +
	// bout) applied to the column x at that depth, as bars on one scale (0…1). The most
	// probable character stands over its bar: in ink where it is also the model's choice at
	// that position, in grey where it is not. The character that actually comes next is
	// underlined. At depth 1 = L the logit lens is the model's own readout (checked below).
	// Per row: the number of positions at which the lens agrees with the model.
	// Data: jspace.specimen.lenses.{logit_probs, model_probs} (column n = t + (b−1)T).
	// Interactive: drag the window, click a cell. Print: window 1.
	import { getContext } from 'svelte';
	import { measure } from '#figures/primitives/measure.js';
	import CorpusStrip from '#figures/primitives/CorpusStrip.svelte';
	import { fmt, fmtProb, showChar, clamp, layoutWidth } from '#figures/primitives/scales.js';
	import DistBars, { distBarsWidth } from '../primitives/DistBars.svelte';
	import { at, fibre, argmax, depthLabel, depthWords, tv } from '../primitives/jdata.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const js = $derived(getContext('artifacts')()?.jspace?.specimen ?? null);
	const T = $derived(specimen.config.T);
	const V = $derived(specimen.config.V);
	const interactive = $derived(mode !== 'print');

	// svelte-ignore state_referenced_locally
	let start = $state(clamp((+attrs.window || 1) - 1, 0, 9));
	// svelte-ignore state_referenced_locally
	let pos = $state(clamp((+attrs.column || 6) - 1, 0, 7));
	// svelte-ignore state_referenced_locally
	let row = $state(clamp(Math.round(2 * (+attrs.depth || 0)), 0, 2));
	let width = $state(868);
	const W = $derived(layoutWidth(width, mode, 330));
	const narrow = $derived(W < 600);

	const depths = $derived(js?.depth ?? []);
	const kinds = $derived(js?.depth_kind ?? []);
	const N = $derived(js ? js.lenses.model_probs.shape[1] : 0);
	const chars = $derived(Array.from(specimen.corpus.slice(start, start + T)));
	const cellP = (n, j) => fibre(js.lenses.logit_probs, n, j);
	const modelP = (n) => fibre(js.lenses.model_probs, n);
	const next = (t) => specimen.vocab.indexOf(specimen.corpus[start + t + 1]);
	const grid = $derived(js ? depths.map((_, j) => Array.from({ length: T }, (_, t) => {
		const n = start * T + t, p = cellP(n, j), m = modelP(n);
		return { p, top: argmax(p), agree: argmax(p) === argmax(m), tv: tv(p, m) };
	})) : []);
	// over all N columns of the export: agreement with the model, per depth
	const agreeAll = $derived(js ? depths.map((_, j) => {
		let k = 0;
		for (let n = 0; n < N; n++) k += argmax(cellP(n, j)) === argmax(modelP(n));
		return k;
	}) : []);
	// at the last depth the logit lens IS the model's readout
	const lastErr = $derived(js ? (() => { let e = 0; const j = depths.length - 1; for (let n = 0; n < N; n++) for (let a = 0; a < V; a++) e = Math.max(e, Math.abs(at(js.lenses.logit_probs, a, n, j) - at(js.lenses.model_probs, a, n))); return e; })() : 1);

	// geometry
	const padL = $derived(narrow ? 30 : 150);
	const padR = $derived(narrow ? 6 : 84);
	const colW = $derived((W - padL - padR) / T);
	const cx = (t) => padL + colW * (t + 0.5);
	const gap = $derived(narrow ? 1 : 2);
	const bw = $derived(Math.max(2.5, (colW * 0.8 - (V - 1) * gap) / V));
	const bW = $derived(distBarsWidth(V, bw, gap));
	const barH = $derived(narrow ? 28 : 40);
	const showChars = $derived(!narrow && bw >= 6);
	const rowH = $derived(barH + (showChars ? 44 : 30));
	const top = $derived(84);
	const base = (j) => top + 22 + barH + j * rowH;
	const H = $derived(base(depths.length - 1) + (showChars ? 24 : 12));
	const stripW = $derived(Math.min(W - 24, specimen.corpus.length * (narrow ? 16 : 19)));

	const words = $derived.by(() => {
		if (!js) return null;
		const c = grid[row][pos], n = start * T + pos, m = modelP(n), mt = argmax(m);
		const top3 = (p) => p.map((v, a) => ({ v, a })).sort((u, w) => w.v - u.v).slice(0, 3).map((o) => `‘${showChar(specimen.vocab[o.a])}’ ${fmtProb(o.v, 2, true)}`).join(', ');
		return {
			who: `Depth ${depthLabel(depths[row])} (${depthWords(depths[row], kinds[row])}), position ${pos + 1} of “${specimen.corpus.slice(start, start + T)}”, the column of ‘${showChar(chars[pos])}’.`,
			lens: `The logit lens reads ${top3(c.p)}; the model reads ${top3(m)}. ${c.agree ? 'They agree' : 'They disagree'} on the top character, and their total variation distance is ${fmt(c.tv, 2)}.`,
			all: `Over all ${N} columns of the ten windows, the logit lens agrees with the model at ${agreeAll.map((k, j) => `${k} at depth ${depthLabel(depths[j])}`).join(', ')}.`
		};
	});
	const pick = (j, t) => { if (interactive) { row = j; pos = t; } };
</script>

<div class="fx fx-logit" use:measure={(w) => (width = w)}>
	{#if !js}
		<p class="fig__pending">The J-lens export (jspace/artifacts/specimen-jlens.json) is missing.</p>
	{:else}
		<svg viewBox="0 0 {W} {H}" width={W} height={H} role={interactive ? 'group' : 'img'}
			aria-label="The logit lens on the window “{specimen.corpus.slice(start, start + T)}” at depths {depths.map(depthLabel).join(', ')}. It agrees with the model's top character at {grid.map((r) => r.filter((c) => c.agree).length).join(', ')} of {T} positions.">
			<CorpusStrip corpus={specimen.corpus} {T} bind:start bind:pos x={(W - stripW) / 2} y={24} width={stripW} showTarget {interactive} label="window and column" />
			{#each chars as ch, t (t)}
				<text class="tok fx-logit__ch" class:tok--reader={interactive && t === pos} x={cx(t)} y={top} text-anchor="middle">{showChar(ch)}</text>
			{/each}
			{#each grid as cells, j (j)}
				{@const y = base(j)}
				<text class="lbl fx-logit__depth" x={narrow ? 2 : padL - 16} y={y - barH / 2 - (narrow ? 0 : 6)} text-anchor={narrow ? 'start' : 'end'}>{narrow ? depthLabel(depths[j]) : `depth ${depthLabel(depths[j])}`}</text>
				{#if !narrow}
					<text class="lbl lbl--micro" x={padL - 16} y={y - barH / 2 + 8} text-anchor="end">{depthWords(depths[j], kinds[j])}</text>
					{#if j === grid.length - 1 && lastErr < 1e-4}
						<text class="lbl lbl--micro" x={padL - 16} y={y - barH / 2 + 21} text-anchor="end">= the model’s readout</text>
					{/if}
				{/if}
				{#each cells as c, t (t)}
					{#if interactive && j === row && t === pos}
						<rect class="fx-logit__sel" x={cx(t) - colW / 2 + 2} y={y - barH - 20} width={colW - 4} height={barH + (showChars ? 40 : 28)} />
					{/if}
					<DistBars values={c.p} x={cx(t) - bW / 2} {y} {bw} {gap} h={barH} vocab={specimen.vocab} chars={showChars ? specimen.vocab : null} target={next(t)} markMuted={!c.agree} charSize={Math.min(11, bw + 2)} />
					{#if interactive}
						<rect class="hit" x={cx(t) - colW / 2} y={y - barH - 20} width={colW} height={rowH} role="button" tabindex="0" aria-label="depth {depthLabel(depths[j])}, position {t + 1}; details in the live readout"
							onclick={() => pick(j, t)} onpointerenter={() => pick(j, t)} onfocus={() => pick(j, t)} onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(j, t); } }} />
					{/if}
				{/each}
				{#if !narrow}
					<text class="lbl" x={W - padR + 12} y={y - barH / 2 + 4}><tspan class="lbl--strong">{cells.filter((c) => c.agree).length}</tspan> of {T}</text>
					{#if j === 0}<text class="lbl lbl--micro" x={W - padR + 12} y={y - barH / 2 - 12}>agree</text>{/if}
				{/if}
			{/each}
		</svg>
		{#if interactive}
			<p class="readout" aria-live="polite">
				<span class="readout__who">{words.who}</span>
				<span class="readout__part">{words.lens}</span>
				<span class="readout__part">{words.all}</span>
				<span class="readout__hint">Point at a cell; drag the window along the line.</span>
			</p>
		{/if}
	{/if}
</div>

<style>
	.fx-logit__ch { font-size: 17px; fill: var(--ink-2); }
	.fx-logit__depth { fill: var(--ink); }
	.fx-logit__sel { fill: var(--value-tint); opacity: 0.55; }
</style>

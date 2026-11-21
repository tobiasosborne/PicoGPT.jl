<script module>
	export const meta = { size: 'wide', title: 'The stream of one window at every depth' };
</script>

<script>
	// Figure: the residual stream of one window of the specimen at its three depths, as a
	// table of spoke glyphs: rows = depths (0 after the embedding, ½ after attention, 1 after
	// the MLP), columns = positions, as X_ℓ ∈ ℝ^{D×T} is written. One spoke scale for every
	// glyph of every window, so lengths compare across depths, positions and windows. Under
	// the last row, the readout of each final column: the model's distribution over the next
	// character. Data: specimen.windowTrace(start) (trained θ): X0, Xmid, Xout, probs.
	// Interactive: drag the window, click a column (or ←/→ on the strip). Print: window 1.
	import { getContext } from 'svelte';
	import { measure } from '#figures/primitives/measure.js';
	import SpokeGlyph from '#figures/primitives/SpokeGlyph.svelte';
	import CorpusStrip from '#figures/primitives/CorpusStrip.svelte';
	import { fmt, fmtProb, showChar, clamp, maxAbs, spokeScale, layoutWidth } from '#figures/primitives/scales.js';
	import DistBars, { distBarsWidth } from '../primitives/DistBars.svelte';
	import { argmax } from '../primitives/jdata.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const T = $derived(specimen.config.T);
	const V = $derived(specimen.config.V);
	const interactive = $derived(mode !== 'print');

	// svelte-ignore state_referenced_locally
	let start = $state(clamp((+attrs.window || 1) - 1, 0, 9));
	// svelte-ignore state_referenced_locally
	let pos = $state(clamp((+attrs.column || 4) - 1, 0, 7));
	let width = $state(868);
	const W = $derived(layoutWidth(width, mode, 330));
	const narrow = $derived(W < 600);

	const tr = $derived(specimen.windowTrace(start));
	const rows = $derived([tr.X0, tr.Xmid, tr.Xout]);
	const names = [
		{ sub: '0', word: 'after the embedding' },
		{ sub: '½', word: 'after attention' },
		{ sub: '1', word: 'after the MLP' }
	];
	// one scale for every column of every window at every depth
	const all = $derived(specimen.windows.map((s) => { const w = specimen.windowTrace(s); return [w.X0, w.Xmid, w.Xout]; }));
	const vmax = $derived(maxAbs(all));
	const norm = (v) => Math.hypot(...v);

	// geometry
	const padL = $derived(narrow ? 30 : 150);
	const colW = $derived((W - padL - 6) / T);
	const rG = $derived(Math.min(colW * 0.4, 30));
	const sc = $derived(spokeScale(rG, vmax));
	const cx = (t) => padL + colW * (t + 0.5);
	const top = $derived(78 + rG);
	const rowH = $derived(2 * rG + (narrow ? 18 : 26));
	const ry = (j) => top + j * rowH;
	const gap = $derived(narrow ? 1 : 2);
	const bw = $derived(Math.max(2.5, (colW * 0.8 - (V - 1) * gap) / V));
	const bW = $derived(distBarsWidth(V, bw, gap));
	const barH = $derived(narrow ? 26 : 34);
	const yBars = $derived(ry(2) + rG + 26 + barH);
	const H = $derived(yBars + (narrow ? 14 : 26));
	const stripW = $derived(Math.min(W - 24, specimen.corpus.length * (narrow ? 16 : 19)));
	const next = (t) => specimen.vocab.indexOf(specimen.corpus[start + t + 1]);

	const words = $derived.by(() => {
		const [a, m, o] = [tr.X0[pos], tr.Xmid[pos], tr.Xout[pos]];
		const p = tr.probs[pos], k = argmax(p);
		const da = a.map((v, i) => m[i] - v), dm = m.map((v, i) => o[i] - v);
		return {
			who: `‘${showChar(tr.chars[pos])}’, position ${pos + 1} of “${tr.text}”.`,
			flow: `Its length is ${fmt(norm(a))} at depth 0, ${fmt(norm(m))} at depth ½ and ${fmt(norm(o))} at depth 1; attention added a vector of length ${fmt(norm(da))}, the MLP one of length ${fmt(norm(dm))}.`,
			out: `The readout of the last column predicts ‘${showChar(specimen.vocab[k])}’ (${fmtProb(p[k], 2, true)})${next(pos) >= 0 ? `; the text continues with ‘${showChar(specimen.vocab[next(pos)])}’` : ''}.`
		};
	});
</script>

<div class="fx fx-depths" use:measure={(w) => (width = w)}>
	<svg viewBox="0 0 {W} {H}" width={W} height={H} role={interactive ? 'group' : 'img'}
		aria-label="The stream of the window “{tr.text}” at depths 0, ½ and 1, one spoke glyph per column, and the model's readout under the last row. {words.flow}">
		<CorpusStrip corpus={specimen.corpus} {T} bind:start bind:pos x={(W - stripW) / 2} y={24} width={stripW} {interactive} label="window and column" />

		{#if interactive}
			<rect class="fx-depths__sel" x={cx(pos) - colW / 2 + 2} y={top - rG - 22} width={colW - 4} height={yBars - top + rG + 30} />
		{/if}
		{#each tr.chars as ch, t (t)}
			<text class="tok fx-depths__ch" class:tok--reader={t === pos} x={cx(t)} y={top - rG - 8} text-anchor="middle">{showChar(ch)}</text>
		{/each}

		{#each rows as X, j (j)}
			<text class="math fx-depths__x" x={narrow ? 4 : padL - 18} y={ry(j) + (narrow ? 4 : -2)} text-anchor={narrow ? 'start' : 'end'}>x<tspan class="fx-depths__sub" dy="4">{names[j].sub}</tspan></text>
			{#if !narrow}<text class="lbl lbl--micro" x={padL - 18} y={ry(j) + 14} text-anchor="end">{names[j].word}</text>{/if}
			{#each X as col, t (t)}
				<SpokeGlyph values={col} x={cx(t)} y={ry(j)} rmax={rG} scale={sc.scale} tip={Math.max(1.4, rG / 12)} />
			{/each}
		{/each}

		<text class="lbl" x={narrow ? 4 : padL - 18} y={yBars - barH / 2} text-anchor={narrow ? 'start' : 'end'}>{narrow ? '' : 'readout'}</text>
		{#each tr.probs as p, t (t)}
			<DistBars values={p} x={cx(t) - bW / 2} y={yBars} {bw} {gap} h={barH} vocab={specimen.vocab} chars={narrow ? null : specimen.vocab} target={next(t)} charSize={Math.min(11, bw + 2)} />
		{/each}

		{#if interactive}
			{#each tr.chars as _, t (t)}
				<rect class="hit" x={cx(t) - colW / 2} y={top - rG - 24} width={colW} height={yBars - top + rG + 30} role="button" tabindex="0" aria-label="column {t + 1}; details in the live readout"
					onclick={() => (pos = t)} onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pos = t; } }} />
			{/each}
		{/if}
	</svg>
	{#if interactive}
		<p class="readout" aria-live="polite">
			<span class="readout__who">{words.who}</span>
			<span class="readout__part">{words.flow}</span>
			<span class="readout__part">{words.out}</span>
			<span class="readout__hint">Click a column; drag the window along the line.</span>
		</p>
	{/if}
</div>

<style>
	.fx-depths__sel { fill: var(--value-tint); opacity: 0.55; }
	.fx-depths__ch { font-size: 17px; fill: var(--ink-2); }
	.fx-depths__x { font-size: 16px; }
	.fx-depths :global(.fx-depths__sub) { font-size: 11px; font-style: normal; }
</style>

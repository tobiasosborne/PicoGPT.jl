<script module>
	export const meta = { size: 'wide', title: 'Both lenses at every depth and position of a window' };
</script>

<script>
	// Figure: what do the two lenses read at each depth and position?
	// One window of the specimen. Columns: its positions (the column is the token), headed by
	// the character there. Rows: the depths 0, ½, 1 (forward runs down), and at the bottom the
	// model's own output. In each cell, the V probabilities in vocabulary order as pairs of
	// bars from a common baseline: the J-lens solid, the logit lens open (both value ink:
	// distributions are computed going forward); the model's row has one bar per character.
	// Under each cell, a tick marks the character that actually comes next.
	// At depth 1 = L both lenses are the model's own readout (J_L = I).
	// Data: jspace.specimen.lenses.{j_probs[a, n, j], logit_probs[a, n, j], model_probs[a, n]},
	// column n = t + (b−1)T, window b starting at corpus position b − 1.
	// Interactive: drag the window; point at a cell for its numbers. Print: the stated window.
	import { getContext } from 'svelte';
	import { measure } from '#figures/primitives/measure.js';
	import CorpusStrip from '#figures/primitives/CorpusStrip.svelte';
	import { fmtProb, showChar, clamp, layoutWidth } from '#figures/primitives/scales.js';
	import { at, argmax, depthLabel } from '../primitives/jdata.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const js = $derived(getContext('artifacts')()?.jspace?.specimen ?? null);
	const T = $derived(specimen.config.T), V = $derived(specimen.config.V);
	const vocab = $derived(specimen.vocab);
	const interactive = $derived(mode !== 'print');

	// svelte-ignore state_referenced_locally
	let start = $state(clamp((+attrs.window || 1) - 1, 0, 9));
	let pos = $state(7);
	let hover = $state(null); // { r, t }
	let width = $state(868);
	const W = $derived(layoutWidth(width, mode, 330));
	const narrow = $derived(W < 640);

	const tr = $derived(specimen.windowTrace(start));
	const depths = $derived(js?.depth ?? []);
	const probs = (t, a, n, j) => (j == null ? at(t, a, n) : at(t, a, n, j));
	const col = (t) => start * T + t;
	const dist = (tensor, t, j) => Array.from({ length: V }, (_, a) => probs(tensor, a, col(t), j));
	// rows: depths (J and logit), then the model
	const rows = $derived(js ? [
		...depths.map((d, j) => ({ label: depthLabel(d), J: Array.from({ length: T }, (_, t) => dist(js.lenses.j_probs, t, j)), L: Array.from({ length: T }, (_, t) => dist(js.lenses.logit_probs, t, j)) })),
		{ label: 'model', J: Array.from({ length: T }, (_, t) => dist(js.lenses.model_probs, t, null)), L: null }
	] : []);
	const next = $derived(Array.from({ length: T }, (_, t) => vocab.indexOf(specimen.corpus[start + t + 1])));

	// layout: one block of T columns, or two blocks of T/2 when narrow
	const perBlock = $derived(narrow ? T / 2 : T);
	const nBlocks = $derived(T / perBlock);
	const padL = $derived(narrow ? 40 : 62);
	const cellW = $derived((W - padL) / perBlock);
	const pg = $derived(Math.max(1.5, cellW * 0.03));
	const bw = $derived(Math.max(1.6, (cellW * 0.84 - 7 - 6 * pg) / 14));
	const pairW = $derived(2 * bw + 1);
	const barsW = $derived(V * pairW + (V - 1) * pg);
	const hB = $derived(narrow ? 28 : 36);
	const rowH = $derived(hB + 14);
	const stripH = 44;
	const headH = 26;
	const blockH = $derived(headH + rows.length * rowH + 22);
	const H = $derived(stripH + nBlocks * blockH + 14);
	const cellX = (t) => padL + (t % perBlock) * cellW + (cellW - barsW) / 2;
	const blockY = (t) => stripH + Math.floor(t / perBlock) * blockH;
	const baseY = (t, r) => blockY(t) + headH + r * rowH + hB + 4;
	const pairX = (t, a) => cellX(t) + a * (pairW + pg);

	const readout = $derived.by(() => {
		if (!rows.length) return null;
		const h = hover ?? { r: 1, t: 3 };
		const row = rows[h.r];
		const top3 = (p) => p.map((v, a) => ({ v, a })).sort((x, y) => y.v - x.v).slice(0, 3).map((o) => `‘${showChar(vocab[o.a])}’ ${fmtProb(o.v, 2, true)}`).join(', ');
		const where = `‘${showChar(tr.chars[h.t])}’ at position ${h.t + 1} of “${tr.text}”`;
		const nx = next[h.t] >= 0 ? `; the next character is ‘${showChar(vocab[next[h.t]])}’` : '';
		if (!row.L) return { who: `${where}, the model:`, body: `${top3(row.J[h.t])}${nx}.` };
		return { who: `${where}, depth ${row.label}:`, body: `the J-lens reads ${top3(row.J[h.t])}; the logit lens ${top3(row.L[h.t])}; the model ${top3(rows.at(-1).J[h.t])}${nx}.` };
	});
	const aria = $derived(rows.length ? `Window “${tr.text}”: ` + rows.map((r) => `${r.label}: J-lens ${r.J.map((p) => showChar(vocab[argmax(p)])).join('')}${r.L ? `, logit lens ${r.L.map((p) => showChar(vocab[argmax(p)])).join('')}` : ''}`).join('; ') : '');
</script>

<div class="fx fx-ro" use:measure={(w) => (width = w)}>
	{#if js}
		<svg viewBox="0 0 {W} {H}" width={W} height={H} role={interactive ? 'group' : 'img'} aria-label="Lens readouts of the specimen. {aria}">
			<CorpusStrip corpus={specimen.corpus} {T} bind:start bind:pos x={padL} y={20} width={Math.min(W - padL - 10, 330)} {interactive} label="the window" />
			{#if !narrow}
				<text class="lbl lbl--micro" x={padL + Math.min(W - padL - 10, 330) + 16} y={18}>bars: J-lens solid, logit lens open</text>
			{/if}
			{#each Array.from({ length: T }, (_, t) => t) as t (t)}
				{@const cx = padL + (t % perBlock) * cellW + cellW / 2}
				<text class="tok fx-ro__head" class:is-on={hover && hover.t === t} x={cx} y={blockY(t) + 18} text-anchor="middle">{showChar(tr.chars[t])}</text>
				<text class="lbl lbl--micro" x={cx + 10} y={blockY(t) + 18}>{t + 1}</text>
				{#each rows as row, r (r)}
					{@const y0 = baseY(t, r)}
					<line class="fx-ro__base" x1={cellX(t) - 2} x2={cellX(t) + barsW + 2} y1={y0} y2={y0} />
					{#each Array.from({ length: V }, (_, a) => a) as a (a)}
						{@const pJ = row.L ? row.J[t][a] : row.J[t][a]}
						{#if row.L}
							<rect class="bar--value" x={pairX(t, a)} y={y0 - pJ * hB} width={bw} height={pJ * hB} />
							<rect class="fx-ro__open" x={pairX(t, a) + bw + 1} y={y0 - row.L[t][a] * hB} width={bw} height={row.L[t][a] * hB} />
						{:else}
							<rect class="bar--value" x={pairX(t, a)} y={y0 - pJ * hB} width={pairW} height={pJ * hB} />
						{/if}
					{/each}
					{#if next[t] >= 0}
						<line class="fx-ro__next" x1={pairX(t, next[t]) - 0.5} x2={pairX(t, next[t]) + pairW + 0.5} y1={y0 + 3} y2={y0 + 3} />
					{/if}
					{#if interactive}
						<rect class="hit" x={padL + (t % perBlock) * cellW} y={y0 - hB - 6} width={cellW} height={rowH} role="button" tabindex="0" aria-label="depth {row.label}, position {t + 1}" onpointerenter={() => (hover = { r, t })} onfocus={() => (hover = { r, t })} />
					{/if}
				{/each}
			{/each}
			{#each Array.from({ length: nBlocks }, (_, b) => b) as b (b)}
				{#each rows as row, r (r)}
					<text class="lbl" class:lbl--head={r === rows.length - 1} x={0} y={baseY(b * perBlock, r) - 4}>{r === rows.length - 1 ? 'model' : narrow ? row.label : `depth ${row.label}`}</text>
				{/each}
				<line class="rule" x1={0} x2={W} y1={baseY(b * perBlock, rows.length - 2) + 7} y2={baseY(b * perBlock, rows.length - 2) + 7} />
				{#each vocab as ch, a (a)}
					<text class="lbl lbl--micro fx-ro__v" x={pairX(b * perBlock, a) + pairW / 2} y={baseY(b * perBlock, rows.length - 1) + 16} text-anchor="middle">{showChar(ch)}</text>
				{/each}
				<text class="lbl lbl--micro" x={W} y={baseY(b * perBlock, rows.length - 1) + 16} text-anchor="end">tick: the next character</text>
			{/each}
		</svg>
		{#if interactive && readout}
			<p class="readout" aria-live="polite">
				<span class="readout__who">{readout.who}</span>
				<span class="readout__part">{readout.body}</span>
				<span class="readout__hint">Drag the window along the text; point at a cell for its numbers.</span>
			</p>
		{/if}
	{:else}
		<p class="fig__pending">The J-lens export (jspace/artifacts/specimen-jlens.json) is not available.</p>
	{/if}
</div>

<style>
	.fx-ro__head { font-size: 17px; fill: var(--ink); }
	.fx-ro__head.is-on { font-weight: 650; }
	.fx-ro__base { stroke: var(--ink-3); stroke-width: var(--hair); }
	.fx-ro__open { fill: var(--paper); stroke: var(--value); stroke-width: 0.8px; }
	.fx-ro__next { stroke: var(--ink); stroke-width: 1.4px; }
	.fx-ro__v { font-family: var(--serif); font-size: 10px; }
</style>

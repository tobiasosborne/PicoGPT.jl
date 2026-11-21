<script module>
	export const meta = { size: 'wide', title: 'Swapping two characters in lens coordinates, here and later' };
</script>

<script>
	// Figure: what does swapping the lens coordinates of two characters s, u in one column do,
	// at that position and at every later position of the window?
	// x′ = x + V(σ(c) − c), V = [v_s v_u], c = V⁺x, applied to one column at depth ℓ, then
	// the forward pass continues. Rows: depths 0, ½, 1. Slots: the window's positions, aligned
	// with its text; the edited position and every later one show the model's distribution
	// before (hairline outlines) and after (solid bars); s and u in full ink. Positions before
	// the edit are left empty: no map reads backwards. Each row states the Gram matrix
	// G = VᵀV and the coordinates c. Numbers: J-lens export interventions.by_depth[j].swap,
	// swap_coordinates, swap_gram; lenses.model_probs.
	import { getContext } from 'svelte';
	import { measure } from '#figures/primitives/measure.js';
	import CorpusStrip from '#figures/primitives/CorpusStrip.svelte';
	import CharRow from './CharRow.svelte';
	import { fmt, fmtProb, showChar, clamp, layoutWidth } from '#figures/primitives/scales.js';
	import { at, tableProbs, depthLabel, tv } from './jdata.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const J = $derived(getContext('artifacts')()?.jspace?.specimen ?? null);
	const interactive = $derived(mode !== 'print');
	let width = $state(868);

	const T = $derived(specimen.config.T);
	const V = $derived(specimen.config.V);
	const vocab = $derived(specimen.vocab);
	const nWin = $derived(J?.sampling.windows ?? 10);

	// svelte-ignore state_referenced_locally
	let start = $state(clamp((+attrs.window || 1) - 1, 0, 9));
	// svelte-ignore state_referenced_locally
	let pos = $state(clamp((+attrs.position || 4) - 1, 0, 7));
	// svelte-ignore state_referenced_locally
	let s = $state(Math.max(0, specimen.vocab.indexOf(attrs.s ?? 'e')));
	// svelte-ignore state_referenced_locally
	let u = $state(Math.max(0, specimen.vocab.indexOf(attrs.u ?? 't')));
	$effect(() => { if (u === s) u = (s + 1) % V; });

	const n = $derived(start * T + pos);
	const pairIndex = $derived(J ? J.interventions.pairs.findIndex(([x, y]) => x === Math.min(s, u) + 1 && y === Math.max(s, u) + 1) : -1);
	const flip = $derived(s > u); // the export orders a pair by character id
	const chars = $derived(Array.from(specimen.corpus.slice(start, start + T)));
	const before = (m) => Array.from({ length: V }, (_, c) => at(J.lenses.model_probs, c, m));
	const argmax = (p) => p.indexOf(Math.max(...p));

	const rows = $derived.by(() => {
		if (!J || pairIndex < 0) return [];
		return J.depth.map((d, j) => {
			const D = J.interventions.by_depth[j];
			const c0 = at(D.swap_coordinates, 0, n, pairIndex), c1 = at(D.swap_coordinates, 1, n, pairIndex);
			const g = (r, k) => at(D.swap_gram, r, k, pairIndex);
			const G = flip ? [[g(1, 1), g(1, 0)], [g(0, 1), g(0, 0)]] : [[g(0, 0), g(0, 1)], [g(1, 0), g(1, 1)]];
			const c = flip ? [c1, c0] : [c0, c1]; // (c_s, c_u)
			const slots = [];
			for (let o = 0; o < T - pos; o++) {
				const after = tableProbs(J, at(D.swap, o, n, pairIndex));
				if (!after) break;
				const prior = before(n + o);
				slots.push({ t: pos + o, prior, after, tv: tv(after, prior), top0: argmax(prior), top1: argmax(after) });
			}
			return { d, j, G, c, slots };
		});
	});

	// geometry: one slot per window position
	const W = $derived(layoutWidth(width, mode, 340));
	const narrow = $derived(W < 620);
	const padL = $derived(narrow ? 8 : 150);
	const padR = 6;
	const sw = $derived((W - padL - padR) / T);
	const sx = (t) => padL + sw * t;
	const bw = $derived((sw * 0.8) / V);
	const barX = (t, c) => sx(t) + sw * 0.1 + bw * c;
	const hBar = $derived(narrow ? 38 : 50);
	const top = $derived(interactive ? (narrow ? 132 : 86) : 40);
	const rowH = $derived(hBar + (narrow ? 66 : 44));
	const base = (j) => top + 26 + j * rowH + hBar + (narrow ? 22 : 0);
	const H = $derived(base(2) + 40);

	const readout = $derived.by(() => {
		if (!rows.length) return null;
		const S = `‘${showChar(vocab[s])}’`, U = `‘${showChar(vocab[u])}’`;
		const parts = rows.map((r) => {
			const here = r.slots[0];
			const later = r.slots.slice(1);
			const maxTV = later.length ? Math.max(...later.map((q) => q.tv)) : 0;
			const changed = later.filter((q) => q.top0 !== q.top1).length;
			return `depth ${depthLabel(r.d)}: c = (${fmt(r.c[0], 2)}, ${fmt(r.c[1], 2)}) becomes (${fmt(r.c[1], 2)}, ${fmt(r.c[0], 2)}); here the top character goes from ‘${showChar(vocab[here.top0])}’ to ‘${showChar(vocab[here.top1])}’ (${fmtProb(here.after[here.top1], 2, true)})${later.length ? `; later positions: ${changed} of ${later.length} change their top character, largest change in total variation ${fmt(maxTV, 2)}` : ''}`;
		});
		return { who: `Swap ${S} and ${U} in column ${pos + 1} of ‘${chars.map(showChar).join('')}’.`, parts };
	});
</script>

{#snippet slot(q, j)}
	{@const b0 = base(j)}
	<line class="rule" x1={sx(q.t) + sw * 0.06} x2={sx(q.t) + sw * 0.94} y1={b0} y2={b0} />
	{#each q.after as p, c (c)}
		<rect class="bar--value swap__bar" class:is-on={c === s || c === u} x={barX(q.t, c) + bw * 0.14} y={b0 - p * hBar} width={bw * 0.72} height={Math.max(0, p * hBar)} />
		<rect class="swap__prior" x={barX(q.t, c) + bw * 0.14} y={b0 - q.prior[c] * hBar} width={bw * 0.72} height={Math.max(0, q.prior[c] * hBar)} />
	{/each}
	<text class="lbl lbl--micro swap__tv" class:is-zero={q.tv < 5e-7} x={sx(q.t) + sw / 2} y={b0 + 14} text-anchor="middle">{q.tv < 5e-7 ? 'unchanged' : `TV ${fmt(q.tv, 2)}`}</text>
{/snippet}

<div class="fx fx-swap" use:measure={(w) => (width = w)}>
	{#if !J}
		<p class="fig__pending">The J-lens export (jspace/artifacts/specimen-jlens.json) is missing.</p>
	{:else}
		<svg viewBox="0 0 {W} {H}" width={W} height={H} role={interactive ? 'group' : 'img'} aria-label="{readout?.who} {readout?.parts.join('; ')}">
			{#if interactive}
				<CorpusStrip corpus={specimen.corpus} {T} bind:start bind:pos maxStart={nWin - 1} x={narrow ? 8 : 8} y={30} width={narrow ? W - 16 : Math.min(330, W * 0.4)} interactive label="edited column" />
				<CharRow {vocab} bind:value={s} marks={[u]} disabled={[u]} x={narrow ? 60 : W - padR - 22 * V} y={narrow ? 82 : 26} step={22} name="swap s" label="the character s" />
				<CharRow {vocab} bind:value={u} marks={[s]} disabled={[s]} x={narrow ? 60 : W - padR - 22 * V} y={narrow ? 112 : 56} step={22} name="with u" label="the character u" />
			{:else}
				<text class="lbl" x={padL} y={16}>Swap <tspan class="swap__tok">‘{showChar(vocab[s])}’</tspan> and <tspan class="swap__tok">‘{showChar(vocab[u])}’</tspan> in column {pos + 1} of the window <tspan class="swap__tok">‘{chars.map(showChar).join('')}’</tspan></text>
			{/if}

			<!-- the window's text, one character per slot -->
			{#each chars as ch, t (t)}
				<text class="tok swap__win" class:is-edit={t === pos} class:is-past={t < pos} x={sx(t) + sw / 2} y={top + 8} text-anchor="middle">{showChar(ch)}</text>
				<text class="lbl lbl--micro" x={sx(t) + sw / 2 + 9} y={top + 11}>{t + 1}</text>
			{/each}
			<text class="lbl lbl--micro" x={sx(pos) + sw / 2} y={top - 10} text-anchor="middle">edited</text>
			{#if pos > 0}<text class="lbl lbl--micro" x={sx(0) + (sw * pos) / 2} y={top + 26} text-anchor="middle">{pos > 1 ? 'earlier positions: no map reads forward in time' : 'earlier'}</text>{/if}

			{#each rows as r, j (j)}
				{@const b0 = base(j)}
				{#if narrow}
					<text class="lbl lbl--head" x={4} y={b0 - hBar - 8}>depth {depthLabel(r.d)}</text>
					<text class="lbl lbl--micro" x={70} y={b0 - hBar - 8}>c = ({fmt(r.c[0], 2)}, {fmt(r.c[1], 2)})</text>
				{:else}
					<text class="lbl lbl--head" x={padL - 16} y={b0 - hBar + 10} text-anchor="end">depth {depthLabel(r.d)}</text>
					<!-- G = VᵀV and c = V⁺x, as numbers -->
					{@const gx = padL - 16}
					<text class="lbl lbl--micro" x={gx - 74} y={b0 - hBar + 32} text-anchor="end">G</text>
					<text class="lbl swap__num" x={gx - 36} y={b0 - hBar + 28} text-anchor="end">{fmt(r.G[0][0], 2)}</text>
					<text class="lbl swap__num" x={gx} y={b0 - hBar + 28} text-anchor="end">{fmt(r.G[0][1], 2)}</text>
					<text class="lbl swap__num" x={gx - 36} y={b0 - hBar + 41} text-anchor="end">{fmt(r.G[1][0], 2)}</text>
					<text class="lbl swap__num" x={gx} y={b0 - hBar + 41} text-anchor="end">{fmt(r.G[1][1], 2)}</text>
					<path class="swap__paren" d="M{gx - 66} {b0 - hBar + 18} q-4 11 0 26 M{gx + 3} {b0 - hBar + 18} q4 11 0 26" />
					<text class="lbl lbl--micro" x={gx - 74} y={b0 - hBar + 62} text-anchor="end">c</text>
					<text class="lbl swap__num" x={gx} y={b0 - hBar + 62} text-anchor="end">({fmt(r.c[0], 2)}, {fmt(r.c[1], 2)})</text>
					<text class="lbl lbl--micro" x={gx} y={b0 - hBar + 76} text-anchor="end">→ ({fmt(r.c[1], 2)}, {fmt(r.c[0], 2)})</text>
				{/if}
				{#each r.slots as q (q.t)}
					{@render slot(q, j)}
				{/each}
			{/each}

			<!-- the characters under the bars of the edited slot, bottom row -->
			{#each vocab as ch, c (c)}
				<text class="tok swap__ch" class:is-on={c === s || c === u} x={barX(pos, c) + bw / 2} y={base(2) + 30} text-anchor="middle">{showChar(ch)}</text>
			{/each}
		</svg>
		{#if interactive && readout}
			<p class="readout" aria-live="polite">
				<span class="readout__who">{readout.who}</span>
				{#each readout.parts as part, k (k)}<span class="readout__part">{part}{k < readout.parts.length - 1 ? ';' : '.'}</span>{/each}
				<span class="readout__hint">Outlines: before the swap; solid: after. Click a character of the line to choose the edited column; choose s and u.</span>
			</p>
		{/if}
	{/if}
</div>

<style>
	.fx-swap :global(.swap__bar) { opacity: 0.3; }
	.fx-swap :global(.swap__bar.is-on) { opacity: 1; }
	.swap__prior { fill: none; stroke: var(--ink); stroke-width: var(--hair); }
	.swap__tv { fill: var(--ink-2); }
	.swap__tv.is-zero { fill: var(--ink-3); }
	.swap__num { font-size: 11.5px; fill: var(--ink); }
	.swap__paren { fill: none; stroke: var(--ink-3); stroke-width: var(--hair); }
	.swap__win { font-size: 17px; fill: var(--ink-2); }
	.swap__win.is-edit { fill: var(--ink); font-weight: 650; }
	.swap__win.is-past { fill: var(--ink-3); }
	.swap__ch { font-size: 11.5px; fill: var(--ink-3); }
	.swap__ch.is-on { fill: var(--ink); font-weight: 650; }
	.swap__tok { font-family: var(--serif); font-size: 14px; fill: var(--ink); }
</style>

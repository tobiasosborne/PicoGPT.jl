<script module>
	export const meta = { size: 'wide', title: 'Every head of the Shakespeare model on one passage' };
</script>

<script>
	// Figure: the trained Shakespeare model reading a passage of held-out text that contains a
	// repeated line. Top: the passage, each character's surprise −ln q (the model's charge for
	// predicting it) as a bar beneath it, and, for the chosen reader, arcs to what it reads in
	// the chosen head. Below: every head of every layer as a lower triangle of area-encoded
	// dots (rows = readers, columns = sources), with three measured habits per head:
	//   self — mean weight on the reader itself;  prev — on the character before it;
	//   copy — over the repeated line, on the character that followed the same character the
	//          first time (the move that would let a model copy the line).
	// Weights below 0.01 are not sent; dots smaller than the eye can see are not drawn.
	import { getContext } from 'svelte';
	import { measure } from '../primitives/measure.js';
	import Arcs from '../primitives/Arcs.svelte';
	import { fmt, fmtDense, layoutWidth } from '../primitives/scales.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const data = $derived(getContext('artifacts')?.()?.shakespeare?.attention ?? null);
	const interactive = $derived(mode !== 'print');
	let width = $state(868);

	const T = $derived(data?.T ?? 0);
	const chars = $derived(data?.chars ?? []);
	const show = (c) => (c === '\n' ? '¶' : c === ' ' ? '·' : c);
	// sparse rows → lookup
	const weight = (head, i, j) => {
		const r = head.rows[i];
		for (let k = 0; k < r.length; k += 2) if (r[k] === j) return r[k + 1];
		return 0;
	};
	// the repeat: the offset d (> 8) at which the passage best matches itself
	const repeat = $derived.by(() => {
		if (!data) return null;
		let best = { d: 0, n: 0 };
		for (let d = 9; d < T - 8; d++) {
			let n = 0;
			for (let i = d; i < T; i++) if (chars[i] === chars[i - d]) n++;
			if (n > best.n) best = { d, n };
		}
		const d = best.d;
		const R = [];
		for (let i = d; i + 1 < T; i++) if (chars[i] === chars[i - d] && chars[i + 1] === chars[i + 1 - d]) R.push(i);
		return { d, R };
	});
	const stats = $derived.by(() => {
		if (!data) return [];
		return data.heads.map((h) => {
			let self = 0, prev = 0;
			for (let i = 1; i < T; i++) { self += weight(h, i, i); prev += weight(h, i, i - 1); }
			const copy = repeat.R.length ? repeat.R.reduce((s, i) => s + weight(h, i, i - repeat.d + 1), 0) / repeat.R.length : 0;
			return { self: self / (T - 1), prev: prev / (T - 1), copy };
		});
	});
	const surprise = $derived(data ? data.next.map((n) => (n.p != null ? -Math.log(n.p) : null)) : []);
	const meanOver = (idx) => idx.reduce((s, i) => s + surprise[i], 0) / Math.max(1, idx.length);
	// the surprise of predicting the repeated line, first time and second time (same characters)
	const firstVsSecond = $derived(repeat ? { first: meanOver(repeat.R.map((i) => i - repeat.d)), second: meanOver(repeat.R) } : null);

	// default head: the one that reads the previous character most; default reader: in the repeat
	const defaultHead = $derived(stats.length ? stats.reduce((b, s, k) => (s.prev > stats[b].prev ? k : b), 0) : 0);
	// svelte-ignore state_referenced_locally
	let headSel = $state(attrs.head != null ? +attrs.head - 1 : -1);
	const head = $derived(headSel >= 0 ? headSel : defaultHead);
	let readerSel = $state(-1);
	let hover = $state(-1);
	const reader = $derived(hover >= 0 ? hover : readerSel >= 0 ? readerSel : repeat?.R?.[Math.floor((repeat?.R?.length ?? 0) / 3)] ?? T - 1);

	// geometry
	const W = $derived(layoutWidth(width, mode, 340));
	const narrow = $derived(W < 700);
	const pad = 8;
	const cw = $derived((W - 2 * pad) / Math.max(1, T));
	const xc = (i) => pad + cw * (i + 0.5);
	const yText = 108;
	const yBars = $derived(yText + 44);
	const barMax = 30;
	const lnV = $derived(Math.log(data?.V ?? 1));
	const sMax = $derived(Math.max(lnV, ...surprise.filter((v) => v != null)));
	const nCols = $derived(narrow ? 2 : 4);
	const gapX = $derived(narrow ? 16 : 22);
	const tri = $derived((W - 30 - (nCols - 1) * gapX) / nCols);
	const cell = $derived(tri / Math.max(1, T));
	const gridTop = $derived(yBars + 46);
	const rowGap = 48;
	const nRows = $derived(data ? Math.ceil(data.heads.length / nCols) : 0);
	const H = $derived(gridTop + nRows * (tri + rowGap) + 4);
	const pos = (k) => ({ x: 30 + (k % nCols) * (tri + gapX), y: gridTop + 22 + Math.floor(k / nCols) * (tri + rowGap) });
	const minDot = $derived((0.36 / (cell * 0.56)) ** 2); // weights whose dot would be under 0.36 px
	// one path per dot radius (to 0.05 px): each dot is a zero-length, round-capped stroke
	function dotPaths(h) {
		const R = cell * 0.56, bins = new Map();
		for (let i = 0; i < T; i++) {
			const r = h.rows[i];
			for (let k = 0; k < r.length; k += 2) {
				const w = r[k + 1];
				if (w < minDot) continue;
				const bin = Math.max(1, Math.round((R * Math.sqrt(w)) / 0.05));
				bins.set(bin, (bins.get(bin) ?? '') + `M${((r[k] + 0.5) * cell).toFixed(1)} ${((i + 0.5) * cell).toFixed(1)}h0`);
			}
		}
		return [...bins].map(([bin, d]) => ({ w: 2 * bin * 0.05, d }));
	}
	const rowFor = $derived.by(() => {
		if (!data) return {};
		const h = data.heads[head];
		const r = new Array(reader + 1).fill(0);
		const sr = h.rows[reader];
		for (let k = 0; k < sr.length; k += 2) r[sr[k]] = sr[k + 1];
		return { [reader]: r };
	});
	const xsAll = $derived(Array.from({ length: T }, (_, i) => xc(i)));
	const topReads = $derived.by(() => {
		if (!data) return [];
		const r = rowFor[reader] ?? [];
		return r.map((w, j) => ({ w, j })).filter((o) => o.w >= 0.05).sort((a, b) => b.w - a.w).slice(0, 3);
	});
	const ctxLabel = (i) => `‘${show(chars[i])}’ at ${i + 1}`;
</script>

<div class="fx fx-sattn" use:measure={(w) => (width = w)}>
	{#if !data}
		<p class="fig__pending">The Shakespeare model’s attention patterns (artifacts/shakespeare/attention.json) are not present in this build.</p>
	{:else}
		<svg viewBox="0 0 {W} {H}" width={W} height={H} role="img" aria-label="Attention of all {data.heads.length} heads on a passage of {T} characters; copy weight at most {fmt(Math.max(...stats.map((s) => s.copy)), 2)}">
			<!-- arcs: what the reader reads in the chosen head -->
			<g class="fx-sattn__arcs">
				<Arcs xs={xsAll} y={yText - 16} A={rowFor} {reader} dir={-1} wmax={9} aspect={0.5} hmax={84} labels={!narrow} labelAt={0.15} loop={5} />
			</g>
			<!-- the passage -->
			{#each chars as c, i (i)}
				<text class="tok fx-sattn__ch" class:is-reader={i === reader} class:is-nl={c === '\n' || c === ' '} x={xc(i)} y={yText} text-anchor="middle" style:font-size="{Math.min(13, cw * 1.3)}px">{show(c)}</text>
				{#if surprise[i - 1] != null && i > 0}
					{@const h = (surprise[i - 1] / sMax) * barMax}
					<rect class="bar--value fx-sattn__bar" x={xc(i) - cw * 0.3} y={yBars - h} width={cw * 0.6} height={h} />
				{/if}
				{#if interactive}
					<rect class="hit" x={xc(i) - cw / 2} y={yText - 16} width={cw} height={yBars - yText + 20} role="button" tabindex="-1" aria-label="reader {i + 1}" onpointerenter={() => (hover = i)} onpointerleave={() => (hover = -1)} onclick={() => (readerSel = i)} onkeydown={() => {}} />
				{/if}
			{/each}
			<line class="axis" x1={pad} x2={W - pad} y1={yBars} y2={yBars} />
			<line class="fx-sattn__lnv" x1={pad} x2={W - pad} y1={yBars - (lnV / sMax) * barMax} y2={yBars - (lnV / sMax) * barMax} />
			<text class="lbl lbl--micro" x={pad} y={yBars + 14}>surprise of each character, −ln <tspan class="m">q</tspan>; dashed: ln {data.V}, uniform guessing</text>
			{#if repeat?.R?.length}
				{@const a = repeat.R[0]}
				{@const b = repeat.R.at(-1) + 1}
				<path class="fx-sattn__br" d="M{xc(a - repeat.d) - cw / 2} {yBars + 22} v4 H{xc(b - repeat.d) + cw / 2} v-4" />
				<path class="fx-sattn__br" d="M{xc(a) - cw / 2} {yBars + 22} v4 H{xc(b) + cw / 2} v-4" />
				<text class="lbl lbl--micro" x={xc(a - repeat.d) - cw / 2} y={yBars + 38}>first time: {fmt(firstVsSecond.first, 2)} nats per character</text>
				<text class="lbl lbl--micro" x={xc(a) - cw / 2} y={yBars + 38}>second time: {fmt(firstVsSecond.second, 2)}</text>
			{/if}

			<!-- every head -->
			{#each data.heads as h, k (k)}
				{@const p = pos(k)}
				{@const st = stats[k]}
				<g class="fx-sattn__head" class:is-on={k === head}>
					{#if k % nCols === 0}<text class="lbl lbl--head" x={p.x - 10} y={p.y + tri / 2} text-anchor="middle" transform="rotate(-90 {p.x - 10} {p.y + tri / 2})">layer {h.layer}</text>{/if}
					<text class="lbl fx-sattn__hl" x={p.x} y={p.y - 8}>head {h.head}{k === head ? ' — arcs above' : ''}</text>
					<path class="fx-sattn__edge" d="M{p.x} {p.y} V{p.y + tri} H{p.x + tri}" />
					<g class="tri fx-sattn__dots" transform="translate({p.x} {p.y})">
						{#each dotPaths(h) as dp (dp.w)}<path d={dp.d} stroke-width={dp.w} />{/each}
					</g>
					<line class="fx-sattn__row" x1={p.x} x2={p.x + (reader + 1) * cell} y1={p.y + (reader + 0.5) * cell} y2={p.y + (reader + 0.5) * cell} />
					<text class="lbl lbl--micro fx-sattn__st" x={p.x} y={p.y + tri + 16}>self {fmtDense(st.self)} · prev {fmtDense(st.prev)} · copy {fmtDense(st.copy)}</text>
					{#if interactive}
						<rect class="hit" x={p.x - 4} y={p.y - 22} width={tri + 8} height={tri + 44} role="button" tabindex="0" aria-label="layer {h.layer}, head {h.head}" aria-pressed={k === head} onclick={() => (headSel = k)} onkeydown={(e) => (e.key === 'Enter' || e.key === ' ') && (headSel = k)} />
					{/if}
				</g>
			{/each}
		</svg>
		{#if interactive}
			<p class="readout" aria-live="polite">
				<span class="readout__who">Layer {data.heads[head].layer}, head {data.heads[head].head}; reader {ctxLabel(reader)}.</span>
				<span class="readout__part">It reads {topReads.map((o) => `${ctxLabel(o.j)} ${fmt(o.w, 2)}`).join(', ') || 'nothing above 0.05'}. The model’s favourite for the next character is ‘{show(data.next[reader].best)}’ ({fmt(data.next[reader].pBest, 2)}){data.next[reader].p != null ? `; the true one, ‘${show(chars[reader + 1])}’, got ${fmt(data.next[reader].p, 2)}` : ''}.</span>
				<span class="readout__hint">Point at a character to follow it through every head; click a head to draw its arcs. ¶ is a line break and · a space.</span>
			</p>
		{/if}
	{/if}
</div>

<style>
	.fx-sattn :global(.m) { font-family: var(--serif); font-style: italic; fill: var(--ink); font-size: 13px; }
	.fx-sattn__ch { font-size: 13px; fill: var(--ink-2); }
	.fx-sattn__ch.is-nl { fill: var(--ink-3); }
	.fx-sattn__ch.is-reader { fill: var(--ink); font-weight: 700; }
	.fx-sattn__lnv { stroke: var(--rule); stroke-width: var(--hair); stroke-dasharray: 2 2; }
	.fx-sattn__br { fill: none; stroke: var(--ink-3); stroke-width: var(--hair); }
	.fx-sattn__edge { fill: none; stroke: var(--rule); stroke-width: var(--hair); }
	.fx-sattn__dots path { fill: none; stroke: var(--value); stroke-linecap: round; }
	.fx-sattn__row { stroke: var(--ink); stroke-width: var(--hair); opacity: 0.5; }
	.fx-sattn__hl { fill: var(--ink-2); }
	.fx-sattn__head.is-on .fx-sattn__hl { fill: var(--ink); font-weight: 650; }
	.fx-sattn__st { fill: var(--ink-2); }
	.fx-sattn__arcs :global(.arcs__label) { font-size: 10px; }
</style>

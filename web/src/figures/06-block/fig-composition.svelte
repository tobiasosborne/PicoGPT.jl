<script module>
	export const meta = { size: 'wide', title: 'The model as one composition' };
</script>

<script>
	// Figure: the specimen as one composition of maps, top to bottom (forward = down).
	// Nodes are spaces, with the shape of their arrays for one window (JuliaMono).
	// Each map carries its parameters as area-encoded squares — one unit of area per
	// number, on one scale for the whole figure — filled for learned parameters and
	// open for the fixed positional code. Counts come from the layout, never typed.
	import { getContext } from 'svelte';
	import { measure } from '../primitives/measure.js';
	import { showChar, layoutWidth } from '../primitives/scales.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const { V, D, T, F, L: nBlocks } = $derived(specimen.config);
	let width = $state(868);
	const W = $derived(layoutWidth(width, mode));
	const narrow = $derived(W < 640);
	const size = (suffix) => specimen.layout.filter((l) => l.name === suffix || l.name.endsWith('.' + suffix)).reduce((s, l) => s + l.size, 0);
	const p = $derived(specimen.paramCount);

	// display names: [symbol, subscript]
	const arr = (name, sym, sub = '', open = false) => ({ name, sym, sub, n: name === 'P' ? D * T : size(name), open });
	const chars = $derived(Array.from(specimen.corpus.slice(0, T)).map(showChar).join(' '));

	const rows = $derived([
		{ node: { sym: 'x', sub: '', shape: `${T}`, note: `${T} characters: ${chars}` } },
		{ map: 'embed', formula: 'E[:, x] + P', acts: 'column by column', params: [arr('E', 'E'), arr('P', 'P', '', true)] },
		{ node: { sym: 'X', sub: '0', shape: `${D}×${T}`, note: 'the stream: one column per token' } },
		{ map: '+ attention', formula: 'X + W_o O + b_o,  O = attention(LN_1(X))', acts: 'across columns: the only map that is', block: true,
			params: [arr('gamma1', 'γ', '1'), arr('beta1', 'β', '1'), arr('Wq', 'W', 'q'), arr('bq', 'b', 'q'), arr('Wk', 'W', 'k'), arr('bk', 'b', 'k'), arr('Wv', 'W', 'v'), arr('bv', 'b', 'v'), arr('Wo', 'W', 'o'), arr('bo', 'b', 'o')] },
		{ node: { sym: 'X', sub: 'mid', shape: `${D}×${T}`, note: '' } },
		{ map: '+ MLP', formula: 'X + W_2 relu(W_1 LN_2(X) + b_1) + b_2', acts: 'column by column', block: true,
			params: [arr('gamma2', 'γ', '2'), arr('beta2', 'β', '2'), arr('W1', 'W', '1'), arr('b1', 'b', '1'), arr('W2', 'W', '2'), arr('b2', 'b', '2')] },
		{ node: { sym: 'X', sub: '1', shape: `${D}×${T}`, note: '' } },
		{ map: 'LayerNorm', formula: 'LN(X; γ_f, β_f)', acts: 'column by column', params: [arr('gammaf', 'γ', 'f'), arr('betaf', 'β', 'f')] },
		{ node: { sym: 'X', sub: 'f', shape: `${D}×${T}`, note: '' } },
		{ map: 'readout', formula: 'W_{out} X_f + b_{out}', acts: 'column by column', params: [arr('Wout', 'W', 'out'), arr('bout', 'b', 'out')] },
		{ node: { sym: 'Z', sub: '', shape: `${V}×${T}`, note: 'logits: negative energies' } },
		{ map: 'softmax', formula: 'of each column', acts: '', params: [] },
		{ node: { sym: 'p', sub: '', shape: `${V}×${T}`, note: `${T} distributions over the ${V} characters` } },
		{ map: 'cross-entropy', formula: 'with the next characters y', acts: '', params: [] },
		{ node: { sym: 'ℓ', sub: '', shape: '1', note: 'one number' } }
	]);

	// geometry
	const xNode = $derived(narrow ? 8 : 40);
	const xLine = $derived(xNode + 10);
	const xMap = $derived(xLine + 22);
	const xPar = $derived(narrow ? xMap : Math.min(W * 0.46, 400));
	const s = $derived(narrow ? 1.55 : 1.9); // px of side per √number
	const side = (n) => s * Math.sqrt(n);
	const gap = 10;
	const layout = $derived.by(() => {
		let y = 22;
		return rows.map((r) => {
			let h;
			if (r.node) h = 44;
			else {
				const maxSide = Math.max(0, ...r.params.map((q) => side(q.n)));
				h = narrow ? 62 + (r.params.length ? maxSide + 44 : 0) : Math.max(64, maxSide + 42);
			}
			const out = { ...r, y, h };
			y += h;
			return out;
		});
	});
	const H = $derived(layout.at(-1).y + layout.at(-1).h + (narrow ? 40 : 30));
	/** square positions along a row, bottom-aligned */
	function placed(params, x0, base) {
		let x = x0;
		return params.map((q) => {
			const sd = side(q.n);
			const out = { ...q, x, y: base - sd, sd };
			x += Math.max(sd, q.sym.length + q.sub.length > 3 ? 26 : 16) + gap;
			return out;
		});
	}
	const blockRows = $derived(layout.filter((r) => r.block));
	/** 'W_o x + b_{out}' → [{t:'W'}, {t:'o', sub:true}, …] */
	function segs(f) {
		const out = [];
		const re = /_(\{[^}]+\}|.)/g;
		let last = 0, m;
		while ((m = re.exec(f))) {
			if (m.index > last) out.push({ t: f.slice(last, m.index) });
			out.push({ t: m[1].replace(/[{}]/g, ''), sub: true });
			last = re.lastIndex;
		}
		if (last < f.length) out.push({ t: f.slice(last) });
		return out;
	}
	const total = $derived(layout.filter((r) => r.map).reduce((t, r) => t + r.params.filter((q) => !q.open).reduce((a, q) => a + q.n, 0), 0));
</script>

<div class="fx fx-comp" use:measure={(w) => (width = w)}>
	<svg viewBox="0 0 {W} {H}" width={W} height={H} role="img"
		aria-label="The specimen as a composition: characters, embedding, attention, MLP, final LayerNorm, readout, softmax, cross-entropy; {total} learned parameters.">
		<!-- the spine: forward runs down -->
		<line class="fx-comp__spine" x1={xLine} x2={xLine} y1={layout[0].y + 30} y2={layout.at(-1).y + 8} />

		<!-- the block bracket -->
		{#if blockRows.length}
			{@const y0 = blockRows[0].y - 6}
			{@const y1 = blockRows.at(-1).y + blockRows.at(-1).h + 6}
			{#if !narrow}
				<path class="fx-comp__bracket" d="M {xNode - 14} {y0} h -6 V {y1} h 6" />
				<text class="lbl lbl--micro fx-comp__blk" x={xNode - 26} y={(y0 + y1) / 2} text-anchor="middle" transform="rotate(-90 {xNode - 26} {(y0 + y1) / 2})">block, × {nBlocks}</text>
			{/if}
		{/if}

		{#each layout as r, k (k)}
			{#if r.node}
				<circle class="fx-comp__node" cx={xLine} cy={r.y + 16} r="3.2" />
				<text class="fx-comp__sym" x={xMap} y={r.y + 21}>{r.node.sym}{#if r.node.sub}<tspan class="fx-comp__sub" dy="4">{r.node.sub}</tspan>{/if}</text>
				<text class="fx-comp__shape" x={xMap + (r.node.sub.length > 1 ? 52 : 34)} y={r.y + 20}>{r.node.shape}</text>
				{#if r.node.note}
					<text class="lbl lbl--micro" x={xMap + (r.node.sub.length > 1 ? 52 : 34) + r.node.shape.length * 8 + 12} y={r.y + 20}>{r.node.note}</text>
				{/if}
			{:else}
				{@const sq = placed(r.params, narrow ? xMap : xPar, narrow ? r.y + r.h - 22 : r.y + r.h - 18)}
				<path class="fx-comp__arrow" d="M {xLine - 3.5} {r.y + r.h - 9} L {xLine} {r.y + r.h - 2} L {xLine + 3.5} {r.y + r.h - 9}" />
				<text class="lbl fx-comp__map" x={xMap} y={r.y + 22}>{r.map}</text>
				<text class="fx-comp__formula" x={xMap} y={r.y + 38}>{#each segs(r.formula) as g, i (i)}{#if g.sub}<tspan class="fx-comp__fsub" dy="3">{g.t}</tspan><tspan dy="-3">&#8203;</tspan>{:else}{g.t}{/if}{/each}</text>
				{#if r.acts}
					<text class="lbl lbl--micro fx-comp__acts" class:is-across={r.acts.startsWith('across')} x={xMap} y={r.y + 54}>{r.acts}</text>
				{/if}
				{#each sq as q (q.name)}
					<rect class="fx-comp__sq" class:is-open={q.open} x={q.x} y={q.y} width={q.sd} height={q.sd} />
					<text class="fx-comp__name" x={q.x + q.sd / 2} y={q.y + q.sd + 13} text-anchor="middle">{q.sym}<tspan class="fx-comp__nsub" dy="3">{q.sub}</tspan></text>
					<text class="lbl lbl--micro" x={q.x + q.sd / 2} y={q.y + q.sd + 25} text-anchor="middle">{q.n}</text>
				{/each}
				{#if r.params.length && !narrow}
					{@const learned = r.params.filter((q) => !q.open).reduce((a, q) => a + q.n, 0)}
					<text class="lbl fx-comp__count" x={W - 6} y={r.y + r.h - 18} text-anchor="end">{learned}</text>
				{:else if !r.params.length}
					<text class="lbl lbl--micro" x={narrow ? xMap + 120 : xPar} y={r.y + 22}>no parameters</text>
				{/if}
			{/if}
		{/each}

		<!-- how to read the squares, said directly -->
		{#if !narrow}
			{@const r0 = layout[1]}
			<text class="lbl lbl--micro" x={xPar} y={r0.y + 4}>area ∝ how many numbers; open: fixed, not learned</text>
			<text class="lbl lbl--micro" x={W - 6} y={r0.y + 4} text-anchor="end">learned</text>
		{/if}

		<!-- the sum -->
		{#if !narrow}
			{@const yT = layout.at(-1).y + 10}
			<line class="rule" x1={W - 120} x2={W - 6} y1={yT - 14} y2={yT - 14} />
			<text class="lbl fx-comp__total" x={W - 6} y={yT + 4} text-anchor="end">θ ∈ ℝ<tspan dy="-5" class="fx-comp__sup">{total}</tspan></text>
			<text class="lbl lbl--micro" x={W - 6} y={yT + 18} text-anchor="end">learned numbers in all</text>
		{:else}
			<text class="lbl fx-comp__total" x={xMap} y={H - 14}>θ ∈ ℝ<tspan dy="-5" class="fx-comp__sup">{total}</tspan>: every learned number</text>
		{/if}
		{#if total !== p}<text class="lbl" x="0" y={H - 2}>count mismatch {total} ≠ {p}</text>{/if}
	</svg>
</div>

<style>
	.fx-comp__spine { stroke: var(--value); stroke-width: var(--line); }
	.fx-comp__node { fill: var(--paper); stroke: var(--value); stroke-width: var(--line); }
	.fx-comp__arrow { fill: none; stroke: var(--value); stroke-width: var(--line); }
	.fx-comp__bracket { fill: none; stroke: var(--ink-3); stroke-width: var(--hair); }
	.fx-comp__blk { fill: var(--ink-3); letter-spacing: 0.04em; }
	.fx-comp__sym { font-family: var(--serif); font-style: italic; font-size: 19px; fill: var(--ink); }
	.fx-comp__sub { font-size: 12px; }
	.fx-comp__shape { font-family: var(--mono); font-size: 12px; fill: var(--ink-2); }
	.fx-comp__map { fill: var(--ink); font-weight: 600; font-size: 12.5px; }
	.fx-comp__formula { font-family: var(--serif); font-style: italic; font-size: 13px; fill: var(--ink-2); }
	.fx-comp__fsub { font-size: 9.5px; }
	.fx-comp__acts { fill: var(--ink-3); }
	.fx-comp__acts.is-across { fill: var(--ink); font-weight: 650; }
	.fx-comp__sq { fill: var(--value); }
	.fx-comp__sq.is-open { fill: none; stroke: var(--ink-3); stroke-width: var(--hair); stroke-dasharray: 2 2; }
	.fx-comp__name { font-family: var(--serif); font-style: italic; font-size: 12.5px; fill: var(--ink); }
	.fx-comp__nsub { font-size: 9px; }
	.fx-comp__count { fill: var(--ink); font-variant-numeric: tabular-nums lining-nums; }
	.fx-comp__total { fill: var(--ink); font-size: 14px; font-weight: 600; }
	.fx-comp__sup { font-size: 10px; }
</style>

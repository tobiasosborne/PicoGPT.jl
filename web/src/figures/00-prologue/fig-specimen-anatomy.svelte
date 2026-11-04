<script module>
	export const meta = { size: 'wide', title: 'The whole specimen' };
	export { default as Key } from './fig-specimen-anatomy.key.svelte';
</script>

<script>
	// Figure: what is the specimen, all of it? Every block of θ (trained) as a Hinton
	// diagram, in the order of the computation, top to bottom: the embedding; block 1's
	// attention (with LN₁ before it) and MLP (with LN₂); the final LayerNorm and the
	// readout. Matrices as written (rows top→bottom); each bias stands as one column right
	// of its matrix, each gain/offset as one column. ONE scale for the whole plate. The
	// layout is a table of placed blocks; pointing maps (x, y) to a block and an entry.
	import { getContext } from 'svelte';
	import { measure } from '../primitives/measure.js';
	import { fmt, showChar, layoutWidth } from '../primitives/scales.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const { D, V, F } = $derived(specimen.config);
	const blocks = $derived(new Map(specimen.params().map((p) => [p.name.split('.').at(-1), p])));
	const vmax = $derived(Math.max(...specimen.theta.map(Math.abs)));
	let width = $state(868);
	let hover = $state(null); // {blk, k}
	const interactive = $derived(mode !== 'print');

	// names as set in the book
	const NAME = {
		E: ['E', ''], gamma1: ['γ', '1'], beta1: ['β', '1'], Wq: ['W', 'q'], bq: ['b', 'q'], Wk: ['W', 'k'], bk: ['b', 'k'],
		Wv: ['W', 'v'], bv: ['b', 'v'], Wo: ['W', 'o'], bo: ['b', 'o'], gamma2: ['γ', '2'], beta2: ['β', '2'],
		W1: ['W', '1'], b1: ['b', '1'], W2: ['W', '2'], b2: ['b', '2'], gammaf: ['γ', 'f'], betaf: ['β', 'f'], Wout: ['W', 'out'], bout: ['b', 'out']
	};
	const ROLE = {
		E: 'the embedding: column a is character a (chapter 2)',
		gamma1: 'LayerNorm 1, gain (chapter 6)', beta1: 'LayerNorm 1, offset (chapter 6)',
		Wq: 'queries: what each reader looks for (chapter 5)', bq: 'query bias (chapter 5)',
		Wk: 'keys: what each source offers (chapter 5)', bk: 'key bias (chapter 5)',
		Wv: 'values: what each source hands over (chapter 5)', bv: 'value bias (chapter 5)',
		Wo: 'output map of attention, back into the stream (chapter 5)', bo: 'output bias (chapter 5)',
		gamma2: 'LayerNorm 2, gain (chapter 6)', beta2: 'LayerNorm 2, offset (chapter 6)',
		W1: 'MLP, up to the inner width (chapter 6)', b1: 'MLP, inner bias (chapter 6)',
		W2: 'MLP, back down to the stream (chapter 6)', b2: 'MLP, output bias (chapter 6)',
		gammaf: 'final LayerNorm, gain (chapter 6)', betaf: 'final LayerNorm, offset (chapter 6)',
		Wout: 'readout: one score per character (chapter 3)', bout: 'readout bias (chapter 3)'
	};

	// geometry: columns of cells, gaps in cells
	const W = $derived(layoutWidth(width, mode, 340));
	const narrow = $derived(W < 640);
	const labelW = $derived(narrow ? 0 : 112);
	const countW = $derived(narrow ? 0 : 54);
	// γ and β stand apart by GB cells so that each name fits over its own column
	const GB = 1.2;
	// widest station, in cells: γ β + gap + 4 × (8 + 1) + 3 gaps … and MLP: γ β + gap + 9 + gap + 33
	const widthCells = $derived(Math.max(2 + GB + 2.2 + 4 * (D + 1.25) + 3 * 1.6, 2 + GB + 2.2 + (D + 1.25) + 2.2 + (F + 1.25)));
	const c = $derived(Math.max(4.2, Math.min(interactive ? 13 : 8.5, (W - labelW - countW - 10) / widthCells)));
	// characters labelling E's columns and W_out's rows: no larger than the cell pitch allows
	const tokFs = $derived(Math.min(11.5, c * 1.2)); // across: E's columns
	const rowFs = $derived(Math.min(11.5, c * 1.08)); // down: W_out's rows, one line per row
	const gx = $derived(labelW + 6);
	const nameH = 26;

	/** place a block: name → {x, y, rows, cols, get(i,j) → θ index} */
	function placeAll() {
		const out = [], stations = [];
		let y = nameH + 6;
		const put = (name, x, yy) => {
			const p = blocks.get(name);
			const [r, cc] = p.shape.length === 2 ? p.shape : [p.shape[0], 1];
			out.push({ name, x, y: yy, rows: r, cols: cc, p });
			return x + cc * c;
		};
		const gap = 1.6 * c, ggap = 2.2 * c;
		// station 1: embedding
		let x = gx + (2 + GB) * c + ggap; // over W_q and W_1
		put('E', x, y + c * 1.1);
		stations.push({ key: 'embed', title: 'embedding', y0: y, y1: y + c * 1.1 + D * c, names: ['E'] });
		y += c * 1.1 + D * c + 3.1 * c + nameH;
		// station 2: LN1 + attention
		x = gx;
		x = put('gamma1', x, y); x = put('beta1', x + GB * c, y);
		x += ggap;
		for (const h of ['q', 'k', 'v', 'o']) { x = put('W' + h, x, y); x = put('b' + h, x + 0.25 * c, y); x += gap; }
		stations.push({ key: 'attn', title: 'attention', y0: y, y1: y + D * c, names: ['gamma1', 'beta1', 'Wq', 'bq', 'Wk', 'bk', 'Wv', 'bv', 'Wo', 'bo'] });
		y += D * c + 2.4 * c + nameH;
		// station 3: LN2 + MLP (8-row things centred on the 32-row W1)
		const mid = y + ((F - D) / 2) * c;
		x = gx;
		x = put('gamma2', x, mid); x = put('beta2', x + GB * c, mid);
		x += ggap;
		x = put('W1', x, y); x = put('b1', x + 0.25 * c, y);
		x += ggap;
		x = put('W2', x, mid); x = put('b2', x + 0.25 * c, mid);
		stations.push({ key: 'mlp', title: 'MLP', y0: y, y1: y + F * c, names: ['gamma2', 'beta2', 'W1', 'b1', 'W2', 'b2'] });
		y += F * c + 2.4 * c + nameH;
		// station 4: final LN + readout
		x = gx;
		x = put('gammaf', x, y); x = put('betaf', x + GB * c, y);
		x += ggap + 1.2 * c; // a lane for W_out's row labels (the characters)
		x = put('Wout', x, y); x = put('bout', x + 0.25 * c, y);
		stations.push({ key: 'out', title: 'readout', y0: y, y1: y + D * c, names: ['gammaf', 'betaf', 'Wout', 'bout'] });
		y += D * c + 2.6 * c;
		return { placed: out, stations, H: y + 20 };
	}
	const lay = $derived(placeAll());
	const H = $derived(lay.H);
	const count = (names) => names.reduce((s, n) => s + blocks.get(n).size, 0);
	const side = (v) => c * 0.92 * Math.sqrt(Math.min(1, Math.abs(v) / vmax));

	// pointer → block and entry
	function locate(e) {
		const svg = e.currentTarget;
		const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
		const q = pt.matrixTransform(svg.getScreenCTM().inverse());
		for (const b of lay.placed) {
			if (q.x >= b.x && q.x < b.x + b.cols * c && q.y >= b.y && q.y < b.y + b.rows * c) {
				const i = Math.floor((q.y - b.y) / c), j = Math.floor((q.x - b.x) / c);
				return { b, i, j, k: i + b.rows * j };
			}
		}
		return null;
	}
	const sub = (s) => s;
	const describe = (h) => {
		if (!h) return null;
		const { b, i, j, k } = h;
		const [sym, s] = NAME[b.name];
		const v = b.p.values[k];
		const idx = b.p.shape.length === 2 ? `[${i + 1}, ${j + 1}]` : `[${i + 1}]`;
		return {
			name: `${sym}${s ? '_' + s : ''}`, sym, s, idx, v,
			shape: b.p.shape.length === 2 ? `${b.p.shape[0]} × ${b.p.shape[1]}` : `${b.p.shape[0]}`,
			size: b.p.size, from: b.p.offset + 1, to: b.p.offset + b.p.size, at: b.p.offset + k + 1, role: ROLE[b.name]
		};
	};
	function parameterKey(e) {
		let k = hover ? hover.b.p.offset + hover.k : 0;
		if (e.key === 'ArrowRight' || e.key === 'ArrowDown') k++;
		else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') k--;
		else if (e.key === 'Home') k = 0;
		else if (e.key === 'End') k = specimen.paramCount - 1;
		else return;
		e.preventDefault(); k = Math.max(0, Math.min(specimen.paramCount - 1, k));
		const b = lay.placed.find((b) => k >= b.p.offset && k < b.p.offset + b.p.size);
		const q = k - b.p.offset;
		hover = { b, k: q, i: q % b.rows, j: Math.floor(q / b.rows) };
	}

	const info = $derived(describe(hover));
</script>

<div class="fx fx-anat" use:measure={(w) => (width = w)}>
	<!-- svelte-ignore a11y_no_noninteractive_tabindex (keyboard navigation on the composite drawing) -->
	<svg
		viewBox="0 0 {W} {H}"
		width={W}
		height={H}
		role={interactive ? 'slider' : 'img'}
		tabindex={interactive ? 0 : undefined}
		aria-valuemin={interactive ? 1 : undefined}
		aria-valuemax={interactive ? specimen.paramCount : undefined}
		aria-valuenow={interactive ? (info?.at ?? 1) : undefined}
		aria-valuetext={interactive && info ? `${info.name}${info.idx} = ${info.v}` : undefined}
		onkeydown={parameterKey}
		onfocus={() => { if (!hover) { const b = lay.placed[0]; hover = { b, k: 0, i: 0, j: 0 }; } }}
		aria-label="All {specimen.paramCount} parameters of the trained specimen, block by block: {lay.stations.map((s) => `${s.title} ${count(s.names)}`).join(', ')}."
		onpointermove={(e) => interactive && (hover = locate(e))}
		onpointerleave={() => (hover = null)}
	>
		<!-- forward: down -->
		{#if !narrow}
			{@const a = lay.stations[1]}
			{@const m = lay.stations[2]}
			<line class="fx-anat__spine" x1={10} x2={10} y1={lay.stations[0].y0} y2={lay.stations.at(-1).y1} />
			<path class="fx-anat__arrow" d="M6,{lay.stations.at(-1).y1 - 6} L10,{lay.stations.at(-1).y1} L14,{lay.stations.at(-1).y1 - 6}" />
			{#each lay.stations as s (s.key)}
				<circle class="fx-anat__node" cx={10} cy={(s.y0 + s.y1) / 2} r="2.2" />
				<text class="lbl fx-anat__station" x={22} y={(s.y0 + s.y1) / 2 - 3}>{s.title}</text>
				<text class="lbl lbl--micro" x={22} y={(s.y0 + s.y1) / 2 + 12}>{count(s.names)} numbers</text>
			{/each}
			<!-- the block: attention + MLP -->
			<path class="fx-anat__bracket" d="M{labelW - 4},{a.y0 - nameH + 4} h-5 V{m.y1 + 4} h5" />
			<text class="lbl lbl--micro" x={labelW - 14} y={a.y0 - nameH + 2} text-anchor="end">block 1</text>
		{/if}

		<!-- column labels of E (characters) and row labels of W_out -->
		{#each lay.placed as b (b.name)}
			{@const [sym, s] = NAME[b.name]}
			{@const on = hover?.b.name === b.name}
			{#if b.name === 'E'}
				{#each specimen.vocab as ch, j (j)}
					<text class="tok fx-anat__tok" x={b.x + c * (j + 0.5)} y={b.y - 4} text-anchor="middle" font-size={tokFs}>{showChar(ch)}</text>
				{/each}
			{/if}
			{#if b.name === 'Wout'}
				{#each specimen.vocab as ch, i (i)}
					<text class="tok fx-anat__tok" x={b.x - 0.9 * c} y={b.y + c * (i + 0.5) + rowFs * (ch === ' ' ? -0.02 : 0.28)} text-anchor="middle" font-size={rowFs}>{showChar(ch)}</text>
				{/each}
			{/if}
			<!-- the name, over the block -->
			<text class="fx-anat__name" class:is-on={on} x={b.x + (b.cols * c) / 2} y={b.y - (b.name === 'E' ? c * 1.1 + 6 : 6)} text-anchor="middle"><tspan class="m">{sym}</tspan>{#if s}<tspan class="m sub" dy="3">{s}</tspan>{/if}</text>
			<rect class="fx-anat__frame" class:is-on={on} x={b.x} y={b.y} width={b.cols * c} height={b.rows * c} />
			<g class="matrix matrix--value">
				{#each b.p.values as v, k (k)}
					{@const sd = side(v)}
					{#if sd > 0.15}
						<rect class="matrix__cell" class:matrix__cell--neg={v < 0} x={b.x + c * Math.floor(k / b.rows) + (c - sd) / 2} y={b.y + c * (k % b.rows) + (c - sd) / 2} width={sd} height={sd} />
					{/if}
				{/each}
			</g>
		{/each}
		{#if hover}
			<rect class="fx-anat__cell" x={hover.b.x + c * hover.j} y={hover.b.y + c * hover.i} width={c} height={c} />
		{/if}

		<!-- the total -->
		<text class="lbl" x={W - 4} y={H - 8} text-anchor="end"><tspan class="m">θ</tspan> ∈ ℝ<tspan dy="-5" class="fx-anat__sup">{specimen.paramCount}</tspan><tspan dy="5">: {specimen.paramCount} numbers in all</tspan></text>
	</svg>
	{#if interactive}
		<p class="readout" aria-live="polite">
			{#if info}
				<span class="readout__who"><i>{info.sym}</i>{#if info.s}<sub>{info.s}</sub>{/if}{info.idx} = {fmt(info.v, 4)}</span>
				<span class="readout__part">θ<sub>{info.at}</sub>. The block <i>{info.sym}</i>{#if info.s}<sub>{info.s}</sub>{/if} is {info.shape}, {info.size} numbers, θ<sub>{info.from}</sub>…θ<sub>{info.to}</sub>: {info.role}.</span>
			{:else}
				<span class="readout__part">Every parameter of the trained specimen, {specimen.paramCount} in all, in the order of the computation.</span>
			{/if}
			<span class="readout__hint">Point at a block to read its name and job, and its value.</span>
		</p>
	{/if}
</div>

<style>
	.fx-anat :global(.m) { font-family: var(--serif); font-style: italic; font-size: 14px; fill: var(--ink); }
	.fx-anat :global(.sub) { font-size: 9.5px; }
	.fx-anat__name.is-on :global(.m) { font-weight: 700; }
	.fx-anat__tok { fill: var(--ink-2); }
	.fx-anat__frame { fill: none; stroke: var(--rule-2); stroke-width: var(--hair); }
	.fx-anat__frame.is-on { stroke: var(--ink-3); }
	.fx-anat__cell { fill: none; stroke: var(--ink); stroke-width: 1px; }
	.fx-anat__spine { stroke: var(--ink-3); stroke-width: var(--hair); }
	.fx-anat__arrow { fill: none; stroke: var(--ink-3); stroke-width: var(--hair); }
	.fx-anat__node { fill: var(--ink-3); }
	.fx-anat__station { fill: var(--ink); font-weight: 600; font-variant-caps: all-small-caps; letter-spacing: 0.05em; font-size: 13px; }
	.fx-anat__bracket { fill: none; stroke: var(--ink-3); stroke-width: var(--hair); }
	.fx-anat__sup { font-size: 9px; }
</style>

<script module>
	export const meta = { size: 'wide', title: 'Every minibatch gradient' };
	export { default as Key } from './fig-minibatch.key.svelte';
</script>

<script>
	// Figure: every minibatch gradient of the specimen, projected onto a fixed plane.
	// M = 10 training windows. For b = 1, 2, 5, 10 the panel holds the gradient of the
	// loss on EVERY set of b windows (10, 45, 252, 1 of them), each the real gradient
	// of that sub-batch (a backward pass on it), projected onto the plane spanned by
	//   u₁ = the direction of the full gradient g,
	//   u₂ = the main direction in which the single windows disagree with it (⟂ u₁).
	// A linear projection maps means to means, so the cross (g) sits at the mean of
	// every cloud: the minibatch gradient is unbiased. The dashed circle is the cloud's
	// rms distance from g; it shrinks like √((M−b)/(b(M−1))).
	// Two states: θ₀ (before training) and the trained θ (after).
	import { getContext } from 'svelte';
	import { measure } from '../primitives/measure.js';
	import { computed } from '../primitives/computed.svelte.js';
	import { fmt, showChar, layoutWidth } from '../primitives/scales.js';
	import { sci } from './paramname.js';
	import { MINIBATCH_KEY, minibatchJob, SIZES } from './minibatch.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const interactive = $derived(mode !== 'print');
	// svelte-ignore state_referenced_locally
	let state = $state(attrs.state === 'after' ? 1 : 0);


	// svelte-ignore state_referenced_locally
	const job = computed(MINIBATCH_KEY, minibatchJob(specimen));

	const data = $derived(job.value);
	const st = $derived(data?.[state]);
	const M = $derived(specimen.lab.train.x.length);
	const winText = (b) => specimen.corpus.slice(b, b + specimen.config.T);

	// geometry
	let width = $state(868);
	const W = $derived(layoutWidth(width, mode, 340));
	const narrow = $derived(W < 620);
	const cols = $derived(narrow ? 2 : 4);
	const gap = $derived(narrow ? 14 : 22);
	const side = $derived((W - gap * (cols - 1)) / cols);
	const head = 44;
	const rowsN = $derived(Math.ceil(SIZES.length / cols));
	const blockH = $derived(rowsN * (side + 34) + 8);

	/** Shared scale per state: the farthest single-window gradient from the panel centre. */
	function frame(s) {
		const [gx, gy] = s.g;
		let r = 0;
		for (const c of s.clouds) for (const p of c.pts) r = Math.max(r, Math.hypot(p[0] - gx, p[1] - gy), Math.hypot(gx, gy) * 0.5);
		r = Math.max(r, Math.hypot(gx, gy) * 1.05);
		return { cx: gx, cy: gy, r: r * 1.12 };
	}
	function niceStep(r) {
		const e = 10 ** Math.floor(Math.log10(r));
		return [1, 2, 5, 10].map((m) => m * e).filter((v) => v <= r * 0.6).at(-1) ?? e;
	}

	/** A window's first prediction, the one its gradient is about at the floor: "o → ␣". */
	const firstStep = (b) => `${showChar(specimen.corpus[b])} → ${showChar(specimen.corpus[b + 1])}`;

	/** Labels for the single-window marks: beside the mark, nudged apart vertically, with a
	 *  hairline leader when moved. Marks within 18% of the scale of g are counted, not named. */
	function placeLabels(s, c, fr, ox, oy) {
		const sc = side / 2 / fr.r;
		const X = (v) => ox + side / 2 + (v - fr.cx) * sc, Y = (v) => oy + side / 2 - (v - fr.cy) * sc;
		const far = [], near = [];
		for (const p of c.pts) (Math.hypot(p[0] - s.g[0], p[1] - s.g[1]) > fr.r * 0.18 ? far : near).push(p);
		const out = [];
		for (const right of [true, false]) {
			const inward = (p) => Math.hypot(p[0] - fr.cx, p[1] - fr.cy) > fr.r * 0.5; // far marks: label towards the centre
			const grp = far.filter((p) => ((p[0] < fr.cx) === inward(p)) === right).map((p) => ({
				key: p[2], right, dx: X(p[0]) + (right ? 3.4 : -3.4), dy: Y(p[1]),
				x: X(p[0]) + (right ? 8 : -8), y0: Y(p[1]) + 4,
				text: firstStep(+p[2])
			})).sort((a, b) => a.y0 - b.y0);
			let last = -Infinity;
			for (const L of grp) { L.y = Math.max(L.y0, last + 12.5); last = L.y; }
			const over = last - (oy + side - 2);
			if (over > 0) for (const L of grp) L.y -= over;
			for (const L of grp) { L.leader = Math.abs(L.y - L.y0) > 2.5; if (L.leader) L.x += right ? 6 : -6; out.push(L); }
		}
		return out;
	}

	let hover = $state(null);
	function onmove(e, s, k, fr, ox, oy) {
		if (!interactive) return;
		const svg = e.currentTarget.ownerSVGElement;
		const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
		const p = pt.matrixTransform(svg.getScreenCTM().inverse());
		const sc = side / 2 / fr.r;
		let best = null, bd = 14;
		for (const q of s.clouds[k].pts) {
			const x = ox + side / 2 + (q[0] - fr.cx) * sc, y = oy + side / 2 - (q[1] - fr.cy) * sc;
			const d = Math.hypot(x - p.x, y - p.y);
			if (d < bd) { bd = d; best = q; }
		}
		hover = best ? { b: s.clouds[k].b, set: best[2].split('.').map(Number), norm: best[3] } : null;
	}

	const formula = (b) => Math.sqrt((M - b) / (b * (M - 1)));
	const readout = $derived.by(() => {
		if (!st) return null;
		const r1 = st.clouds[0].rms;
		return {
			state: state ? 'After training' : 'Before training',
			g: `the full gradient has length ${fmt(st.gnorm, 3)}`,
			unbiased: `every cloud’s mean equals it to within ${sci(st.meanErr, 1)}`,
			rms: st.clouds.filter((c) => c.b < M).map((c) => `b = ${c.b}: ${fmt(c.rms, 3)} (${fmt(c.rms / r1, 2)}; formula ${fmt(formula(c.b), 2)})`).join(', ')
		};
	});
</script>

{#snippet panel(s, k, ox, oy, fr, labels, named)}
	{@const c = s.clouds[k]}
	{@const sc = side / 2 / fr.r}
	{@const X = (v) => ox + side / 2 + (v - fr.cx) * sc}
	{@const Y = (v) => oy + side / 2 - (v - fr.cy) * sc}
	<g>
		<text class="lbl lbl--head" x={ox} y={oy - 10}><tspan class="m">b</tspan> = {c.b}</text>
		<text class="lbl lbl--micro" x={ox + side} y={oy - 10} text-anchor="end">{c.n} {c.n === 1 ? 'batch' : 'batches'}</text>
		<!-- the origin: zero gradient -->
		{#if X(0) > ox && X(0) < ox + side && Y(0) > oy && Y(0) < oy + side}
			<line class="axis" x1={X(0) - 5} x2={X(0) + 5} y1={Y(0)} y2={Y(0)} />
			<line class="axis" x1={X(0)} x2={X(0)} y1={Y(0) - 5} y2={Y(0) + 5} />
			<text class="lbl lbl--micro" x={X(0) - 5} y={Y(0) + 13} text-anchor="end">0</text>
		{/if}
		{#if c.b < M}
			<circle class="fx-mb__rms" cx={X(s.g[0])} cy={Y(s.g[1])} r={c.rmsP * sc} />
		{/if}
		<g class="fx-mb__marks">
			{#each c.pts as p, j (j)}
				{@const on = hover && hover.b === c.b && hover.set.join('.') === p[2]}
				<rect class="fx-mb__bar" class:is-on={on} x={X(p[0]) - (c.b === M ? 6 : 3.4)} y={Y(p[1]) - 0.8} width={c.b === M ? 12 : 6.8} height="1.6" />
			{/each}
		</g>
		<!-- the full gradient g, and the mean of the cloud (the same point) -->
		<line class="fx-mb__g" x1={X(s.g[0])} x2={X(s.g[0])} y1={Y(s.g[1]) - 7} y2={Y(s.g[1]) + 7} />
		<line class="fx-mb__g" x1={X(s.g[0]) - 7} x2={X(s.g[0]) + 7} y1={Y(s.g[1])} y2={Y(s.g[1])} />
		{#if named && c.b === 1}
			{#each placeLabels(s, c, fr, ox, oy) as L (L.key)}
				{#if L.leader}<line class="leader" x1={L.dx} y1={L.dy} x2={L.x - (L.right ? 2 : -2)} y2={L.y - 4} />{/if}
				<text class="tok fx-mb__win" x={L.x} y={L.y} text-anchor={L.right ? 'start' : 'end'}>{L.text}</text>
			{/each}
		{/if}
		{#if named && c.b === 1}
			{@const near = c.pts.filter((p) => Math.hypot(p[0] - s.g[0], p[1] - s.g[1]) <= fr.r * 0.18)}
			{#if near.length}
				<text class="lbl lbl--micro" x={X(s.g[0]) - 8} y={Y(s.g[1]) - 8} text-anchor="end">{near.length} more</text>
			{/if}
		{/if}
		{#if c.b < M && labels}
			<text class="lbl lbl--micro" x={ox + side - 4} y={oy + side - 6} text-anchor="end">{c.b === 1 ? 'rms distance' : `rms × ${fmt(c.rmsP / s.clouds[0].rmsP, 2)}`}</text>
		{/if}
		{#if interactive}
			<rect class="hit" x={ox} y={oy} width={side} height={side} role="presentation" onpointermove={(e) => onmove(e, s, k, fr, ox, oy)} onpointerleave={() => (hover = null)} />
		{/if}
	</g>
{/snippet}

{#snippet block(s, y0, title, named)}
	{@const fr = frame(s)}
	{@const unit = niceStep(fr.r)}
	{#if title}<text class="lbl lbl--head" x="0" y={y0 + 12}>{title}</text>{/if}
	{#each s.clouds as c, k (c.b)}
		{@const col = k % cols}
		{@const row = Math.floor(k / cols)}
		{@render panel(s, k, col * (side + gap), y0 + head + row * (side + 34), fr, true, named)}
	{/each}
	<!-- scale bar -->
	{@const sy = y0 + head + rowsN * (side + 34) - 14}
	<line class="axis" x1={W - unit * (side / 2 / fr.r)} x2={W} y1={sy} y2={sy} />
	<line class="axis" x1={W - unit * (side / 2 / fr.r)} x2={W - unit * (side / 2 / fr.r)} y1={sy - 3} y2={sy + 3} />
	<line class="axis" x1={W} x2={W} y1={sy - 3} y2={sy + 3} />
	<text class="lbl lbl--micro" x={W - unit * (side / 2 / fr.r) - 5} y={sy + 3.5} text-anchor="end">gradient length {unit < 0.1 ? fmt(unit, 2) : unit}</text>
{/snippet}

<div class="fx fx-mb" use:measure={(w) => (width = w)}>
	{@html job.tag}
	{#if !data}
		<svg viewBox="0 0 {W} 60" width={W} height="60" role="img" aria-label="computing every minibatch gradient">
			<line class="rule" x1="0" x2={W} y1="30" y2="30" />
			<line class="fx-mb__progress" x1="0" x2={W * job.progress} y1="30" y2="30" />
			<text class="lbl lbl--micro" x="0" y="20">computing all 616 minibatch gradients…</text>
		</svg>
	{:else if interactive}
		<svg viewBox="0 0 {W} {blockH + head}" width={W} height={blockH + head} role="img"
			aria-label="{readout.state}: {readout.g}; {readout.unbiased}; rms distance {readout.rms}">
			<g class="fx-mb__states">
				{#each ['before training', 'after training'] as label, k (k)}
					<text class="lbl fx-mb__state" class:is-on={state === k} x={k * 118} y="14" role="button" tabindex="0" aria-pressed={state === k}
						onclick={() => (state = k)} onkeydown={(e) => (e.key === 'Enter' || e.key === ' ') && ((state = k), e.preventDefault())}>{label}</text>
				{/each}
			</g>
			{@render block(st, 0, '', state === 1)}
		</svg>
		<p class="readout" aria-live="polite">
			{#if hover}
				<span class="readout__who">{hover.set.length === 1 ? 'One window' : `${hover.set.length} windows`}: {hover.set.map((b) => `“${winText(b)}”`).join(', ')}.</span>
				<span class="readout__part">{hover.set.length === 1 ? `Its first prediction is ${firstStep(hover.set[0])}. Its` : 'Their'} gradient has length {fmt(hover.norm, 3)}; the full gradient’s is {fmt(st.gnorm, 3)}.</span>
			{:else}
				<span class="readout__who">{readout.state}:</span>
				<span class="readout__part">{readout.g}, and {readout.unbiased}. The rms distance from it: {readout.rms}.</span>
				<span class="readout__hint">Click “before” or “after training”; point at a mark to see its windows.</span>
			{/if}
		</p>
	{:else}
		<svg viewBox="0 0 {W} {2 * (blockH + 22)}" width={W} height={2 * (blockH + 22)} role="img" aria-label="Every minibatch gradient before and after training">
			{@render block(data[0], 0, 'before training', false)}
			{@render block(data[1], blockH + 22, 'after training', true)}
		</svg>
	{/if}
</div>

<style>
	.fx-mb :global(.m) { font-family: var(--serif); font-style: italic; }
	.fx-mb__marks { mix-blend-mode: var(--overprint); }
	.fx-mb__bar { fill: var(--covector); }
	.fx-mb__bar.is-on { fill: var(--ink); }
	.fx-mb__g { stroke: var(--ink); stroke-width: 1.6px; }
	.fx-mb__rms { fill: none; stroke: var(--ink-3); stroke-width: var(--hair); stroke-dasharray: 2 2.5; }
	.fx-mb__win { font-size: 12px; fill: var(--ink-2); paint-order: stroke; stroke: var(--paper); stroke-width: 3px; stroke-linejoin: round; }
	.fx-mb__state { cursor: pointer; fill: var(--ink-3); font-weight: 500; }
	.fx-mb__state.is-on { fill: var(--ink); font-weight: 650; text-decoration: underline; text-underline-offset: 3px; }
	.fx-mb__progress { stroke: var(--covector); stroke-width: 1.5px; }
</style>

<script module>
	export const meta = { size: 'wide', title: 'A language model is a map from contexts to distributions' };
</script>

<script>
	// Figure: the specimen as a map x_{≤t} ↦ q_θ(· | x_{≤t}), on the T prefixes of one window.
	// Small multiples, one column per prefix length t: the context on top (growing by one
	// character per column), and below it the V probabilities as horizontal bars, rows =
	// characters in alphabet order, aligned across columns, all on one scale (0…1).
	// The character that actually follows is outlined. Probabilities ≥ 0.1 are labelled.
	// Data: windowTrace(start).probs. Drag the window along the corpus.
	import { getContext } from 'svelte';
	import { measure } from '../primitives/measure.js';
	import { fmt, showChar, clamp, layoutWidth } from '../primitives/scales.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const T = $derived(specimen.config.T);
	const V = $derived(specimen.config.V);
	const maxStart = $derived(specimen.corpus.length - T);
	// svelte-ignore state_referenced_locally
	let start = $state(clamp((+attrs.window || 1) - 1, 0, specimen.corpus.length - specimen.config.T));
	let focus = $state(-1);
	let width = $state(868);
	const interactive = $derived(mode !== 'print');

	const tr = $derived(specimen.windowTrace(start));
	const nextId = (t) => specimen.vocab.indexOf(specimen.corpus[start + t + 1]); // 0-based, −1 past the end

	// geometry
	const W = $derived(layoutWidth(width, mode, 330));
	const narrow = $derived(W < 640);
	const padL = $derived(narrow ? 20 : 44);
	const colW = $derived((W - padL - 6) / T);
	const x0 = (t) => padL + colW * t;
	const barMax = $derived(colW * 0.8);
	const strip = $derived({ y: 18, cs: Math.min(narrow ? 15 : 20, (W - 40) / specimen.corpus.length) });
	const stripX0 = $derived((W - strip.cs * specimen.corpus.length) / 2);
	const yCtx = $derived(strip.y + 64);
	const ctxLines = $derived(narrow ? 2 : 1);
	const yBars = $derived(yCtx + 26 + (ctxLines - 1) * 14);
	const rowH = $derived(narrow ? 17 : 21);
	const H = $derived(yBars + rowH * V + 34);
	const fsCtx = $derived(narrow ? 11 : Math.min(14.5, colW / 6.2));

	let dragging = false, grab = 0;
	function svgPoint(svg, e) { const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY; return p.matrixTransform(svg.getScreenCTM().inverse()); }
	function stripDown(e) {
		const p = svgPoint(e.currentTarget.ownerSVGElement, e);
		const k = Math.floor((p.x - stripX0) / strip.cs);
		grab = k >= start && k < start + T ? k - start : Math.floor(T / 2);
		dragging = true;
		e.currentTarget.setPointerCapture(e.pointerId);
		moveTo(p.x);
	}
	function stripMove(e) { if (dragging) moveTo(svgPoint(e.currentTarget.ownerSVGElement, e).x); }
	function moveTo(x) { start = clamp(Math.round((x - stripX0) / strip.cs - grab - 0.5), 0, maxStart); }
	function key(e) {
		if (!interactive) return;
		if (e.key === 'ArrowLeft') { start = Math.max(0, start - 1); e.preventDefault(); }
		if (e.key === 'ArrowRight') { start = Math.min(maxStart, start + 1); e.preventDefault(); }
	}

	const describe = (t) => {
		const p = tr.probs[t];
		const ctx = tr.chars.slice(0, t + 1).map(showChar).join('');
		const top = p.map((v, a) => ({ v, a })).filter((o) => o.v >= 0.01).sort((u, w) => w.v - u.v).map((o) => `‘${showChar(specimen.vocab[o.a])}’ ${fmt(o.v)}`).join(', ');
		const nx = nextId(t);
		return `After “${ctx}”: ${top}.${nx >= 0 ? ` The text continues with ‘${showChar(specimen.vocab[nx])}’.` : ' The corpus ends here.'}`;
	};
	const readout = $derived(focus >= 0 ? describe(focus) : null);
	const split = (t) => {
		const s = tr.chars.slice(0, t + 1).map(showChar);
		if (ctxLines === 1 || s.length <= 4) return [s.join('')];
		return [s.slice(0, s.length - 4).join(''), s.slice(-4).join('')];
	};
</script>

<div class="fx fx-ctx" use:measure={(w) => (width = w)}>
	<svg
		viewBox="0 0 {W} {H}"
		width={W}
		height={H}
		role={interactive ? 'slider' : 'img'}
		tabindex={interactive ? 0 : undefined}
		aria-valuemin={interactive ? 1 : undefined}
		aria-valuemax={interactive ? maxStart + 1 : undefined}
		aria-valuenow={interactive ? start + 1 : undefined}
		aria-label="The model's next-character distributions for the prefixes of “{tr.text}”. {describe(T - 1)}"
		onkeydown={key}
	>
		<!-- the corpus, with the window -->
		<g class:is-live={interactive}>
			{#each Array.from(specimen.corpus) as ch, k (k)}
				<text class="tok tok--small" class:tok--out={k < start || k >= start + T} x={stripX0 + strip.cs * (k + 0.5)} y={strip.y + 10} text-anchor="middle">{showChar(ch)}</text>
			{/each}
			<path class="fx-reads__bracket" d="M {stripX0 + strip.cs * start + 2} {strip.y + 16} v 4 H {stripX0 + strip.cs * (start + T) - 2} v -4" />
			<text class="lbl lbl--micro" x={stripX0 + strip.cs * (start + T) + 6} y={strip.y + 21}>window</text>
			{#if interactive}
				<rect class="hit hit--drag" x={stripX0 - 6} y={strip.y - 6} width={strip.cs * specimen.corpus.length + 12} height="34" role="presentation" onpointerdown={stripDown} onpointermove={stripMove} onpointerup={() => (dragging = false)} />
			{/if}
		</g>

		<!-- character rows, labelled once on the left -->
		{#each specimen.vocab as c, a (a)}
			<text class="tok fx-ctx__row" x={padL - 12} y={yBars + rowH * (a + 0.5) + 4.5} text-anchor="end">{showChar(c)}</text>
			<line class="fx-ctx__grid" x1={padL} x2={W - 6} y1={yBars + rowH * (a + 1)} y2={yBars + rowH * (a + 1)} />
		{/each}

		{#each tr.probs as p, t (t)}
			{@const nx = nextId(t)}
			<g class:is-dim={focus >= 0 && focus !== t}>
				<text class="lbl lbl--micro" x={x0(t) + 4} y={yCtx - 22}><tspan class="m">t</tspan> = {t + 1}</text>
				{#each split(t) as line, li (li)}
					<text class="tok fx-ctx__ctx" x={x0(t) + 4} y={yCtx + li * 14} font-size={fsCtx}>{line}</text>
				{/each}
				<line class="axis" x1={x0(t) + 4} x2={x0(t) + 4} y1={yBars} y2={yBars + rowH * V} />
				{#each p as v, a (a)}
					{@const y = yBars + rowH * a}
					<rect class="bar--value" x={x0(t) + 4} y={y + rowH * 0.24} width={Math.max(0, v * barMax)} height={rowH * 0.52} />
					{#if a === nx}
						<rect class="fx-ctx__next" x={x0(t) + 1.5} y={y + rowH * 0.1} width={barMax + 5} height={rowH * 0.8} />
					{/if}
					{#if v >= 0.1}
						<text class="lbl lbl--micro fx-ctx__p" class:lbl--strong={true} x={x0(t) + 4 + v * barMax + (v > 0.78 ? -3 : 3)} y={y + rowH * 0.5 + 3.6} text-anchor={v > 0.78 ? 'end' : 'start'} class:fx-ctx__p--in={v > 0.78}>{fmt(v, 2).replace(/^0\./, '.')}</text>
					{/if}
				{/each}
			</g>
			{#if interactive}
				<rect class="hit" x={x0(t)} y={yCtx - 34} width={colW} height={yBars + rowH * V - yCtx + 34} role="presentation" onpointerenter={() => (focus = t)} onpointerleave={() => (focus = -1)} />
			{/if}
		{/each}
		<text class="lbl lbl--micro" x={padL + 4} y={H - 10}>outlined: the character that actually comes next</text>
	</svg>
	{#if interactive}
		<p class="readout" aria-live="polite">
			<span class="readout__part">{readout ?? describe(T - 1)}</span>
			<span class="readout__hint">Point at a column; drag the window along the line (or ←/→).</span>
		</p>
	{/if}
</div>

<style>
	.fx-ctx :global(.m) { font-family: var(--serif); font-style: italic; font-size: 12px; fill: var(--ink-3); }
	.fx-ctx__row { font-size: 15px; fill: var(--ink-2); }
	.fx-ctx__ctx { fill: var(--ink); white-space: pre; }
	.fx-ctx__grid { stroke: var(--rule-2); stroke-width: var(--hair); }
	.fx-ctx__next { fill: none; stroke: var(--ink); stroke-width: 0.8px; }
	.fx-ctx__p { font-size: 10px; }
	.fx-ctx__p--in { fill: var(--paper) !important; }
	.is-dim { opacity: 0.38; }
</style>

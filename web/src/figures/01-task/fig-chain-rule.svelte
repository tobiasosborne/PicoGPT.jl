<script module>
	export const meta = { size: 'wide', title: 'The chain rule as a staircase' };
</script>

<script>
	// Figure: how does the probability of a text factor into next-character predictions?
	// One training window: x₁ (given) and x₂ … x_{T+1}. Above each predicted character, a bar
	// of the factor q_θ(x_{t+1} | x_{≤t}) (trained θ). Below, the staircase of accumulated
	// surprise Σ −log q, which ends at −log q_θ(x₂…x_{T+1} | x₁): in value ink for the trained
	// specimen, in grey for θ₀ (before training), against the straight line of uniform
	// guessing, ln V per character. Right: the whole distribution at the chosen position.
	// Data: specimen.predictions(θ) and predictions(θ₀) on the training windows.
	import { getContext } from 'svelte';
	import { measure } from '../primitives/measure.js';
	import { fmt, showChar, clamp, layoutWidth } from '../primitives/scales.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const T = $derived(specimen.config.T);
	const V = $derived(specimen.config.V);
	const train = $derived(specimen.lab.train);
	const B = $derived(train.x.length);
	// svelte-ignore state_referenced_locally
	let b = $state(clamp((+attrs.window || 2) - 1, 0, specimen.lab.train.x.length - 1));
	// svelte-ignore state_referenced_locally
	let sel = $state(clamp((+attrs.pos || 1) - 1, 0, specimen.config.T - 1)); // which prediction is opened
	let width = $state(868);
	const interactive = $derived(mode !== 'print');

	const P1 = $derived(specimen.predictions(specimen.theta));
	const P0 = $derived(specimen.theta0 ? specimen.predictions(specimen.theta0) : null);
	const chars = $derived([train.x[b][0], ...train.y[b]].map((id) => showChar(specimen.vocab[id - 1])));
	const q = (P, t) => P.probs[b * T + t][train.y[b][t] - 1];
	const s1 = $derived(Array.from({ length: T }, (_, t) => -Math.log(q(P1, t))));
	const s0 = $derived(P0 ? Array.from({ length: T }, (_, t) => -Math.log(q(P0, t))) : null);
	const cum = (s) => s.reduce((acc, v) => [...acc, acc.at(-1) + v], [0]);
	const c1 = $derived(cum(s1));
	const c0 = $derived(s0 ? cum(s0) : null);
	const lnV = $derived(Math.log(V));
	const yTop = $derived(Math.max(T * lnV, c0?.at(-1) ?? 0, c1.at(-1)));

	// geometry
	const W = $derived(layoutWidth(width, mode, 330));
	const narrow = $derived(W < 660);
	const insetW = $derived(narrow ? 0 : 176);
	const padL = $derived(narrow ? 16 : 60);
	const step = $derived((W - padL - insetW - 26) / (T + 1));
	const xs = $derived(Array.from({ length: T + 1 }, (_, k) => padL + step * (k + 0.5)));
	const strip = $derived({ y: 18, cs: Math.min(narrow ? 15 : 20, (W - 40) / specimen.corpus.length) });
	const stripX0 = $derived((W - strip.cs * specimen.corpus.length) / 2);
	const hB = 54;
	const yBarBase = $derived(strip.y + 54 + hB);
	const fs = $derived(narrow ? 22 : 30);
	const yChar = $derived(yBarBase + fs + 6);
	const yPlot0 = $derived(yChar + 26); // top of the staircase panel
	const hP = $derived(narrow ? 130 : 168);
	const yBase = $derived(yPlot0 + hP); // 0 nats
	const yv = (v) => yBase - (v / yTop) * hP;
	const H = $derived(yBase + 40);
	// the staircase: position k is the boundary after character k+1
	const xEdge = (k) => xs[0] + step * 0.5 + step * k; // after x₁ … x_{k+1}
	const stairs = (c) => {
		let d = `M${xEdge(0)},${yv(0)}`;
		for (let t = 0; t < T; t++) d += ` H${xs[t + 1]} V${yv(c[t + 1])}`;
		return d + ` H${xEdge(T)}`;
	};

	// inset: the full distribution at the selected prediction
	const ix = $derived(W - insetW + 6);
	const iy = $derived(strip.y + 66);
	const ibar = $derived(narrow ? 0 : (insetW - 20) / V);
	const ih = 88;
	const pSel = $derived(P1.probs[b * T + sel]);
	const ySel = $derived(train.y[b][sel] - 1);
	const ctxSel = $derived(chars.slice(0, sel + 1).join(''));

	let dragging = false, grab = 0;
	const maxStart = $derived(B - 1);
	function svgPoint(svg, e) { const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY; return p.matrixTransform(svg.getScreenCTM().inverse()); }
	function stripDown(e) {
		const p = svgPoint(e.currentTarget.ownerSVGElement, e);
		const k = Math.floor((p.x - stripX0) / strip.cs);
		grab = k >= b && k <= b + T ? k - b : Math.floor(T / 2);
		dragging = true;
		e.currentTarget.setPointerCapture(e.pointerId);
		moveTo(p.x);
	}
	function stripMove(e) { if (dragging) moveTo(svgPoint(e.currentTarget.ownerSVGElement, e).x); }
	function moveTo(x) { b = clamp(Math.round((x - stripX0) / strip.cs - grab - 0.5), 0, maxStart); }
	function key(e) {
		if (!interactive) return;
		if (e.key === 'ArrowLeft') { sel = Math.max(0, sel - 1); e.preventDefault(); }
		if (e.key === 'ArrowRight') { sel = Math.min(T - 1, sel + 1); e.preventDefault(); }
		if (e.key === 'ArrowUp') { b = Math.max(0, b - 1); e.preventDefault(); }
		if (e.key === 'ArrowDown') { b = Math.min(maxStart, b + 1); e.preventDefault(); }
	}
	const total1 = $derived(c1.at(-1));
	const total0 = $derived(c0?.at(-1));
	const summary = $derived(`The trained specimen gives the window probability exp(−${fmt(total1, 3)}) = ${fmt(Math.exp(-total1), 3)}${P0 ? `; before training it gave exp(−${fmt(total0, 2)})` : ''}. Uniform guessing would give exp(−${fmt(T * lnV, 2)}).`);
</script>

<div class="fx fx-chain" use:measure={(w) => (width = w)}>
	<svg
		viewBox="0 0 {W} {H}"
		width={W}
		height={H}
		role={interactive ? 'group' : 'img'}
		tabindex={interactive ? 0 : undefined}
		aria-label="The chain rule on “{chars.join('')}”. {summary}"
		onkeydown={key}
	>
		<!-- the corpus: the window and its one extra character -->
		<g class:is-live={interactive}>
			{#each Array.from(specimen.corpus) as ch, k (k)}
				<text class="tok tok--small" class:tok--out={k < b || k > b + T} x={stripX0 + strip.cs * (k + 0.5)} y={strip.y + 10} text-anchor="middle">{showChar(ch)}</text>
			{/each}
			<path class="fx-reads__bracket" d="M {stripX0 + strip.cs * b + 2} {strip.y + 16} v 4 H {stripX0 + strip.cs * (b + T + 1) - 2} v -4" />
			<text class="lbl lbl--micro" x={stripX0 + strip.cs * (b + (T + 1) / 2)} y={strip.y + 34} text-anchor="middle">window {b + 1}, and the character after it</text>
			{#if interactive}
				<rect class="hit hit--drag" x={stripX0 - 6} y={strip.y - 6} width={strip.cs * specimen.corpus.length + 12} height="34" role="presentation" onpointerdown={stripDown} onpointermove={stripMove} onpointerup={() => (dragging = false)} />
			{/if}
		</g>

		<!-- factors q(x_{t+1} | x_{≤t}) -->
		{#each s1 as _, t (t)}
			{@const p = q(P1, t)}
			{@const x = xs[t + 1]}
			<rect class="fx-chain__frame" x={x - 7} y={yBarBase - hB} width="14" height={hB} />
			<rect class="bar--value" x={x - 7} y={yBarBase - p * hB} width="14" height={p * hB} />
			<text class="lbl lbl--micro" class:lbl--strong={p >= 0.1} x={x} y={yBarBase - hB - 6} text-anchor="middle">{fmt(p, 2)}</text>
		{/each}
		{#if !narrow}
			<text class="lbl lbl--micro" x={padL - 8} y={yBarBase - hB + 8} text-anchor="end">1</text>
			<text class="lbl lbl--micro" x={padL - 8} y={yBarBase + 3} text-anchor="end">0</text>
			<text class="lbl" x={padL - 8} y={yBarBase - hB / 2 + 4} text-anchor="end"><tspan class="m">q</tspan></text>
		{/if}

		<!-- the characters -->
		{#each chars as c, k (k)}
			<text class="tok fx-chain__char" class:is-given={k === 0} class:is-on={k === sel + 1} x={xs[k]} y={yChar} text-anchor="middle" font-size={fs}>{c}</text>
			<text class="lbl lbl--micro" x={xs[k] + fs * 0.34} y={yChar + 3}>{k + 1}</text>
			{#if interactive && k > 0}
				<rect class="hit" x={xs[k] - step / 2} y={yBarBase - hB - 18} width={step} height={yChar - yBarBase + hB + 26} role="button" tabindex="-1" aria-label="open the distribution at position {k}" onclick={() => (sel = k - 1)} onkeydown={() => {}} />
			{/if}
		{/each}
		<text class="lbl lbl--micro" x={xs[0]} y={yChar + 18} text-anchor="middle">given</text>

		<!-- the staircase of surprise -->
		<line class="axis" x1={xEdge(0)} x2={xEdge(T)} y1={yBase} y2={yBase} />
		<line class="axis" x1={xEdge(0)} x2={xEdge(0)} y1={yBase} y2={yPlot0} />
		{#each [0, 5, 10, 15].filter((v) => v <= yTop) as v (v)}
			<line class="axis" x1={xEdge(0) - 4} x2={xEdge(0)} y1={yv(v)} y2={yv(v)} />
			<text class="lbl lbl--micro" x={xEdge(0) - 7} y={yv(v) + 3.5} text-anchor="end">{v}</text>
		{/each}
		<text class="lbl lbl--micro" x={xEdge(0) + 6} y={yPlot0 + 2}>Σ −log <tspan class="m m--s">q</tspan>, nats</text>
		<line class="fx-chain__uniform" x1={xEdge(0)} y1={yv(0)} x2={xEdge(T)} y2={yv(T * lnV)} />
		<text class="lbl lbl--micro fx-chain__ulbl" x={xEdge(T * 0.62) + 6} y={yv(T * 0.62 * lnV) + 16}>{narrow ? 'uniform guessing' : `uniform guessing: ln ${V} per character`}</text>
		{#if c0}
			<path class="fx-chain__stairs0" d={stairs(c0)} />
			<text class="lbl lbl--micro" x={xEdge(T) - 2} y={yv(total0) - 7} text-anchor="end">before training: {fmt(total0, 2)}</text>
		{/if}
		<path class="fx-chain__stairs" d={stairs(c1)} />
		<text class="lbl lbl--value fx-chain__tlbl" x={xEdge(T) - 2} y={yv(total1) - 8} text-anchor="end">trained: {fmt(total1, 3)} nats</text>

		<!-- inset: the whole distribution at the opened position -->
		{#if !narrow}
			<g class="fx-chain__inset">
				<text class="lbl lbl--micro" x={ix} y={iy - 30}>at position {sel + 2}, given</text>
				<text class="tok fx-chain__ictx" x={ix} y={iy - 12}>{ctxSel}</text>
				<line class="axis" x1={ix} x2={ix + ibar * V} y1={iy + ih} y2={iy + ih} />
				{#each pSel as v, a (a)}
					{@const x = ix + ibar * (a + 0.5)}
					<rect class="bar--value" x={x - ibar * 0.3} y={iy + ih - v * ih} width={ibar * 0.6} height={v * ih} />
					{#if a === ySel}<rect class="fx-ctx__next" x={x - 7} y={iy + ih + 3} width="14" height="17" />{/if}
					{#if v >= 0.1}<text class="lbl lbl--micro lbl--strong" x={x} y={iy + ih - v * ih - 5} text-anchor="middle">{fmt(v, 2).replace(/^0\./, '.')}</text>{/if}
					<text class="tok fx-chain__ich" x={x} y={iy + ih + 15} text-anchor="middle">{showChar(specimen.vocab[a])}</text>
				{/each}
				<text class="lbl lbl--micro" x={ix} y={iy + ih + 36}>outlined: what came next</text>
			</g>
		{/if}
	</svg>
	{#if interactive}
		<p class="readout" aria-live="polite">
			<span class="readout__who">“{chars.join('')}”.</span>
			<span class="readout__part">{summary}</span>
			<span class="readout__hint">Click a character to open its distribution (←/→); drag the window along the line (↑/↓).</span>
		</p>
	{/if}
</div>

<style>
	.fx-chain :global(.m) { font-family: var(--serif); font-style: italic; font-size: 15px; fill: var(--ink); }
	.fx-chain :global(.m--s) { font-size: 12px; }
	.fx-chain__char { font-weight: 380; }
	.fx-chain__char.is-given { fill: var(--ink-3); }
	.fx-chain__char.is-on { font-weight: 640; }
	.fx-chain__frame { fill: none; stroke: var(--rule); stroke-width: var(--hair); }
	.fx-chain__uniform { stroke: var(--ink-3); stroke-width: var(--hair); }
	.fx-chain__ulbl { fill: var(--ink-2); }
	.fx-chain__stairs0 { fill: none; stroke: var(--ink-3); stroke-width: 0.9px; }
	.fx-chain__stairs { fill: none; stroke: var(--value); stroke-width: 2px; stroke-linejoin: miter; }
	.fx-chain__tlbl { font-weight: 600; }
	.fx-chain__ictx { font-size: 15px; white-space: pre; }
	.fx-chain__ich { font-size: 14px; }
	.fx-chain :global(.fx-ctx__next) { fill: none; stroke: var(--ink); stroke-width: 0.8px; }
</style>

<script module>
	export const meta = { size: 'wide', title: 'What enters the first block' };
</script>

<script>
	// Figure: X₀ = E[:, x] + P for one window. Three rows of spoke glyphs, one column per
	// position: the character's column of E, the positional code P_t, and their sum X₀[:, t]
	// (from the engine's forward trace). One scale for all three rows and every window, so
	// the window can be dragged without the glyphs changing size. Repeated characters in
	// the window are joined by a hairline above their glyphs.
	import { getContext } from 'svelte';
	import { measure } from '../primitives/measure.js';
	import SpokeGlyph from '../primitives/SpokeGlyph.svelte';
	import { fmt, showChar, clamp, maxAbs, spokeScale, layoutWidth } from '../primitives/scales.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const T = $derived(specimen.config.T);
	const D = $derived(specimen.config.D);
	const maxStart = $derived(specimen.windows.length - 1);
	// svelte-ignore state_referenced_locally
	let start = $state(clamp((+attrs.window || 1) - 1, 0, specimen.windows.length - 1));
	let focus = $state(-1);
	let width = $state(868);
	const interactive = $derived(mode !== 'print');

	const tr = $derived(specimen.windowTrace(start));
	const Ev = $derived(specimen.params().find((p) => p.name === 'E').values);
	const Ecol = (a) => Array.from({ length: D }, (_, i) => Ev[i + D * a]); // a 0-based
	const rowsE = $derived(tr.ids.map((id) => Ecol(id - 1)));
	const rowsP = $derived(specimen.P.slice(0, T));
	const rowsX = $derived(tr.X0);
	const vmax = $derived(maxAbs(Ev, specimen.P, specimen.windows.map((s) => specimen.windowTrace(s).X0)));

	// repeated characters in this window: groups of positions
	const repeats = $derived.by(() => {
		const g = new Map();
		tr.chars.forEach((c, t) => g.set(c, [...(g.get(c) ?? []), t]));
		return [...g.entries()].filter(([, ts]) => ts.length > 1);
	});

	// geometry
	const W = $derived(layoutWidth(width, mode, 330));
	const narrow = $derived(W < 600);
	const padL = $derived(narrow ? 14 : 112);
	const padR = $derived(narrow ? 8 : 20);
	const step = $derived((W - padL - padR) / T);
	const xs = $derived(Array.from({ length: T }, (_, t) => padL + step * (t + 0.5)));
	const rG = $derived(Math.min(narrow ? 19 : 32, step * 0.42));
	const sc = $derived(spokeScale(rG, vmax));
	const strip = $derived({ y: 18, cs: Math.min(narrow ? 15 : 20, (W - 40) / specimen.corpus.length) });
	const stripX0 = $derived((W - strip.cs * specimen.corpus.length) / 2);
	const fs = $derived(narrow ? 22 : 30);
	const yChar = $derived(strip.y + 74);
	const gap = $derived(2 * rG + (narrow ? 26 : 34));
	const yE = $derived(yChar + 26 + rG + 12);
	const yP = $derived(yE + gap);
	const yX = $derived(yP + gap);
	const H = $derived(yX + rG + 30);

	// dragging the window along the corpus (as in the attention figures)
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

	const dist = (u, v) => Math.hypot(...u.map((x, i) => x - v[i]));
	const readout = $derived.by(() => {
		if (focus >= 0) {
			const t = focus, c = showChar(tr.chars[t]);
			return `Position ${t + 1}, ‘${c}’: X₀ = E[:, ${tr.ids[t]}] + P${t + 1}, of length ${fmt(Math.hypot(...rowsX[t]), 2)} (the character’s column alone has ${fmt(Math.hypot(...rowsE[t]), 2)}, the position’s ${fmt(Math.hypot(...rowsP[t]), 2)}).`;
		}
		if (!repeats.length) return `No character repeats in “${tr.text}”.`;
		return repeats.map(([c, ts]) => `‘${showChar(c)}’ at positions ${ts.map((t) => t + 1).join(' and ')}: one column of E, but X₀ columns ${fmt(dist(rowsX[ts[0]], rowsX[ts[1]]), 2)} apart`).join('; ') + '.';
	});
</script>

<div class="fx fx-x0" use:measure={(w) => (width = w)}>
	<!-- svelte-ignore a11y_no_noninteractive_tabindex (composite figure supports arrow-key navigation) -->
	<svg
		viewBox="0 0 {W} {H}"
		width={W}
		height={H}
		role={interactive ? 'group' : 'img'}
		tabindex={interactive ? 0 : undefined}
		aria-label="X₀ for the window “{tr.text}”. {readout}"
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

		<!-- repeated characters: a hairline joining their positions -->
		{#each repeats as [c, ts], g (c)}
			{@const y0 = yChar - fs * 0.95}
			{@const y1 = y0 - 6 - 7 * g}
			<path class="fx-x0__same" d="M{xs[ts[0]]},{y0} V{y1} H{xs[ts.at(-1)]} V{y0}" />
			{#each ts.slice(1, -1) as t (t)}<line class="fx-x0__same" x1={xs[t]} x2={xs[t]} y1={y1} y2={y0} />{/each}
		{/each}

		{#each tr.chars as ch, t (t)}
			<text class="tok fx-x0__char" class:is-on={t === focus} x={xs[t]} y={yChar} text-anchor="middle" font-size={fs}>{showChar(ch)}</text>
			<text class="lbl lbl--micro" x={xs[t] + fs * 0.34} y={yChar + 3}>{t + 1}</text>
			<SpokeGlyph values={rowsE[t]} x={xs[t]} y={yE} rmax={sc.rmax} scale={sc.scale} tip={narrow ? 1.6 : 2.1} />
			<SpokeGlyph values={rowsP[t]} x={xs[t]} y={yP} rmax={sc.rmax} scale={sc.scale} tip={narrow ? 1.6 : 2.1} />
			<SpokeGlyph values={rowsX[t]} x={xs[t]} y={yX} rmax={sc.rmax} scale={sc.scale} weight={1.35} tip={narrow ? 1.9 : 2.5} />
			{#if interactive}
				<rect class="hit" x={xs[t] - step / 2} y={yChar - fs} width={step} height={yX + rG - yChar + fs} role="button" aria-label="read this entry; details in the live readout" tabindex="0" onpointerenter={() => (focus = t)} onfocus={() => (focus = t)} onpointerleave={() => (focus = -1)} onblur={() => (focus = -1)} onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); (focus = t); } }} />
			{/if}
		{/each}

		<!-- row labels and operators -->
		{#if !narrow}
			{#each [[yE, 'E', '[:, xₜ]', 'the character'], [yP, 'P', 'ₜ', 'the position'], [yX, 'X', '₀[:, t]', 'their sum']] as [y, sym, rest, word] (sym)}
				<text class="lbl" x={padL - 30} y={y - 2} text-anchor="end"><tspan class="m">{sym}</tspan><tspan class="m m--s">{rest}</tspan></text>
				<text class="lbl lbl--micro" x={padL - 30} y={y + 13} text-anchor="end">{word}</text>
			{/each}
		{/if}
		<text class="fx-x0__op" x={narrow ? 8 : padL - 12} y={(yE + yP) / 2 + 6} text-anchor="middle">+</text>
		<text class="fx-x0__op" x={narrow ? 8 : padL - 12} y={(yP + yX) / 2 + 6} text-anchor="middle">=</text>
		<line class="rule" x1={padL} x2={W - padR} y1={(yP + yX) / 2} y2={(yP + yX) / 2} />
	</svg>
	{#if interactive}
		<p class="readout" aria-live="polite">
			<span class="readout__who">“{tr.text}”.</span>
			<span class="readout__part">{readout}</span>
			<span class="readout__hint">Drag the window along the line (or ←/→); point at a column.</span>
		</p>
	{/if}
</div>

<style>
	.fx-x0 :global(.m) { font-family: var(--serif); font-style: italic; font-size: 15px; fill: var(--ink); }
	.fx-x0 :global(.m--s) { font-size: 13px; }
	.fx-x0__char { font-weight: 380; }
	.fx-x0__char.is-on { font-weight: 640; }
	.fx-x0__op { font-family: var(--serif); font-size: 20px; fill: var(--ink-3); }
	.fx-x0__same { fill: none; stroke: var(--ink-3); stroke-width: var(--hair); }
</style>

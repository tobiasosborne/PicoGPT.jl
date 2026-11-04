<script module>
	export const meta = { size: 'wide', title: 'What does each character read?' };
	export { default as Key } from './fig-attention-reads.key.svelte';
</script>

<script>
	// Figure: what does each character read?
	// State: window start s (corpus position), reader i. Attrs (1-based, as in the
	// prose): window=<s+1>, reader=<i+1>. Encoding: one arc per source j ≤ i, width ∝ A_ij;
	// head 1 above the line, head 2 below. Interaction: drag the window along the
	// corpus; click (or ←/→) to choose the reader; hover previews.
	import { getContext } from 'svelte';
	import { measure } from '../primitives/measure.js';
	import Arcs from '../primitives/Arcs.svelte';
	import { fmt, showChar, clamp, layoutWidth } from '../primitives/scales.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const T = $derived(specimen.config.T);
	const maxStart = $derived(specimen.corpus.length - T);

	// svelte-ignore state_referenced_locally
	let start = $state(clamp((+attrs.window || 11) - 1, 0, 10));
	// svelte-ignore state_referenced_locally
	let reader = $state(clamp((+attrs.reader || 8) - 1, 0, 7));
	let preview = $state(-1);
	let width = $state(868);
	const interactive = $derived(mode !== 'print');

	const tr = $derived(specimen.windowTrace(start));
	const W = $derived(layoutWidth(width, mode));
	const narrow = $derived(W < 560);

	// geometry
	const fs = $derived(narrow ? 30 : 42);
	const padL = $derived(narrow ? 10 : 64);
	const padR = $derived(narrow ? 10 : 24);
	const step = $derived((W - padL - padR) / T);
	const xs = $derived(Array.from({ length: T }, (_, j) => padL + step * (j + 0.5)));
	const strip = $derived({ y: 16, cs: Math.min(narrow ? 15 : 21, (W - 40) / specimen.corpus.length) });
	const stripX0 = $derived((W - strip.cs * specimen.corpus.length) / 2);
	const aspect = $derived(narrow ? 0.5 : 0.56);
	// Room for each head = its longest arc that carries visible weight (A ≥ 0.02) in ANY
	// window: stable while the reader drags, but no space kept for arcs never drawn.
	const reach = $derived.by(() => {
		const r = [1, 1];
		for (const s of specimen.windows) {
			const w = specimen.windowTrace(s);
			w.heads.forEach((h, k) => h.A.forEach((row, i) => row.forEach((a, j) => { if (j <= i && a >= 0.02) r[k] = Math.max(r[k], i - j); })));
		}
		return r;
	});
	const hUp = $derived((reach[0] * step) / 2 * aspect + 18);
	const hDown = $derived((reach[1] * step) / 2 * aspect + 18);
	const yBase = $derived(strip.y + 30 + hUp + 18 + fs * 0.7);
	const yUp = $derived(yBase - fs * 0.8);
	const yDown = $derived(yBase + fs * 0.26);
	const H = $derived(yDown + hDown + 30);
	const wmax = $derived(narrow ? 9 : 13);

	const shown = $derived(preview >= 0 ? preview : reader);

	function pick(i) { reader = i; preview = -1; }
	function key(e) {
		if (!interactive) return;
		if (e.key === 'ArrowLeft') { reader = Math.max(0, reader - 1); e.preventDefault(); }
		if (e.key === 'ArrowRight') { reader = Math.min(T - 1, reader + 1); e.preventDefault(); }
	}

	// dragging the window along the corpus
	let dragging = false;
	let grab = 0;
	function stripDown(e) {
		if (!interactive) return;
		const svg = e.currentTarget.ownerSVGElement;
		const p = svgPoint(svg, e);
		const k = Math.floor((p.x - stripX0) / strip.cs);
		grab = clamp(k - start, 0, T - 1);
		if (k < start || k >= start + T) grab = Math.floor(T / 2);
		dragging = true;
		e.currentTarget.setPointerCapture(e.pointerId);
		moveTo(p.x);
	}
	function stripMove(e) { if (dragging) moveTo(svgPoint(e.currentTarget.ownerSVGElement, e).x); }
	function stripUp() { dragging = false; }
	function moveTo(x) { start = clamp(Math.round((x - stripX0) / strip.cs - grab - 0.5), 0, maxStart); }
	function svgPoint(svg, e) {
		const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
		return pt.matrixTransform(svg.getScreenCTM().inverse());
	}

	const readout = $derived.by(() => {
		const i = reader;
		const parts = tr.heads.map((h, k) => {
			const row = h.A[i].slice(0, i + 1).map((w, j) => ({ w, j })).sort((a, b) => b.w - a.w).slice(0, 3);
			return `head ${k + 1} reads ${row.map((r) => `‘${showChar(tr.chars[r.j])}’${r.j === i ? ' (itself)' : ''} ${fmt(r.w)}`).join(', ')}`;
		});
		return { who: `‘${showChar(tr.chars[i])}’, position ${i + 1} of “${tr.text}”`, parts };
	});
</script>

<div class="fx fx-reads" use:measure={(w) => (width = w)}>
	{#snippet body()}
		<!-- the corpus, with the window -->
		<g class="fx-reads__strip" class:is-live={interactive}>
			{#each Array.from(specimen.corpus) as ch, k (k)}
				<text class="tok tok--small" class:tok--out={k < start || k >= start + T} x={stripX0 + strip.cs * (k + 0.5)} y={strip.y + 10} text-anchor="middle">{showChar(ch)}</text>
			{/each}
			<path class="fx-reads__bracket" d="M {stripX0 + strip.cs * start + 2} {strip.y + 16} v 4 H {stripX0 + strip.cs * (start + T) - 2} v -4" />
			<text class="lbl lbl--micro" x={stripX0 + strip.cs * (start + T) + 6} y={strip.y + 21}>window</text>
			{#if interactive}
				<rect class="hit hit--drag" x={stripX0 - 6} y={strip.y - 6} width={strip.cs * specimen.corpus.length + 12} height="34" onpointerdown={stripDown} onpointermove={stripMove} onpointerup={stripUp} role="slider" tabindex="-1" aria-label="window position" aria-valuenow={start + 1} aria-valuemin="1" aria-valuemax={maxStart + 1} />
			{/if}
		</g>

		<!-- direct labels for the two heads -->
		{#if !narrow}
			<text class="lbl lbl--head" x={4} y={yUp - 18}>head 1</text>
			<text class="lbl lbl--head" x={4} y={yDown + 30}>head 2</text>
		{/if}

		<!-- arcs: head 1 above, head 2 below -->
		{#if preview >= 0 && preview !== reader}
			<Arcs {xs} y={yUp} A={tr.heads[0].A} reader={reader} dir={-1} {wmax} {aspect} muted />
			<Arcs {xs} y={yDown} A={tr.heads[1].A} reader={reader} dir={1} {wmax} {aspect} muted />
		{/if}
		<Arcs {xs} y={yUp} A={tr.heads[0].A} reader={shown} dir={-1} {wmax} {aspect} />
		<Arcs {xs} y={yDown} A={tr.heads[1].A} reader={shown} dir={1} {wmax} {aspect} />

		<!-- the window's characters -->
		{#each tr.chars as ch, j (j)}
			<text
				class="tok tok--big"
				class:tok--reader={j === shown}
				class:tok--future={j > shown}
				x={xs[j]}
				y={yBase}
				text-anchor="middle"
				font-size={fs}>{showChar(ch)}</text
			>
		{/each}
		{#each tr.chars as _, j (j)}
			<text class="lbl lbl--micro lbl--pos" x={xs[j] + fs * 0.34} y={yBase + 3} text-anchor="start">{j + 1}</text>
		{/each}
		{#if shown < T - 1}
			<text class="lbl lbl--micro lbl--future" x={(xs[shown + 1] + xs[T - 1]) / 2} y={yBase + fs * 0.62} text-anchor="middle">future · masked</text>
		{/if}

		{#if interactive}
			{#each tr.chars as _, j (j)}
				<rect
					class="hit"
					x={xs[j] - step / 2}
					y={yBase - fs}
					width={step}
					height={fs * 1.3}
					role="button"
					tabindex="-1"
					aria-label="read from position {j + 1}"
					onpointerenter={() => (preview = j)}
					onpointerleave={() => (preview = -1)}
					onclick={() => pick(j)}
					onkeydown={() => {}}
				/>
			{/each}
		{/if}
	{/snippet}
	{#if interactive}
		<svg viewBox="0 0 {W} {H}" width={W} height={H} role="slider" tabindex="0" aria-label="Reader position (arrow keys)" aria-valuemin="1" aria-valuemax={T} aria-valuenow={reader + 1} aria-valuetext="reader {readout.who}: {readout.parts.join('; ')}" onkeydown={key}>
			{@render body()}
		</svg>
	{:else}
		<svg viewBox="0 0 {W} {H}" width={W} height={H} role="img" aria-label="Attention arcs for the reader {readout.who}: {readout.parts.join('; ')}">
			{@render body()}
		</svg>
	{/if}
	{#if interactive}
		<p class="readout" aria-live="polite">
			<span class="readout__who">Reader {readout.who}.</span>
			{#each readout.parts as p, k (k)}<span class="readout__part">{p}{k < readout.parts.length - 1 ? ';' : '.'}</span>{/each}
			<span class="readout__hint">Click a character to change the reader; drag the window along the line.</span>
		</p>
	{/if}
</div>

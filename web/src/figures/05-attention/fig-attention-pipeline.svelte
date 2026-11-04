<script module>
	export const meta = { size: 'wide', title: 'From query and keys to the mixed value' };
	export { default as Key } from './fig-attention-pipeline.key.svelte';
</script>

<script>
	// Figure: one reader i, one head h, top to bottom (forward = down):
	//   keys k_j (spoke glyphs)        query q_i (the reader's column, right)
	//   scores S_ij = q_i·k_j/√d       (lollipops about zero)
	//   weights A_ij                   (bars; Σ_j = 1)        temperature τ (drag)
	//   values v_j, and A_ij v_j       output o_i = Σ_j A_ij v_j (right)
	// At τ = √d every number is the model's own. Dragging τ recomputes the Gibbs
	// weights exp(q·k/τ)/Z and the output from the same q, k, v.
	import { getContext } from 'svelte';
	import { measure } from '../primitives/measure.js';
	import SpokeGlyph from '../primitives/SpokeGlyph.svelte';
	import { fmt, fmtProb, showChar, clamp, maxAbs, spokeScale, layoutWidth } from '../primitives/scales.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const T = $derived(specimen.config.T);
	const d = $derived(specimen.d);
	const sqrtD = $derived(Math.sqrt(d));
	// svelte-ignore state_referenced_locally
	const start = clamp((+attrs.window || 10) - 1, 0, specimen.windows.length - 1);
	// svelte-ignore state_referenced_locally
	let reader = $state(clamp((+attrs.reader || 8) - 1, 0, 7));
	// svelte-ignore state_referenced_locally
	let head = $state(clamp((+attrs.head || 1) - 1, 0, 1));
	let tauSet = $state(null); // null = the model's own temperature √d
	const tau = $derived(tauSet ?? Math.sqrt(specimen.d));
	let width = $state(868);
	const interactive = $derived(mode !== 'print');

	const tr = $derived(specimen.windowTrace(start));
	const hd = $derived(tr.heads[head]);
	const i = $derived(reader);
	const q = $derived(hd.q[i]);
	const n = $derived(i + 1); // sources j = 0..i
	const dots = $derived(hd.k.slice(0, n).map((k) => k.reduce((s, kr, r) => s + kr * q[r], 0)));
	const S = $derived(dots.map((x) => x / sqrtD)); // the model's scores
	const weights = $derived.by(() => {
		const z = dots.map((x) => x / tau);
		const m = Math.max(...z);
		const e = z.map((x) => Math.exp(x - m));
		const Z = e.reduce((a, b) => a + b, 0);
		return e.map((x) => x / Z);
	});
	const out = $derived(Array.from({ length: d }, (_, r) => weights.reduce((s, w, j) => s + w * hd.v[j][r], 0)));
	const atModel = $derived(Math.abs(tau - sqrtD) < 1e-9);

	// common scales
	const W = $derived(layoutWidth(width, mode, 340));
	const narrow = $derived(W < 640);
	const L = $derived(narrow ? 8 : 96);
	const R = $derived(narrow ? 92 : 150);
	const step = $derived((W - L - R) / T);
	const xs = $derived(Array.from({ length: T }, (_, j) => L + step * (j + 0.5)));
	const xR = $derived(W - R / 2 + (narrow ? 6 : 10));
	const rG = $derived(Math.min(30, step * 0.4));
	const qk = $derived(spokeScale(rG, maxAbs(q, hd.k.slice(0, T))));
	const vo = $derived(spokeScale(rG, maxAbs(hd.v.slice(0, T), out)));
	const sMax = $derived(Math.max(1e-9, maxAbs(hd.S.map((r, a) => r.slice(0, a + 1)))));

	const Y = { char: 30, key: 100, score: 232, weight: 348, value: 470 };
	const sAmp = 46; // px for the largest score in this head and window
	const bH = 70; // px for weight 1
	const H = 548;

	// temperature axis (log), in the reader's column
	const tauMin = 0.25, tauMax = 16;
	const tY0 = Y.score - 40, tY1 = Y.weight + 30; // top = hot, bottom = cold
	const tauToY = (t) => tY1 - ((Math.log(t) - Math.log(tauMin)) / (Math.log(tauMax) - Math.log(tauMin))) * (tY1 - tY0);
	const yToTau = (y) => Math.exp(Math.log(tauMin) + ((tY1 - clamp(y, tY0, tY1)) / (tY1 - tY0)) * (Math.log(tauMax) - Math.log(tauMin)));
	let dragging = $state(false);
	function svgPoint(svg, e) {
		const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
		return pt.matrixTransform(svg.getScreenCTM().inverse());
	}
	function tDown(e) { dragging = true; e.currentTarget.setPointerCapture(e.pointerId); tMove(e); }
	function tMove(e) {
		if (!dragging) return;
		let t = yToTau(svgPoint(e.currentTarget.ownerSVGElement, e).y);
		if (Math.abs(Math.log(t / sqrtD)) < 0.06) t = sqrtD; // snap to the model's temperature
		tauSet = t;
	}
	function tUp() { dragging = false; }
	function tKey(e) {
		const f = e.shiftKey ? 1.5 : 1.12;
		if (e.key === 'ArrowUp') { tauSet = Math.min(tauMax, tau * f); e.preventDefault(); }
		if (e.key === 'ArrowDown') { tauSet = Math.max(tauMin, tau / f); e.preventDefault(); }
		if (e.key === 'Home' || e.key === 'Enter' || e.key === '0') { tauSet = null; e.preventDefault(); }
	}

	const readout = $derived.by(() => {
		const best = weights.map((w, j) => ({ w, j })).sort((a, b) => b.w - a.w).slice(0, 3);
		const H = -weights.reduce((s, w) => s + (w > 0 ? w * Math.log(w) : 0), 0);
		return {
			who: `‘${showChar(tr.chars[i])}’, position ${i + 1}, head ${head + 1}`,
			temp: atModel ? `τ = √d = ${fmt(sqrtD, 0)}, the model’s own temperature` : `τ = ${fmt(tau, 2)} (the model uses √d = ${fmt(sqrtD, 0)})`,
			reads: best.map((b) => `‘${showChar(tr.chars[b.j])}’ at ${b.j + 1} ${fmtProb(b.w, 2, true)}`).join(', '),
			entropy: `entropy ${fmt(H, 2)} of a possible ${fmt(Math.log(n), 2)} nats`
		};
	});
	const ticks = [0.25, 0.5, 1, 2, 4, 8, 16];
</script>

<div class="fx fx-pipe" use:measure={(w) => (width = w)}>
	<svg viewBox="0 0 {W} {H}" width={W} height={H} role={interactive ? 'group' : 'img'} aria-label="Attention for reader {readout.who}: weights {readout.reads}">
		<!-- row labels (direct labelling) -->
		{#if !narrow}
			<g class="fx-pipe__rows">
				{#each [['keys', 'k', 'j', Y.key], ['scores', 'S', 'ij', Y.score], ['weights', 'A', 'ij', Y.weight], ['values', 'v', 'j', Y.value]] as [word, sym, sub, y] (word)}
					<text class="lbl" x={L - 14} y={y - 3} text-anchor="end">{word}</text>
					<text class="lbl" x={L - 14} y={y + 13} text-anchor="end"><tspan class="m">{sym}</tspan><tspan class="m sub" dy="3">{sub}</tspan></text>
				{/each}
			</g>
		{/if}

		<!-- heads: click the name -->
		<g class="fx-pipe__heads">
			{#each [0, 1] as h (h)}
				{#if interactive}
					<text
						class="lbl fx-pipe__head"
						class:is-on={h === head}
						x={Math.min(W - 24, xR + (h ? 30 : -30))}
						y={Y.char - 4}
						text-anchor="middle"
						role="button"
						tabindex="0"
						aria-pressed={h === head}
						onclick={() => (head = h)}
						onkeydown={(e) => (e.key === 'Enter' || e.key === ' ') && (head = h)}>head {h + 1}</text
					>
				{:else if h === head}
					<text class="lbl fx-pipe__head is-on" x={xR} y={Y.char - 4} text-anchor="middle">head {h + 1}</text>
				{/if}
			{/each}
		</g>

		<!-- characters: sources and the reader -->
		{#each tr.chars as ch, j (j)}
			<text class="tok fx-pipe__char" class:tok--reader={j === i} class:tok--future={j > i} x={xs[j]} y={Y.char + 14} text-anchor="middle">{showChar(ch)}</text>
			{#if interactive}
				<rect class="hit" x={xs[j] - step / 2} y={Y.char - 10} width={step} height="34" role="button" tabindex="0" aria-label="reader at position {j + 1}" onclick={() => (reader = j)} onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); (reader = j); } }} />
			{/if}
		{/each}

		<!-- keys and the query -->
		{#each hd.k.slice(0, n) as k, j (j)}
			<SpokeGlyph values={k} x={xs[j]} y={Y.key} rmax={rG} scale={qk.scale} />
		{/each}
		<SpokeGlyph values={q} x={xR} y={Y.key} rmax={rG} scale={qk.scale} weight={1.35} tip={2.6} />
		<text class="lbl" x={xR} y={Y.key + rG + 20} text-anchor="middle">query <tspan class="m">q</tspan><tspan class="m sub" dy="3">{i + 1}</tspan></text>

		<!-- scores: lollipops about zero (up = larger score = lower energy) -->
		<line class="axis" x1={xs[0] - step * 0.45} x2={xs[n - 1] + step * 0.45} y1={Y.score} y2={Y.score} />
		<text class="lbl lbl--micro" x={xs[0] - step * 0.45 - 4} y={Y.score + 3.5} text-anchor="end">0</text>
		{#each S as s, j (j)}
			{@const y = Y.score - (s / sMax) * sAmp}
			<line class="stem" x1={xs[j]} x2={xs[j]} y1={Y.score} y2={y} />
			<circle class={s >= 0 ? 'dot--value' : 'dot--open'} cx={xs[j]} cy={y} r="3.2" />
			<text class="lbl lbl--micro lbl--num" x={xs[j]} y={s >= 0 ? y - 8 : y + 15} text-anchor="middle">{fmt(s, 2)}</text>
		{/each}

		<!-- weights: bars, Σ = 1 -->
		<line class="rule" x1={xs[0] - step * 0.45} x2={xs[n - 1] + step * 0.45} y1={Y.weight + 30} y2={Y.weight + 30} />
		{#each weights as w, j (j)}
			{@const h = w * bH}
			<rect class="bar--value" x={xs[j] - Math.min(9, step * 0.16)} y={Y.weight + 30 - h} width={Math.min(18, step * 0.32)} height={h} />
			<text class="lbl lbl--micro lbl--num" class:lbl--strong={w >= 0.1} x={xs[j]} y={Y.weight + 30 - h - 6} text-anchor="middle">{fmtProb(w, 2)}</text>
		{/each}

		<!-- temperature: drag along the log axis in the reader's column -->
		<g class="fx-pipe__tau" class:is-live={interactive}>
			<line class="axis" x1={xR} x2={xR} y1={tY0} y2={tY1} />
			{#each ticks as t (t)}
				<line class="axis" x1={xR - 4} x2={xR} y1={tauToY(t)} y2={tauToY(t)} />
				<text class="lbl lbl--micro" x={xR - 8} y={tauToY(t) + 3.5} text-anchor="end">{t < 1 ? fmt(t, 2).replace(/^0/, '') : t}</text>
			{/each}
			<line class="fx-pipe__model" x1={xR} x2={xR + 16} y1={tauToY(sqrtD)} y2={tauToY(sqrtD)} />
			<text class="lbl lbl--micro" x={xR + 20} y={tauToY(sqrtD) + 3.5}>√d</text>
			<text class="lbl lbl--micro" x={xR} y={tY0 - 9} text-anchor="middle">hot</text>
			<text class="lbl lbl--micro" x={xR} y={tY1 + 16} text-anchor="middle">cold</text>
			<g transform="translate({xR} {tauToY(tau)})">
				<circle class="handle" class:handle--active={dragging} r="6.5" />
				<text class="lbl fx-pipe__taulabel" x="-12" y="-10" text-anchor="end"><tspan class="m">τ</tspan> = {fmt(tau, 2)}</text>
			</g>
			{#if interactive}
				<rect
					class="handle-hit"
					x={xR - 26}
					y={tY0 - 8}
					width="52"
					height={tY1 - tY0 + 16}
					role="slider"
					tabindex="0"
					aria-label="temperature τ"
					aria-valuemin={tauMin}
					aria-valuemax={tauMax}
					aria-valuenow={tau}
					aria-valuetext="τ = {fmt(tau, 2)}"
					onpointerdown={tDown}
					onpointermove={tMove}
					onpointerup={tUp}
					onkeydown={tKey}
					ondblclick={() => (tauSet = null)}
				/>
			{/if}
		</g>

		<!-- values: full v_j (muted) and its share A_ij v_j (solid); the output on the right -->
		{#each hd.v.slice(0, n) as v, j (j)}
			<SpokeGlyph values={v} x={xs[j]} y={Y.value} rmax={rG} scale={vo.scale} muted />
			<SpokeGlyph values={v.map((x) => x * weights[j])} x={xs[j]} y={Y.value} rmax={rG} scale={vo.scale} compass={false} tip={1.5 + 1.3 * Math.sqrt(weights[j])} />
		{/each}
		<text class="lbl fx-pipe__sum" x={(xs[n - 1] + xR) / 2 + 4} y={Y.value + 5} text-anchor="middle">Σ →</text>
		<SpokeGlyph values={out} x={xR} y={Y.value} rmax={rG} scale={vo.scale} weight={1.35} tip={2.6} />
		<text class="lbl" x={xR} y={Y.value + rG + 22} text-anchor="middle">output</text>
		<text class="lbl" x={xR} y={Y.value + rG + 38} text-anchor="middle"><tspan class="m">o</tspan><tspan class="m sub" dy="3">{i + 1}</tspan><tspan dy="-3"> = Σ </tspan><tspan class="m">A</tspan><tspan class="m sub" dy="3">{i + 1}j</tspan><tspan dy="-3"> </tspan><tspan class="m">v</tspan><tspan class="m sub" dy="3">j</tspan></text>
	</svg>
	{#if interactive}
		<p class="readout" aria-live="polite">
			<span class="readout__who">Reader {readout.who}.</span>
			<span class="readout__part">At {readout.temp}, it reads {readout.reads}; {readout.entropy}.</span>
			<span class="readout__hint">Drag τ (double-click returns it to √d). Click a character to change the reader, or a head’s name.</span>
		</p>
	{/if}
</div>

<style>
	.fx-pipe :global(.m) { font-family: var(--serif); font-style: italic; font-size: 14px; fill: var(--ink); }
	.fx-pipe :global(.sub) { font-size: 10px; }
	.fx-pipe__char { font-size: 22px; }
	.fx-pipe__head { cursor: pointer; fill: var(--ink-3); font-weight: 500; }
	.fx-pipe__head.is-on { fill: var(--ink); font-weight: 650; text-decoration: underline; text-underline-offset: 3px; }
	.fx-pipe__model { stroke: var(--ink); stroke-width: 1px; }
	.fx-pipe__taulabel { fill: var(--ink); font-size: 12px; }
	.fx-pipe__sum { fill: var(--ink-3); font-size: 13px; }
	.fx-pipe :global(.lbl--num) { fill: var(--ink-2); }
	:global([data-mode='print']) .fx-pipe__head:not(.is-on) { display: none; }
	:global([data-mode='print']) .fx-pipe__head.is-on { text-decoration: none; }
</style>

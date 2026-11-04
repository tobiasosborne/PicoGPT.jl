<script module>
	export const meta = { size: 'wide', title: 'LayerNorm, geometrically' };
	export { default as Key } from './fig-layernorm-sphere.key.svelte';
</script>

<script>
	// Figure: the second LayerNorm of the block, LN₂(x) = γ ⊙ x̂ + β with x̂ = Πx/σ,
	// on every column x of one window (x = X_mid, the stream before the MLP).
	//
	// Left, the meridian plane. A vector x ∈ ℝᴰ is drawn at (a, r) with
	//   a = 𝟏ᵀx/√D   (its component along the unit vector 𝟏/√D),
	//   r = ‖Πx‖     (its distance from the 𝟏 axis),
	// so ‖x‖² = a² + r² and the circle of radius √D is the sphere ‖x‖ = √D.
	// Centring (Π) moves a point horizontally onto the vertical axis; scaling (÷σ)
	// moves it along that axis to √D·(1 − ε/σ²)^½, just inside the circle; the
	// diagonal affine map γ⊙·+β then moves it off again. Every position is exact.
	// The inset magnifies the landing points under the top of the circle.
	// Right: the selected column as a spoke glyph at each of the four stages.
	// Clock t ∈ [0, 1]: x (0) → centred (⅓) → scaled (⅔) → affine (1).
	import { getContext } from 'svelte';
	import { measure } from '../primitives/measure.js';
	import SpokeGlyph from '../primitives/SpokeGlyph.svelte';
	import Scrubber from '../primitives/Scrubber.svelte';
	import { fmt, showChar, clamp, maxAbs, spokeScale, layoutWidth } from '../primitives/scales.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const T = $derived(specimen.config.T);
	const D = $derived(specimen.config.D);
	const eps = $derived(specimen.config.eps ?? 1e-5);
	const B = $derived(specimen.lab.train.x.length);
	// svelte-ignore state_referenced_locally
	const win = clamp((+attrs.window || 1) - 1, 0, 9);
	// svelte-ignore state_referenced_locally
	let col = $state(clamp((+attrs.column || 4) - 1, 0, 7));
	// svelte-ignore state_referenced_locally
	let t = $state(attrs.t != null ? +attrs.t : 1);
	$effect(() => { if (attrs.t != null) t = +attrs.t; });
	let width = $state(868);
	const interactive = $derived(mode !== 'print');
	const W = $derived(layoutWidth(width, mode));
	const narrow = $derived(W < 640);

	const b = $derived(Math.min(win, B - 1));
	const gt = $derived(specimen.lab.gradTrace(specimen.theta));
	const w = $derived(gt.windows[b]);
	const chars = $derived(Array.from(specimen.corpus.slice(b, b + T)));
	const prm = $derived(Object.fromEntries(specimen.params().map((p) => [p.name.split('.').at(-1), p.values])));
	const gamma = $derived(prm.gamma2), beta = $derived(prm.beta2);

	const sum = (v) => v.reduce((s, x) => s + x, 0);
	const norm = (v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));
	/** the four stages of one column, exactly as the code computes them */
	function stages(x) {
		const mu = sum(x) / D;
		const u = x.map((v) => v - mu);
		const sigma = Math.sqrt(norm(u) ** 2 / D + eps);
		const xh = u.map((v) => v / sigma);
		const y = xh.map((v, i) => gamma[i] * v + beta[i]);
		return { x, mu, u, sigma, xh, y, gap: Math.sqrt(D) - norm(xh) };
	}
	const cols = $derived(w.val.Xmid.map(stages));
	const sel = $derived(cols[col]);

	/** the vector at time t: linear in t inside each stage (the path of the data) */
	function at(c, t) {
		const s = clamp(t, 0, 1) * 3;
		if (s <= 1) return c.x.map((v) => v - s * c.mu);
		if (s <= 2) { const k = 1 + (s - 1) * (1 / c.sigma - 1); return c.u.map((v) => v * k); }
		const q = s - 2;
		return c.xh.map((v, i) => v + q * (c.y[i] - v));
	}
	const merid = (v) => { const a = sum(v) / Math.sqrt(D); const m = sum(v) / D; return { a, r: norm(v.map((x) => x - m)) }; };
	/** the polyline of a column through all four stages, sampled finely in the affine stage */
	function trail(c, upto) {
		const pts = [];
		const n = 24;
		for (let k = 0; k <= 3 * n; k++) {
			const tt = k / (3 * n);
			if (tt > upto + 1e-9) break;
			pts.push(merid(at(c, tt)));
		}
		pts.push(merid(at(c, upto)));
		return pts;
	}

	// ── geometry ──
	const sqrtD = $derived(Math.sqrt(D));
	const allPts = $derived(cols.flatMap((c) => [merid(c.x), merid(c.y), merid(c.u)]));
	const aMax = $derived(Math.max(sqrtD, ...allPts.map((p) => Math.abs(p.a))) * 1.12);
	const rMax = $derived(Math.max(sqrtD, ...allPts.map((p) => p.r)) * 1.06);
	const PW = $derived(narrow ? W - 104 : Math.min(420, W * 0.5));
	const unit = $derived(Math.min((PW - 64) / (2 * aMax), (narrow ? 330 : 400) / rMax));
	const ox = $derived((narrow ? 14 : 40) + aMax * unit);
	const oy = $derived(40 + rMax * unit);
	const X = (a) => ox + a * unit;
	const Y = (r) => oy - r * unit;
	const PH = $derived(Math.max(oy + 54, Y(Math.sqrt(D)) - 8 + 120 + 46));

	// inset: depth below the sphere, magnified
	const gapMax = $derived(Math.max(...cols.map((c) => c.gap)));
	const decade = $derived(10 ** Math.floor(Math.log10(gapMax)));
	const insetTop = 18; // px below the inset's √D line... set from layout
	const inH = 120;
	const inX = $derived(narrow ? W - 58 : X(aMax) + 46);
	const inY0 = $derived(Y(sqrtD) - 8);
	const inScale = $derived(inH / (Math.ceil(gapMax / decade) * decade));
	const ticks = $derived(Array.from({ length: Math.ceil(gapMax / decade) + 1 }, (_, k) => k * decade));

	// right: four glyphs
	const STAGES = [
		{ key: 'x', label: 'x', note: 'the column' },
		{ key: 'u', label: 'Πx', note: 'centred: − μ𝟏' },
		{ key: 'xh', label: 'x̂', note: 'scaled: ÷ σ' },
		{ key: 'y', label: 'γ⊙x̂+β', note: 'gain and offset' }
	];
	// one scale for the four stages of the selected column (the glyphs compared)
	const gScale = $derived(maxAbs(sel.x, sel.u, sel.xh, sel.y));
	const GX0 = $derived(narrow ? 0 : PW + 70 + (inX > PW ? 40 : 0));
	const GW = $derived(W - GX0);
	const rG = $derived(narrow ? Math.min(40, (W - 24) / 8.6) : Math.min(58, (GW - 40) / 4.6));
	const gsc = $derived(spokeScale(rG, gScale));
	const gpos = $derived(
		narrow
			? STAGES.map((_, k) => ({ x: (W / 4) * (k + 0.5), y: PH + 58 + rG }))
			: STAGES.map((_, k) => ({ x: GX0 + (GW / 2) * ((k % 2) + 0.5), y: 120 + rG + Math.floor(k / 2) * (2 * rG + 92) }))
	);
	const H = $derived(narrow ? PH + 2 * rG + 120 : Math.max(PH, 120 + 2 * (2 * rG + 92)));
	const stageNow = $derived(Math.min(3, Math.floor(clamp(t, 0, 1) * 3 + 1e-6)));
	const charsY = $derived(narrow ? PH + 16 : 50);

	const PHASES = [
		{ at: 0, label: 'x' },
		{ at: 1 / 3, label: 'centre' },
		{ at: 2 / 3, label: 'scale' },
		{ at: 1, label: 'γ, β' }
	];

	function sci(x) {
		if (x === 0) return '0';
		const e = Math.floor(Math.log10(Math.abs(x)));
		const m = x / 10 ** e;
		return `${fmt(m, 1)}·10${String(e).replace('-', '⁻').replace(/\d/g, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[d])}`;
	}
	const readout = $derived.by(() => {
		const c = sel, ch = showChar(chars[col]);
		return {
			who: `Column ‘${ch}’, position ${col + 1} of “${chars.join('')}”.`,
			centre: `Mean μ = ${fmt(c.mu, 2)}; after centring ‖Πx‖ = ${fmt(norm(c.u), 2)}, so σ = ${fmt(c.sigma, 3)}.`,
			lands: `x̂ lands at ‖x̂‖ = √D − ${sci(c.gap)}, just inside the sphere (ε/σ² = ${sci(eps / c.sigma ** 2)}).`,
			after: `γ⊙x̂+β moves it to ‖·‖ = ${fmt(norm(c.y), 2)}.`
		};
	});
	function key(e) {
		if (e.key === 'ArrowRight') { col = Math.min(T - 1, col + 1); e.preventDefault(); }
		if (e.key === 'ArrowLeft') { col = Math.max(0, col - 1); e.preventDefault(); }
	}
	const arcPath = $derived(`M ${X(-sqrtD)} ${Y(0)} A ${sqrtD * unit} ${sqrtD * unit} 0 0 1 ${X(sqrtD)} ${Y(0)}`);
	const shown = $derived(interactive ? t : 1);
</script>

<div class="fx fx-ln" use:measure={(w) => (width = w)}>
	<!-- svelte-ignore a11y_no_noninteractive_element_interactions, a11y_no_noninteractive_tabindex -->
	<svg
		viewBox="0 0 {W} {H}"
		width={W}
		height={H}
		role="img"
		tabindex={interactive ? 0 : -1}
		onkeydown={key}
		aria-label="LayerNorm of the columns of “{chars.join('')}”: centring moves each column onto the axis orthogonal to 1, scaling moves it to just inside the sphere of radius √{D}. {readout.lands}"
	>
		<!-- ── the meridian plane ── -->
		<g class="fx-ln__plane">
			<line class="axis" x1={X(-aMax)} x2={X(aMax)} y1={Y(0)} y2={Y(0)} />
			<line class="axis" x1={X(0)} x2={X(0)} y1={Y(0)} y2={Y(rMax)} />
			<path class="fx-ln__sphere" d={arcPath} />
			<text class="lbl lbl--micro" x={X(aMax)} y={Y(0) + (narrow ? 30 : 16)} text-anchor="end">along 𝟏</text>
			<text class="lbl lbl--micro" x={X(0) + 6} y={Y(rMax) + 4}>distance from the 𝟏 axis, ‖Πx‖</text>
			<text class="lbl fx-ln__circ" x={X(-sqrtD * Math.SQRT1_2) - 6} y={Y(sqrtD * Math.SQRT1_2) - 4} text-anchor="end">‖x‖ = √{D}</text>
			<text class="lbl lbl--micro" x={X(0)} y={Y(0) + 16} text-anchor="middle">0</text>
			{#each [-2, -1, 1, 2] as k (k)}
				{#if Math.abs(k) < aMax}
					<line class="axis" x1={X(k)} x2={X(k)} y1={Y(0)} y2={Y(0) + 4} />
					<text class="lbl lbl--micro" x={X(k)} y={Y(0) + 16} text-anchor="middle">{fmt(k, 0)}</text>
				{/if}
			{/each}
			{#each [1, 2, 3, 4, 5, 6, 7] as k (k)}
				{#if k < rMax}
					<line class="axis" x1={X(0) - 4} x2={X(0)} y1={Y(k)} y2={Y(k)} />
					<text class="lbl lbl--micro" x={X(0) - 7} y={Y(k) + 3.5} text-anchor="end">{k}</text>
				{/if}
			{/each}

			<!-- trails and dots: every column of the window -->
			<g class="marks">
				{#each cols as c, j (j)}
					{@const p0 = merid(c.x)}
					{@const tr = trail(c, shown)}
					{@const p = tr.at(-1)}
					<polyline class="fx-ln__trail" class:is-on={j === col} points={tr.map((q) => `${X(q.a)},${Y(q.r)}`).join(' ')} />
					<circle class="fx-ln__start" cx={X(p0.a)} cy={Y(p0.r)} r={j === col ? 3.6 : 2.6} />
					<circle class="fx-ln__dot" class:is-on={j === col} cx={X(p.a)} cy={Y(p.r)} r={j === col ? 4.6 : 3} />
				{/each}
			</g>
			{#each cols as c, j (j)}
				{@const p0 = merid(c.x)}
				<text class="tok fx-ln__lab" class:is-on={j === col} x={X(p0.a) + 7} y={Y(p0.r) + 5}>{showChar(chars[j])}</text>
				{#if interactive}
					<circle class="hit" cx={X(p0.a)} cy={Y(p0.r)} r="11" role="button" tabindex="-1" aria-label="column {j + 1}" onclick={() => (col = j)} onkeydown={() => {}} />
				{/if}
			{/each}

			<!-- inset: how far inside the sphere x̂ lands -->
			<g class="fx-ln__inset">
				<line class="leader" x1={X(0) + 5} y1={Y(sqrtD)} x2={inX - 30} y2={inY0} />
				<line class="fx-ln__sphere" x1={inX - 26} x2={inX + 26} y1={inY0} y2={inY0} />
				<text class="lbl lbl--micro" x={inX + 30} y={inY0 + 3.5}>√{D}</text>
				<line class="axis" x1={inX - 26} x2={inX - 26} y1={inY0} y2={inY0 + inH} />
				{#each ticks as v, k (k)}
					{#if k > 0}
						<line class="axis" x1={inX - 30} x2={inX - 26} y1={inY0 + v * inScale} y2={inY0 + v * inScale} />
						<text class="lbl lbl--micro" x={inX - 33} y={inY0 + v * inScale + 3.5} text-anchor="end">{sci(v)}</text>
					{/if}
				{/each}
				<g class="marks">
					{#each cols as c, j (j)}
						{@const landed = shown >= 2 / 3 - 1e-9}
						<line class="fx-ln__depth" class:is-on={j === col} class:is-pending={!landed} x1={inX - 22} x2={inX + 22} y1={inY0 + c.gap * inScale} y2={inY0 + c.gap * inScale} />
					{/each}
				</g>
				<text class="tok fx-ln__lab is-on" x={inX + 26} y={inY0 + sel.gap * inScale + 5}>{showChar(chars[col])}</text>
				<text class="lbl lbl--micro" x={inX - 26} y={inY0 + inH + 18}>√D − ‖x̂‖,</text>
				<text class="lbl lbl--micro" x={inX - 26} y={inY0 + inH + 31}>magnified</text>
			</g>
		</g>

		<!-- ── the window: click a character ── -->
		{#if !narrow}
			<text class="lbl lbl--micro" x={GX0 + 6} y={charsY - 22}>column</text>
		{/if}
		{#each chars as ch, j (j)}
			{@const cx = narrow ? (W / (T + 1)) * (j + 1) : GX0 + 10 + ((GW - 20) / T) * (j + 0.5)}
			<text class="tok fx-ln__char" class:tok--reader={j === col} x={cx} y={charsY} text-anchor="middle">{showChar(ch)}</text>
			{#if interactive}
				<rect class="hit" x={cx - 14} y={charsY - 20} width="28" height="28" role="button" tabindex="-1" aria-label="column {j + 1}" onclick={() => (col = j)} onkeydown={() => {}} />
			{/if}
		{/each}

		<!-- ── the selected column at the four stages ── -->
		{#each STAGES as s, k (s.key)}
			{@const v = sel[s.key]}
			{@const g = gpos[k]}
			{@const rms = norm(v) / Math.sqrt(D)}
			<circle class="fx-ln__rms" cx={g.x} cy={g.y} r={gsc.scale} />
			<SpokeGlyph values={v} x={g.x} y={g.y} rmax={rG} scale={gsc.scale} weight={interactive && k === stageNow ? 1.5 : 1} tip={interactive && k === stageNow ? 2.8 : 2.2} />
			<text class="lbl fx-ln__stage" class:is-on={interactive && k === stageNow} x={g.x} y={g.y + rG + 22} text-anchor="middle">{s.label}</text>
			<text class="lbl lbl--micro" x={g.x} y={g.y + rG + 36} text-anchor="middle">{s.note}</text>
			<text class="lbl lbl--micro" x={g.x} y={g.y + rG + 49} text-anchor="middle">RMS {fmt(rms, 3)}</text>
		{/each}
	</svg>
	{#if interactive}
		<Scrubber bind:t phases={PHASES} label="the stages of LayerNorm" duration={3000} />
		<p class="readout" aria-live="polite">
			<span class="readout__who">{readout.who}</span>
			<span class="readout__part">{readout.centre}</span>
			<span class="readout__part">{readout.lands}</span>
			<span class="readout__part">{readout.after}</span>
			<span class="readout__hint">Run the stages, or drag along the line. Click a point or a character (← → also work) to follow another column.</span>
		</p>
	{/if}
</div>

<style>
	.fx-ln :global(.fx-ln__sphere) { fill: none; stroke: var(--ink); stroke-width: var(--hair); }
	.fx-ln__circ { fill: var(--ink-2); font-size: 11px; }
	.fx-ln__trail { fill: none; stroke: var(--value); stroke-width: var(--hair); opacity: 0.55; }
	.fx-ln__trail.is-on { stroke-width: var(--line); opacity: 1; }
	.fx-ln__start { fill: var(--paper); stroke: var(--value); stroke-width: var(--hair); }
	.fx-ln__dot { fill: var(--value); }
	.fx-ln__lab { font-size: 13px; fill: var(--ink-3); }
	.fx-ln__lab.is-on { fill: var(--ink); font-weight: 600; font-size: 15px; }
	.fx-ln__depth { stroke: var(--value); stroke-width: var(--hair); }
	.fx-ln__depth.is-on { stroke-width: var(--heavy); }
	.fx-ln__depth.is-pending { opacity: 0.35; }
	.fx-ln__rms { fill: none; stroke: var(--ink-3); stroke-width: var(--hair); stroke-dasharray: 1.5 2.5; }
	.fx-ln__char { font-size: 20px; }
	.fx-ln__stage { fill: var(--ink-2); font-family: var(--serif); font-style: italic; font-size: 14px; }
	.fx-ln__stage.is-on { fill: var(--ink); font-weight: 600; }
</style>

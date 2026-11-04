<script module>
	export const meta = { size: 'wide', title: 'The residual stream of one window' };
	export { default as Key } from './fig-residual-stem.key.svelte';
</script>

<script>
	// Figure: the whole model for one window, one stem per token column (forward = down).
	// Each stem: x₀ → attention branch → ⊕ → x_mid → MLP branch → ⊕ → x_out → p.
	// Stem width ∝ ‖x‖/√D, return-leg width ∝ ‖update‖/√D (one scale for every stem and
	// arc, over all ten windows); arcs between the attention branches carry, exactly,
	// what source j hands to reader i: Σ_h W_o,h A_ij v_j — the only marks that cross
	// between columns. The clock t runs the forward pass; at each ⊕ the glyph morphs
	// x → x + s·f(x): an Euler step drawn literally.
	import { getContext } from 'svelte';
	import { measure } from '../primitives/measure.js';
	import ResidualStem, { FORWARD_PHASES } from '../primitives/ResidualStem.svelte';
	import Scrubber from '../primitives/Scrubber.svelte';
	import SpokeGlyph from '../primitives/SpokeGlyph.svelte';
	import { stemWindow, stemGeometry, rms } from '../primitives/stem.js';
	import { fmt, fmtProb, showChar, clamp, maxAbs, spokeScale, layoutWidth } from '../primitives/scales.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const T = $derived(specimen.config.T);
	const D = $derived(specimen.config.D);
	const gt = $derived(specimen.lab.gradTrace(specimen.theta));
	const B = $derived(gt.windows.length);
	const all = $derived(gt.windows.map((_, b) => stemWindow(gt, b, specimen.config)));

	// svelte-ignore state_referenced_locally
	let win = $state(clamp((+attrs.window || 1) - 1, 0, 9));
	// svelte-ignore state_referenced_locally
	let col = $state(clamp((+attrs.column || 8) - 1, 0, 7));
	// svelte-ignore state_referenced_locally
	let t = $state(attrs.t != null ? +attrs.t : 1);
	$effect(() => { if (attrs.t != null) t = +attrs.t; });
	let hover = $state(-1);
	let width = $state(868);
	const interactive = $derived(mode !== 'print');
	const W = $derived(layoutWidth(width, mode));
	const narrow = $derived(W < 600);

	const sw = $derived(all[win]);
	const chars = $derived(Array.from(specimen.corpus.slice(win, win + T)));
	const shown = $derived(hover >= 0 ? hover : col);

	// one scale for every stem, branch and arc in every window
	const maxRms = $derived(Math.max(...all.flatMap((s) => s.cols.flatMap((c) => [rms(c.x0), rms(c.a), rms(c.mid), rms(c.y), rms(c.out)]))));

	const G = $derived(stemGeometry({ W, T, top: 70, narrow }));
	const wMax = $derived(narrow ? G.lane * 0.2 : Math.min(13, G.lane * 0.16));
	const k = $derived(wMax / maxRms);
	// the detail stem's five glyphs are compared with each other: one scale per column
	const sc = $derived.by(() => { const c = sw.cols[shown]; return spokeScale(G.rG || 10, maxAbs(c.x0, c.a, c.mid, c.y, c.out)); });
	const barH = 30;
	const yBase = $derived(G.Y.end + barH + 4);
	const yPred = $derived(yBase + 20);
	const eqTop = $derived(yPred + 34);
	const eqR = 17;
	const H = $derived(narrow ? eqTop + 2 * (2 * eqR + 30) + 6 : yPred + 14);

	// arcs between the attention branches: what j hands to i (j < i)
	const arcs = $derived.by(() => {
		const out = [];
		const y = G.Y.n1 - 2;
		const room = G.Y.n1 - G.Y.x0 - 2;
		for (let i = 1; i < T; i++)
			for (let j = 0; j < i; j++) {
				const v = rms(sw.arcF[i][j]);
				const x1 = G.bxs[j], x2 = G.bxs[i], rx = (x2 - x1) / 2;
				const ry = Math.min(room, 5 + rx * 0.2);
				out.push({ i, j, v, w: v * k, d: `M ${x1} ${y} A ${rx} ${ry} 0 0 1 ${x2} ${y}` });
			}
		return out;
	});
	const arcU = $derived(Math.min(1, Math.max(0, (t - 0.22) / 0.08)));

	function widths(c) {
		return {
			w: { x0: rms(c.x0) * k, mid: rms(c.mid) * k, out: rms(c.out) * k },
			wb: { a: rms(c.a) * k, y: rms(c.y) * k }
		};
	}

	// corpus strip and dragging the window (training windows 1…10)
	const strip = $derived({ y: 14, cs: Math.min(narrow ? 14 : 19, (W - 60) / specimen.corpus.length) });
	const stripX0 = $derived((W - strip.cs * specimen.corpus.length) / 2);
	let dragging = false, grab = 0;
	function svgPoint(svg, e) {
		const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
		return pt.matrixTransform(svg.getScreenCTM().inverse());
	}
	function stripDown(e) {
		const p = svgPoint(e.currentTarget.ownerSVGElement, e);
		const kx = Math.floor((p.x - stripX0) / strip.cs);
		grab = kx >= win && kx < win + T ? kx - win : Math.floor(T / 2);
		dragging = true;
		e.currentTarget.setPointerCapture(e.pointerId);
		win = clamp(Math.round((p.x - stripX0) / strip.cs - grab - 0.5), 0, B - 1);
	}
	function stripMove(e) {
		if (!dragging) return;
		const p = svgPoint(e.currentTarget.ownerSVGElement, e);
		win = clamp(Math.round((p.x - stripX0) / strip.cs - grab - 0.5), 0, B - 1);
	}
	function key(e) {
		if (e.key === 'ArrowLeft') { col = Math.max(0, col - 1); e.preventDefault(); }
		else if (e.key === 'ArrowRight') { col = Math.min(T - 1, col + 1); e.preventDefault(); }
		else if (e.key === 'PageUp' || e.key === ',') { win = Math.max(0, win - 1); e.preventDefault(); }
		else if (e.key === 'PageDown' || e.key === '.') { win = Math.min(B - 1, win + 1); e.preventDefault(); }
	}

	const readout = $derived.by(() => {
		const c = sw.cols[shown];
		const src = sw.arcF[shown].slice(0, shown + 1).map((v, j) => ({ j, v: rms(v) })).sort((a, b) => b.v - a.v)[0];
		return {
			who: `‘${showChar(chars[shown])}’, position ${shown + 1} of “${specimen.corpus.slice(win, win + T)}”.`,
			flow: `The stream’s RMS goes ${fmt(rms(c.x0))} → ${fmt(rms(c.mid))} under an attention update of RMS ${fmt(rms(c.a))}` +
				(shown > 0 ? ` (the largest share, ${fmt(src.v)}, from ‘${showChar(chars[src.j])}’ at ${src.j + 1}${src.j === shown ? ', itself' : ''})` : ', all of it read from itself') +
				`; under an MLP update of RMS ${fmt(rms(c.y))}, it goes ${fmt(rms(c.mid))} → ${fmt(rms(c.out))}.`,
			pred: `It predicts ‘${showChar(specimen.vocab[c.pred])}’ with p = ${fmtProb(c.probs[c.pred], 2, true)}; the next character is ‘${showChar(specimen.vocab[c.target])}’.`
		};
	});
	const sub = (r) => String(r).split('').map((ch) => '₀₁₂₃₄₅₆₇₈₉'[+ch]).join('');
</script>

{#snippet labels()}
	{#if !narrow}
		<g class="fx-stem__rows">
			{#each [[G.Y.x0, 'x', '0', 'embedding'], [G.Y.n1, 'a', '', 'attention'], [G.Y.mid, 'x', 'mid', ''], [G.Y.n2, 'y', '', 'MLP'], [G.Y.out, 'x', 'out', ''], [yBase - barH / 2, 'p', '', 'next character']] as [y, s, ss, word] (word + s + ss)}
				{#if word}<text class="lbl" x={G.padL - 14} y={y - 3} text-anchor="end">{word}</text>{/if}
				<text class="lbl" x={G.padL - 14} y={word ? y + 13 : y + 4} text-anchor="end"><tspan class="m">{s}</tspan>{#if ss}<tspan class="m sub" dy="3">{ss}</tspan>{/if}</text>
			{/each}
		</g>
	{/if}
{/snippet}

{#snippet strand(c, t, x, bx, detail)}
	{@const wd = widths(c)}
	<ResidualStem {x} {bx} Y={G.Y} w={wd.w} wb={wd.wb} g={detail ? { x0: c.x0, a: c.a, mid: c.mid, y: c.y, out: c.out } : null} rG={detail ? G.rG : 0} scale={sc.scale} glyphs={detail} {t}
		bars={{ values: c.probs, scale: barH, offset: barH + 4, mark: c.pred, width: detail ? 6 : narrow ? 2.4 : 3.2, gap: detail ? 2.2 : narrow ? 0.9 : 1.3 }} />
{/snippet}

<div class="fx fx-stem" use:measure={(w) => (width = w)}>
	{#snippet body()}
		<!-- the corpus, with the window -->
		<g class="fx-stem__strip">
			{#each Array.from(specimen.corpus) as ch, kx (kx)}
				<text class="tok tok--small" class:tok--out={kx < win || kx >= win + T} x={stripX0 + strip.cs * (kx + 0.5)} y={strip.y + 10} text-anchor="middle">{showChar(ch)}</text>
			{/each}
			<path class="fx-reads__bracket" d="M {stripX0 + strip.cs * win + 2} {strip.y + 16} v 4 H {stripX0 + strip.cs * (win + T) - 2} v -4" />
			<text class="lbl lbl--micro" x={stripX0 + strip.cs * (win + T) + 6} y={strip.y + 21}>window {win + 1}</text>
			{#if interactive}
				<rect class="hit hit--drag" x={stripX0 - 6} y={strip.y - 6} width={strip.cs * specimen.corpus.length + 12} height="32" onpointerdown={stripDown} onpointermove={stripMove} onpointerup={() => (dragging = false)} role="presentation" />
			{/if}
		</g>

		{@render labels()}

		<!-- what crosses between columns: only attention -->
		<g class="fx-stem__arcs">
			{#each arcs as a (a.i * 8 + a.j)}
				{#if a.w > 0.15 && arcU > 0}
					<path class="fx-stem__arc" class:is-on={a.i === shown} d={a.d} stroke-width={a.w} pathLength="1" stroke-dasharray="{arcU} 1" />
				{/if}
			{/each}
		</g>

		<!-- the stems: all columns, and the selected one in detail -->
		{#each sw.cols as c, i (i)}
			{@render strand(c, t, G.xs[i], G.bxs[i], false)}
		{/each}
		{#if !narrow}
			{@const c = sw.cols[shown]}
			{@render strand(c, t, G.detail.x, G.detail.bx, true)}
			<text class="tok fx-stem__dchar" x={G.detail.x} y={G.Y.char} text-anchor="middle">{showChar(chars[shown])}</text>
			<text class="lbl lbl--micro" x={G.detail.x + 12} y={G.Y.char + 2}>{shown + 1}</text>
			{#if t >= 0.95}
				{@const px = G.detail.x - (7 * 6 + 6 * 2.2) / 2 + c.pred * 8.2 + 3}
				<text class="tok fx-stem__dpred" x={px} y={yPred} text-anchor="middle">{showChar(specimen.vocab[c.pred])}</text>
				<text class="lbl lbl--micro" x={px + 9} y={yPred}>{fmtProb(c.probs[c.pred])}</text>
			{/if}
		{/if}

		<!-- input characters on top, predictions at the foot -->
		{#each chars as ch, i (i)}
			<text class="tok fx-stem__char" class:tok--reader={i === shown} x={G.xs[i]} y={G.Y.char} text-anchor="middle">{showChar(ch)}</text>
			<text class="lbl lbl--micro" x={G.xs[i] + 7} y={G.Y.char + 2}>{i + 1}</text>
			{@const c = sw.cols[i]}
			{#if t >= 0.95}
				<text class="tok fx-stem__pred" class:is-wrong={c.pred !== c.target} x={G.xs[i]} y={yPred} text-anchor="middle">{showChar(specimen.vocab[c.pred])}</text>
			{/if}
		{/each}
		{#if !narrow}
			<text class="lbl" x={G.padL - 14} y={yPred} text-anchor="end">predicted</text>
		{/if}
		<line class="fx-stem__sel" x1={G.xs[shown] - 7} x2={G.xs[shown] + 7} y1={G.Y.char + 6} y2={G.Y.char + 6} />
		{#if interactive}
			{#each chars as _, i (i)}
				<rect class="hit" x={G.ox + G.lane * i} y={G.Y.char - 22} width={G.lane} height={yPred - G.Y.char + 30} role="button" aria-label="read this entry; details in the live readout" tabindex="0"
					onpointerenter={() => (hover = i)} onfocus={() => (hover = i)} onpointerleave={() => (hover = -1)} onblur={() => (hover = -1)} onclick={() => { col = i; hover = -1; }} onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); { col = i; hover = -1; }; } }} />
			{/each}
		{/if}

		<!-- narrow screens: the selected column's two Euler steps, as glyph arithmetic -->
		{#if narrow}
			{@const c = sw.cols[shown]}
			{@const e = spokeScale(eqR, maxAbs(c.x0, c.a, c.mid, c.y, c.out))}
			{@const cx = (n) => W / 2 + (n - 2) * (W / 5.2)}
			{#each [[c.x0, c.a, c.mid, ['x', '0'], ['a', ''], ['x', 'mid']], [c.mid, c.y, c.out, ['x', 'mid'], ['y', ''], ['x', 'out']]] as row, r (r)}
				{@const yy = eqTop + eqR + r * (2 * eqR + 30)}
				<SpokeGlyph values={row[0]} x={cx(0)} y={yy} rmax={eqR} scale={e.scale} />
				<text class="lbl fx-stem__op" x={(cx(0) + cx(1)) / 2} y={yy + 4} text-anchor="middle">+</text>
				<SpokeGlyph values={row[1]} x={cx(1)} y={yy} rmax={eqR} scale={e.scale} />
				<text class="lbl fx-stem__op" x={(cx(1) + cx(2)) / 2} y={yy + 4} text-anchor="middle">=</text>
				<SpokeGlyph values={row[2]} x={cx(2)} y={yy} rmax={eqR} scale={e.scale} />
				{#each [3, 4, 5] as n, q (q)}
					<text class="lbl" x={cx(q)} y={yy + eqR + 15} text-anchor="middle"><tspan class="m">{row[n][0]}</tspan>{#if row[n][1]}<tspan class="m sub" dy="3">{row[n][1]}</tspan>{/if}</text>
				{/each}
			{/each}
		{/if}
	{/snippet}
	{#if interactive}
		<svg viewBox="0 0 {W} {H}" width={W} height={H} role="slider" tabindex="0"
			aria-label="The forward pass of the window “{specimen.corpus.slice(win, win + T)}”, one stem per token. {readout.who} {readout.flow} {readout.pred}"
			aria-valuemin="1" aria-valuemax={T} aria-valuenow={col + 1} onkeydown={key}>{@render body()}</svg>
	{:else}
		<svg viewBox="0 0 {W} {H}" width={W} height={H} role={interactive ? 'group' : 'img'}
			aria-label="The forward pass of the window “{specimen.corpus.slice(win, win + T)}”, one stem per token. {readout.who} {readout.flow} {readout.pred}">{@render body()}</svg>
	{/if}
	{#if interactive}
		<Scrubber bind:t phases={FORWARD_PHASES} label="the forward pass" duration={3600} />
		<p class="readout" aria-live="polite">
			<span class="readout__who">{readout.who}</span>
			<span class="readout__part">{readout.flow}</span>
			<span class="readout__part">{t >= 0.95 ? readout.pred : ''}</span>
			<span class="readout__hint">Widths are sizes ‖x‖/√D. Click a stem (←/→); drag the window along the line; play the forward pass.</span>
		</p>
	{/if}
</div>

<style>
	.fx-stem :global(.m) { font-family: var(--serif); font-style: italic; font-size: 14px; fill: var(--ink); }
	.fx-stem :global(.sub) { font-size: 10px; }
	.fx-stem__char { font-size: 17px; }
	.fx-stem__dchar { font-size: 26px; font-weight: 560; }
	.fx-stem__dpred { font-size: 19px; fill: var(--ink); }
	.fx-stem__pred { font-size: 16px; fill: var(--ink); }
	.fx-stem__pred.is-wrong { fill: var(--ink-3); }
	.fx-stem__arc { fill: none; stroke: var(--value); opacity: 0.2; mix-blend-mode: var(--overprint); stroke-linecap: butt; }
	.fx-stem__arc.is-on { opacity: 0.9; }
	.fx-stem__sel { stroke: var(--ink); stroke-width: 1.2px; }
	.fx-stem__op { font-size: 15px; fill: var(--ink-2); }
	:global([data-mode='print']) .fx-stem__arc { opacity: 0.55; }
	@media (max-width: 37.5rem) { .fx-stem__char { font-size: 17px; } .fx-stem__pred { font-size: 15px; } }
</style>

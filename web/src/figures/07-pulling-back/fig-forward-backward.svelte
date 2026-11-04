<script module>
	export const meta = { size: 'wide', title: 'Forward, then back' };
	export { default as Key } from './fig-forward-backward.key.svelte';
</script>

<script>
	// Figure: the book's organising idea (BRIEF §1). One clock t ∈ [0, 1]:
	//   t ∈ [0, ½]  the forward pass runs DOWN the stems in value ink;
	//   t ∈ [½, 1]  the loss sends z̄ = (p − e_y)/N into every column and the adjoints run
	//               UP the same stems in covector ink (hatched, bar tips).
	// Widths: ‖x‖/√D (value) and ‖x̄‖/√D (covector), each with one scale over all ten
	// windows at the chosen θ. At a forward ⊕ the adjoint is copied both ways; at a forward
	// fork the copies' adjoints ADD (the covector ⊕). The arcs of the backward pass run
	// from each reader back to its sources: exactly what reader i hands back to source j.
	// θ is θ₀ (before training) by default; "after training" shows that only the
	// ambiguous predictions still send anything back.
	import { getContext } from 'svelte';
	import { measure } from '../primitives/measure.js';
	import ResidualStem, { FORWARD_PHASES, BACKWARD_PHASES } from '../primitives/ResidualStem.svelte';
	import Scrubber from '../primitives/Scrubber.svelte';
	import { stemWindow, stemGeometry, rms } from '../primitives/stem.js';
	import { fmt, fmtProb, showChar, clamp, maxAbs, spokeScale, layoutWidth } from '../primitives/scales.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const T = $derived(specimen.config.T);

	// svelte-ignore state_referenced_locally
	let trained = $state(attrs.theta === 'trained');
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

	const theta = $derived(trained ? specimen.theta : specimen.theta0);
	const gt = $derived(specimen.lab.gradTrace(theta));
	const B = $derived(gt.windows.length);
	const all = $derived(gt.windows.map((_, b) => stemWindow(gt, b, specimen.config)));
	const sw = $derived(all[win]);
	const chars = $derived(Array.from(specimen.corpus.slice(win, win + T)));
	const shown = $derived(hover >= 0 ? hover : col);

	// scales: value and covector, each over all ten windows at this θ
	const maxV = $derived(Math.max(...all.flatMap((s) => s.cols.flatMap((c) => [rms(c.x0), rms(c.a), rms(c.mid), rms(c.y), rms(c.out)]))));
	const maxC = $derived(Math.max(...all.flatMap((s) => s.cols.flatMap((c) => [rms(c.bx0), rms(c.batt), rms(c.bmid), rms(c.bmlp), rms(c.bout)]))));
	const maxZ = $derived(maxAbs(all.map((s) => s.cols.map((c) => c.zbar))));

	const G = $derived(stemGeometry({ W, T, top: 70, narrow }));
	const barH = 30;

	const tf = $derived(Math.min(1, 2 * t));
	const tb = $derived(Math.max(0, 2 * t - 1));
	const back = $derived(t > 0.5);

	// (label stops: the middle of each branch; the forward pass takes the first half)
	const phases = [
		{ at: 0, label: 'forward' },
		{ at: FORWARD_PHASES[1].at / 2, label: '' },
		{ at: FORWARD_PHASES[3].at / 2, label: 'MLP' },
		{ at: 0.5, label: 'backward' },
		{ at: 0.5 + BACKWARD_PHASES[1].at / 2, label: 'MLP' },
		{ at: 0.5 + BACKWARD_PHASES[3].at / 2, label: 'attention' }
	];

	/** Everything that depends on a geometry: width scales, rows, arcs. */
	function geo(Gp, nar) {
		const wMax = nar ? Gp.lane * 0.2 : Math.min(13, Gp.lane * 0.16);
		const kV = wMax / maxV, kC = wMax / maxC;
		const yBase = Gp.Y.end + barH + 4, yPred = yBase + 20;
		const arcs = [];
		for (let i = 1; i < T; i++)
			for (let j = 0; j < i; j++) {
				const y = Gp.Y.n1 - 2, x1 = Gp.bxs[j], x2 = Gp.bxs[i], rx = (x2 - x1) / 2;
				const ry = Math.min(Gp.Y.n1 - Gp.Y.x0 - 2, 5 + rx * 0.2);
				arcs.push({
					i, j,
					wf: rms(sw.arcF[i][j]) * kV,
					wb: rms(sw.arcB[i][j]) * kC,
					df: `M ${x1} ${y} A ${rx} ${ry} 0 0 1 ${x2} ${y}`,
					db: `M ${x2} ${y} A ${rx} ${ry} 0 0 0 ${x1} ${y}`
				});
			}
		return { kV, kC, yBase, yPred, H: yPred + 14, arcs, nar };
	}
	const S = $derived(geo(G, narrow));
	const H = $derived(S.H);
	const arcU = (tt, a) => Math.min(1, Math.max(0, (tt - a) / 0.08));

	// print: forward and backward side by side, each the overview of the eight stems
	const pGap = 26;
	const PG = $derived(stemGeometry({ W: (W - pGap) / 2, T, top: 58, narrow: true }));
	const PS = $derived(geo(PG, true));

	const valueOf = (c, kV) => ({ w: { x0: rms(c.x0) * kV, mid: rms(c.mid) * kV, out: rms(c.out) * kV }, wb: { a: rms(c.a) * kV, y: rms(c.y) * kV } });
	const covOf = (c, kC) => ({ w: { x0: rms(c.bx0) * kC, mid: rms(c.bmid) * kC, out: rms(c.bout) * kC }, wb: { a: rms(c.batt) * kC, y: rms(c.bmlp) * kC } });

	// detail stem, value glyphs: one scale over the selected column's five vectors
	const dc = $derived(sw.cols[shown]);
	const scV = $derived(spokeScale(G.rG || 10, maxAbs(dc.x0, dc.a, dc.mid, dc.y, dc.out)));
	// covector glyphs: one scale over every column of every window at this θ, so that a
	// column whose adjoints are a thousand times smaller looks it
	const maxCs = $derived(maxAbs(all.map((s) => s.cols.map((c) => [c.bx0, c.batt, c.bmid, c.bmlp, c.bout]))));
	const scC = $derived(spokeScale(G.rG || 10, maxCs));

	// which positions of this window are ambiguous (their context has several continuations)
	const ambiguousHere = $derived.by(() => {
		const out = [];
		for (let i = 0; i < T; i++) {
			const ctx = specimen.corpus.slice(win, win + i + 1);
			const a = specimen.lab.ambiguous.find((q) => q.context === ctx);
			if (a) out.push({ i, a });
		}
		return out;
	});

	// corpus strip and window drag
	const stripY = 14;
	const cs = $derived(Math.min(narrow || !interactive ? 14 : 19, (W - (interactive ? 60 : 340)) / specimen.corpus.length));
	const stripX0 = $derived((W - cs * specimen.corpus.length) / 2);
	let dragging = false, grab = 0;
	function svgPoint(svg, e) {
		const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
		return pt.matrixTransform(svg.getScreenCTM().inverse());
	}
	function stripAt(e) { return svgPoint(e.currentTarget.ownerSVGElement, e).x; }
	function stripDown(e) {
		const x = stripAt(e), kx = Math.floor((x - stripX0) / cs);
		grab = kx >= win && kx < win + T ? kx - win : Math.floor(T / 2);
		dragging = true;
		e.currentTarget.setPointerCapture(e.pointerId);
		win = clamp(Math.round((x - stripX0) / cs - grab - 0.5), 0, B - 1);
	}
	function stripMove(e) { if (dragging) win = clamp(Math.round((stripAt(e) - stripX0) / cs - grab - 0.5), 0, B - 1); }
	function key(e) {
		if (e.key === 'ArrowLeft') { col = Math.max(0, col - 1); e.preventDefault(); }
		else if (e.key === 'ArrowRight') { col = Math.min(T - 1, col + 1); e.preventDefault(); }
		else if (e.key === 'PageUp' || e.key === ',') { win = Math.max(0, win - 1); e.preventDefault(); }
		else if (e.key === 'PageDown' || e.key === '.') { win = Math.min(B - 1, win + 1); e.preventDefault(); }
	}

	/** 3.2 × 10⁻³ */
	function sci(x) {
		if (!(x > 0)) return '0';
		const e = Math.floor(Math.log10(x)), m = x / 10 ** e;
		const sup = String(e).replace('-', '⁻').replace(/\d/g, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+d]);
		return e >= -1 && e <= 1 ? fmt(x, 2) : `${fmt(m, 1)}×10${sup}`;
	}

	const readout = $derived.by(() => {
		const c = sw.cols[shown];
		const who = `‘${showChar(chars[shown])}’, position ${shown + 1} of “${specimen.corpus.slice(win, win + T)}”, ${trained ? 'after training' : 'at θ₀, before training'}.`;
		if (!back) return { who, part: `Forward: the stream enters at ${fmt(rms(c.x0))}, attention adds ${fmt(rms(c.a))}, the MLP adds ${fmt(rms(c.y))}; p(next = ‘${showChar(specimen.vocab[c.target])}’) = ${fmtProb(c.probs[c.target], 2, true)}.` };
		const z = rms(c.zbar), x0 = rms(c.bx0);
		const sent = sw.arcB[shown].slice(0, shown).map((v, j) => ({ j, v: rms(v) })).sort((a, b) => b.v - a.v)[0];
		const got = sw.arcB.slice(shown + 1).map((r, q) => ({ i: shown + 1 + q, v: rms(r[shown]) })).sort((a, b) => b.v - a.v)[0];
		let part = `Back: the loss sends (p − e_y)/N, of size ${sci(z)}, into this column; the adjoint that reaches the embedding has size ${sci(x0)}.`;
		if (sent) part += ` Its attention hands most back to ‘${showChar(chars[sent.j])}’ at ${sent.j + 1} (${sci(sent.v)})`;
		if (got) part += `${sent ? '; it' : ' It'} receives most from the reader ‘${showChar(chars[got.i])}’ at ${got.i + 1} (${sci(got.v)})`;
		if (sent || got) part += '.';
		if (trained) {
			const amb = ambiguousHere.map((q) => `position ${q.i + 1} (‘${showChar(q.a.context)}’ is followed by ${Object.keys(q.a.followers).map((f) => `‘${showChar(f)}’`).join(' or ')})`);
			part += amb.length ? ` After training, only ambiguous predictions still send back anything visible: here ${amb.join(', ')}.` : ' After training no prediction in this window is ambiguous, and nothing visible comes back: drag the window to one that starts with ‘o’ or ‘␣’.';
		}
		return { who, part };
	});
</script>

{#snippet stemOf(c, x, bx, detail, Gp, Sp, tF, tB, isBack)}
	{@const v = valueOf(c, Sp.kV)}
	{@const cv = covOf(c, Sp.kC)}
	{@const bw = detail ? 6 : Sp.nar ? 2.4 : 3.2}
	{@const bg = detail ? 2.2 : Sp.nar ? 0.9 : 1.3}
	<ResidualStem {x} {bx} Y={Gp.Y} w={v.w} wb={v.wb} g={detail ? { x0: c.x0, a: c.a, mid: c.mid, y: c.y, out: c.out } : null} rG={detail ? Gp.rG : 0} scale={scV.scale} glyphs={detail} t={tF} muted={isBack}
		bars={isBack && !detail ? null : { values: c.probs, scale: barH, offset: barH + 4, mark: c.target, width: bw, gap: bg }} />
	{#if isBack}
		<ResidualStem {x} {bx} Y={Gp.Y} w={cv.w} wb={cv.wb} g={detail ? { x0: c.bx0, a: c.batt, mid: c.bmid, y: c.bmlp, out: c.bout } : null} rG={detail ? Gp.rG : 0} scale={scC.scale} glyphs={detail} t={tB} ink="covector" track={false} tipScale={0.8}
			bars={{ values: c.zbar, scale: barH / 2 / maxZ, offset: barH / 2 + 4, signed: true, mark: c.target, width: bw, gap: bg }} />
	{/if}
{/snippet}

{#snippet overview(Gp, Sp, tF, tB, isBack)}
	<!-- arcs: forward (value), then back (covector) -->
	<g>
		{#each Sp.arcs as a (a.i * 8 + a.j)}
			{#if a.wf > 0.15 && arcU(tF, 0.22) > 0}
				<path class="fx-fb__arc" class:is-on={a.i === shown} class:is-muted={isBack} d={a.df} stroke-width={a.wf} pathLength="1" stroke-dasharray="{arcU(tF, 0.22)} 1" />
			{/if}
		{/each}
		{#each Sp.arcs as a (a.i * 8 + a.j)}
			{#if isBack && a.wb > 0.15 && arcU(tB, 0.66) > 0}
				<path class="fx-fb__arc fx-fb__arc--cov" class:is-on={a.i === shown} d={a.db} stroke-width={a.wb} pathLength="1" stroke-dasharray="{arcU(tB, 0.66)} 1" />
			{/if}
		{/each}
	</g>
	{#each sw.cols as c, i (i)}
		{@render stemOf(c, Gp.xs[i], Gp.bxs[i], false, Gp, Sp, tF, tB, isBack)}
	{/each}
	<!-- input characters on top; the targets at the foot -->
	{#each chars as ch, i (i)}
		<text class="tok fx-fb__char" class:tok--reader={i === shown} x={Gp.xs[i]} y={Gp.Y.char} text-anchor="middle">{showChar(ch)}</text>
		<text class="lbl lbl--micro" x={Gp.xs[i] + 7} y={Gp.Y.char + 2}>{i + 1}</text>
		<text class="tok fx-fb__next" x={Gp.xs[i]} y={Sp.yPred} text-anchor="middle">{showChar(specimen.vocab[sw.cols[i].target])}</text>
	{/each}
	<line class="fx-fb__sel" x1={Gp.xs[shown] - 7} x2={Gp.xs[shown] + 7} y1={Gp.Y.char + 6} y2={Gp.Y.char + 6} />
	{#if trained && isBack}
		{#each ambiguousHere as q (q.i)}
			<text class="lbl lbl--micro lbl--covector" x={Gp.xs[q.i]} y={Sp.yPred + 13} text-anchor="middle">ambiguous</text>
		{/each}
	{/if}
{/snippet}

{#snippet strip()}
	<g>
		{#each Array.from(specimen.corpus) as ch, kx (kx)}
			<text class="tok tok--small" class:tok--out={kx < win || kx >= win + T} x={stripX0 + cs * (kx + 0.5)} y={stripY + 10} text-anchor="middle">{showChar(ch)}</text>
		{/each}
		<path class="fx-reads__bracket" d="M {stripX0 + cs * win + 2} {stripY + 16} v 4 H {stripX0 + cs * (win + T) - 2} v -4" />
		<text class="lbl lbl--micro" x={stripX0 + cs * (win + T) + 6} y={stripY + 21}>window {win + 1}</text>
		{#if interactive}
			<rect class="hit hit--drag" x={stripX0 - 6} y={stripY - 6} width={cs * specimen.corpus.length + 12} height="32" onpointerdown={stripDown} onpointermove={stripMove} onpointerup={() => (dragging = false)} role="presentation" />
		{/if}
	</g>
{/snippet}

<div class="fx fx-fb" use:measure={(w) => (width = w)}>
	{#if interactive}
		<svg viewBox="0 0 {W} {H}" width={W} height={H} role="slider" tabindex="0"
			aria-label="Forward and backward pass of the window “{specimen.corpus.slice(win, win + T)}”. {readout.who} {readout.part}"
			aria-valuemin="1" aria-valuemax={T} aria-valuenow={col + 1} onkeydown={key}>
			{@render strip()}
			<!-- θ: before or after training -->
			{#each [[false, narrow ? 'θ₀' : 'θ₀, before training'], [true, narrow ? 'trained' : 'after training']] as [v, word], q (q)}
				<text class="lbl fx-fb__theta" class:is-on={trained === v} x={narrow ? 6 + q * 30 : W - 6 - (q ? 0 : 104)} y={narrow ? 52 : stripY + 10} text-anchor={narrow ? 'start' : 'end'} role="button" tabindex="0" aria-pressed={trained === v}
					onclick={() => (trained = v)} onkeydown={(e) => (e.key === 'Enter' || e.key === ' ') && (trained = v)}>{word}</text>
			{/each}

			<!-- row labels -->
			{#if !narrow}
				{#each [[G.Y.x0, 'x', '0', 'embedding'], [G.Y.n1, 'a', '', 'attention'], [G.Y.mid, 'x', 'mid', ''], [G.Y.n2, 'y', '', 'MLP'], [G.Y.out, 'x', 'out', ''], [S.yBase - barH / 2, back ? 'z' : 'p', '', back ? 'from the loss' : 'next character']] as [y, sy, ss, word] (word + y + ss)}
					{#if word}<text class="lbl" x={G.padL - 14} y={y - 3} text-anchor="end">{word}</text>{/if}
					{#if back && (sy === 'a' || sy === 'y')}
						<text class="lbl lbl--covector" x={G.padL - 14} y={y + 13} text-anchor="end">hands back</text>
					{:else}
						<text class="lbl fx-fb__sym" class:is-cov={back} x={G.padL - 14} y={word ? y + 13 : y + 4} text-anchor="end"><tspan class:fx-fb__bar={back}>{sy}</tspan>{#if ss}<tspan class="fx-fb__sub" dy="3">{ss}</tspan>{/if}</text>
					{/if}
				{/each}
				<text class="lbl" x={G.padL - 14} y={S.yPred} text-anchor="end">target</text>
			{/if}

			{@render overview(G, S, tf, tb, back)}

			{#if !narrow}
				{@render stemOf(dc, G.detail.x, G.detail.bx, true, G, S, tf, tb, back)}
				<text class="tok fx-fb__dchar" x={G.detail.x} y={G.Y.char} text-anchor="middle">{showChar(chars[shown])}</text>
				<text class="lbl lbl--micro" x={G.detail.x + 12} y={G.Y.char + 2}>{shown + 1}</text>
				<!-- the size of what lands at each station -->
				{#each [[G.Y.x0, dc.x0, dc.bx0, 0.06, 0.92], [G.Y.mid, dc.mid, dc.bmid, 0.46, 0.54], [G.Y.out, dc.out, dc.bout, 0.84, 0.2]] as [y, v, bv, ta, tbb], q (q)}
					{#if back ? tb >= tbb : tf >= ta}
						<text class="lbl lbl--micro fx-fb__norm" class:is-cov={back} x={G.detail.x + G.rG + 5} y={y + 4}>{back ? sci(rms(bv)) : fmt(rms(v))}</text>
					{/if}
				{/each}
				{@const px = G.detail.x - (7 * 6 + 6 * 2.2) / 2 + dc.target * 8.2 + 3}
				<text class="tok fx-fb__target" x={px} y={S.yPred} text-anchor="middle">{showChar(specimen.vocab[dc.target])}</text>
			{/if}

			{#each chars as _, i (i)}
				<rect class="hit" x={G.ox + G.lane * i} y={G.Y.char - 22} width={G.lane} height={S.yPred - G.Y.char + 30} role="button" aria-label="read this entry; details in the live readout" tabindex="0"
					onpointerenter={() => (hover = i)} onfocus={() => (hover = i)} onpointerleave={() => (hover = -1)} onblur={() => (hover = -1)} onclick={() => { col = i; hover = -1; }} onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); { col = i; hover = -1; }; } }} />
			{/each}
		</svg>
	{:else}
		{@const PH = PS.H + 6}
		<svg viewBox="0 0 {W} {PH}" width={W} height={PH} role={interactive ? 'group' : 'img'}
			aria-label="Forward pass (left, values flowing down) and backward pass (right, adjoints flowing up) of the window “{specimen.corpus.slice(win, win + T)}”, {trained ? 'after training' : 'at θ₀, before training'}.">
			{@render strip()}
			<text class="lbl lbl--head" x="4" y={stripY + 10}>forward: values, down</text>
			<text class="lbl lbl--head lbl--covector" x={W - 4} y={stripY + 10} text-anchor="end">back: adjoints, up</text>
			<g>{@render overview(PG, PS, 1, 0, false)}</g>
			<g transform="translate({(W + pGap) / 2} 0)">{@render overview(PG, PS, 1, 1, true)}</g>
		</svg>
	{/if}
	{#if interactive}
		<Scrubber bind:t {phases} label="the forward pass, then the backward pass" duration={7200} ink={back ? 'covector' : 'value'} />
		<p class="readout" aria-live="polite">
			<span class="readout__who">{readout.who}</span>
			<span class="readout__part">{readout.part}</span>
			<span class="readout__hint">Widths are sizes, ‖·‖/√D: of values in value ink, of adjoints in covector ink. Click a stem (←/→), drag the window, switch θ, scrub the clock.</span>
		</p>
	{/if}
</div>

<style>
	.fx-fb__dchar { font-size: 26px; font-weight: 560; }
	.fx-fb__char { font-size: 17px; }
	.fx-fb__next { font-size: 16px; fill: var(--ink-2); }
	.fx-fb__target { font-size: 19px; fill: var(--ink); }
	.fx-fb__sym { font-family: var(--serif); font-style: italic; font-size: 14px; fill: var(--ink); }
	.fx-fb__sym.is-cov { fill: var(--covector-deep); }
	.fx-fb__bar { text-decoration: overline; }
	.fx-fb__sub { font-size: 10px; }
	.fx-fb__norm { fill: var(--value-deep); }
	.fx-fb__norm.is-cov { fill: var(--covector-deep); }
	.fx-fb__theta { cursor: pointer; fill: var(--ink-3); }
	.fx-fb__theta.is-on { fill: var(--ink); font-weight: 600; text-decoration: underline; text-underline-offset: 3px; }
	.fx-fb__arc { fill: none; stroke: var(--value); opacity: 0.2; mix-blend-mode: var(--overprint); stroke-linecap: butt; }
	.fx-fb__arc.is-on { opacity: 0.9; }
	.fx-fb__arc.is-muted { stroke: var(--ink-3); opacity: 0.12; }
	.fx-fb__arc--cov { stroke: var(--covector); opacity: 0.25; }
	.fx-fb__arc--cov.is-on { opacity: 0.95; }
	.fx-fb__sel { stroke: var(--ink); stroke-width: 1.2px; }
	:global([data-mode='print']) .fx-fb__arc { mix-blend-mode: normal; }
	:global([data-mode='print']) .fx-fb__theta { text-decoration: none; }
</style>

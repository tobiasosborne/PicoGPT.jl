<script module>
	export const meta = { size: 'wide', title: 'The gradient check' };
</script>

<script>
	// Figure: the pulled-back gradient θ̄ at θ₀ against central differences
	//   ĝᵢ(h) = (ℓ(θ + h eᵢ) − ℓ(θ − h eᵢ)) / 2h.
	// Left: |ĝᵢ − θ̄ᵢ| against h (decades, h shrinking to the right) for the largest entry of
	// θ̄ in four arrays: it falls like h² (Taylor) until rounding in the two losses takes
	// over and it rises like u ℓ / h. Right: the same error for every parameter of the
	// model at one h, grouped by array, on the same vertical scale; a covector bar above
	// each group marks its largest |θ̄ᵢ|. Both share one log axis.
	// Cost: 2p forward passes per h, so the numbers are computed once (computed.svelte.js).
	import { getContext, onMount } from 'svelte';
	import { measure } from '../primitives/measure.js';
	import { computed } from '../primitives/computed.svelte.js';
	import { fmt, clamp, layoutWidth } from '../primitives/scales.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = getContext('specimen')();
	const interactive = $derived(mode !== 'print');
	let width = $state(868);

	const HS = Array.from({ length: 13 }, (_, k) => -(k + 1)); // log10 h: −1 … −13
	const STRIPS = [-2, -4, -6, -8, -10]; // log10 h with a full check of every parameter
	const CURVES = ['E', 'blocks.0.Wq', 'blocks.0.W1', 'Wout'];

	const job = computed('fig-gradcheck:v1', function* () {
		const ev = specimen.lab.evaluator();
		const theta = Float64Array.from(specimen.theta0);
		const g = ev.grad(theta);
		const loss = ev.loss(theta);
		const fd = (i, h) => {
			const v = theta[i];
			theta[i] = v + h; const a = ev.loss(theta);
			theta[i] = v - h; const b = ev.loss(theta);
			theta[i] = v;
			return (a - b) / (2 * h);
		};
		const lg = (x) => (x > 0 ? Math.log10(x) : null);
		const curves = CURVES.map((name) => {
			const l = specimen.layout.find((q) => q.name === name);
			let best = l.offset;
			for (let i = l.offset; i < l.offset + l.size; i++) if (Math.abs(g[i]) > Math.abs(g[best])) best = i;
			return { name, i: best, g: g[best], err: HS.map((e) => lg(Math.abs(fd(best, 10 ** e) - g[best]))) };
		});
		yield 0.02;
		const strips = {}, rel = {};
		let done = 0;
		for (const e of STRIPS) {
			const h = 10 ** e, err = new Array(g.length);
			let num = 0, n1 = 0, n2 = 0;
			for (let i = 0; i < g.length; i++) {
				const est = fd(i, h);
				err[i] = lg(Math.abs(est - g[i]));
				num += (est - g[i]) ** 2; n1 += est * est; n2 += g[i] * g[i];
				if (i % 64 === 63) yield 0.02 + (0.98 * (done * g.length + i)) / (STRIPS.length * g.length);
			}
			strips[e] = err;
			rel[e] = Math.sqrt(num) / Math.max(Math.sqrt(n1), Math.sqrt(n2));
			done++;
		}
		const groups = specimen.layout.map((l) => {
			let m = 0;
			for (let i = l.offset; i < l.offset + l.size; i++) m = Math.max(m, Math.abs(g[i]));
			return { name: l.name, offset: l.offset, size: l.size, gmax: lg(m) };
		});
		return { loss, curves, strips, rel, groups, p: g.length };
	}, { digits: 5 });

	// svelte-ignore state_referenced_locally
	let sel = $state(STRIPS.includes(+attrs.h) ? +attrs.h : -6);

	// live timing of one forward and one forward + backward pass (interactive only)
	let timing = $state(null);
	onMount(() => {
		if (mode === 'print') return;
		const id = setTimeout(() => {
			const ev = specimen.lab.evaluator();
			const th = Float64Array.from(specimen.theta0);
			for (let k = 0; k < 20; k++) { ev.loss(th); ev.grad(th); }
			const n = 60;
			let t0 = performance.now();
			for (let k = 0; k < n; k++) ev.loss(th);
			const f = (performance.now() - t0) / n;
			t0 = performance.now();
			for (let k = 0; k < n; k++) ev.grad(th);
			const fb = (performance.now() - t0) / n;
			timing = { f, fb };
		}, 400);
		return () => clearTimeout(id);
	});

	// ── geometry ──────────────────────────────────────────────────────────────
	const W = $derived(layoutWidth(width, mode, 340));
	const narrow = $derived(W < 700);
	const top = 58;
	const plotH = 300;
	const yMin = -20, yMax = 0; // log10 error axis, shared by both panels
	const Y = (v) => top + ((yMax - v) / (yMax - yMin)) * plotH;
	// left panel
	const lx0 = 44;
	const lw = $derived(narrow ? W - lx0 - 18 : Math.min(330, W * 0.36));
	const X = (e) => lx0 + ((-e - 1) / 12) * lw; // e = log10 h ∈ [−1, −13]
	// right panel (below on narrow screens)
	const rx0 = $derived(narrow ? lx0 : lx0 + lw + 64);
	const rw = $derived(narrow ? W - lx0 - 18 : W - rx0 - 18);
	const rTop = $derived(narrow ? top + plotH + 120 : top);
	const RY = (v) => rTop + ((yMax - v) / (yMax - yMin)) * plotH;
	const H = $derived(rTop + plotH + 80);

	// array groups: widths ∝ √size, small gaps
	const gx = $derived.by(() => {
		const gs = specimen.layout;
		const gapw = 5;
		const total = gs.reduce((s, l) => s + Math.sqrt(l.size), 0);
		const k = (rw - gapw * (gs.length - 1)) / total;
		let x = rx0;
		return gs.map((l) => {
			const w = Math.sqrt(l.size) * k;
			const out = { x, w, mid: x + w / 2 };
			x += w + gapw;
			return out;
		});
	});
	// a deterministic spread of the dots of one group across its width
	const spreadX = (gi, i) => {
		const l = specimen.layout[gi], G = gx[gi];
		const k = i - l.offset;
		const u = l.size > 1 ? ((k * 0.618034) % 1) : 0.5; // golden-ratio sequence: even, no jitter
		return G.x + 1.5 + u * Math.max(0, G.w - 3);
	};
	const pretty = (name) => {
		const base = name.split('.').at(-1);
		const m = base.match(/^(gamma|beta)(\d|f)$/);
		if (m) return { s: m[1] === 'gamma' ? 'γ' : 'β', sub: m[2] };
		const w = base.match(/^([Wb])(q|k|v|o|1|2|out)$/);
		if (w) return { s: w[1], sub: w[2] };
		return { s: base, sub: '' };
	};
	const u = 2 ** -53;
	const v = $derived(job.value);
	const roundLine = $derived(v ? (e) => Math.log10((u * v.loss) / 10 ** e) : null);
	const stripDots = $derived.by(() => {
		if (!v) return [];
		const err = v.strips[sel];
		const out = [];
		specimen.layout.forEach((l, gi) => {
			for (let i = l.offset; i < l.offset + l.size; i++) {
				const e = err[i];
				out.push({ i, x: spreadX(gi, i), y: RY(e == null ? yMin : clamp(e, yMin, yMax)), zero: e == null });
			}
		});
		return out;
	});
	const ticks = [0, -4, -8, -12, -16, -20];
	const powTxt = (e) => (e === 0 ? '1' : `10${String(e).replace('-', '⁻').replace(/\d/g, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[d])}`);

	function key(e) {
		const k = STRIPS.indexOf(sel);
		if (e.key === 'ArrowRight' && k < STRIPS.length - 1) { sel = STRIPS[k + 1]; e.preventDefault(); }
		if (e.key === 'ArrowLeft' && k > 0) { sel = STRIPS[k - 1]; e.preventDefault(); }
	}
	const readout = $derived.by(() => {
		if (!v) return null;
		const err = v.strips[sel].filter((x) => x != null);
		const worst = Math.max(...err);
		const gmax = Math.max(...v.groups.map((q) => q.gmax ?? -99));
		return {
			h: `h = ${powTxt(sel)}.`,
			worst: `The largest error over all ${v.p} parameters is ${(10 ** worst).toExponential(1).replace(/-/g, '−')}; the largest gradient entry is ${(10 ** gmax).toExponential(1).replace(/-/g, '−')}.`,
			rel: `As gradcheck reports it (the length of the difference over the larger of the two lengths): ${v.rel[sel].toExponential(1).replace(/-/g, '−')}.`
		};
	});
</script>

<div class="fx fx-gc" use:measure={(w) => (width = w)}>
	{@html job.tag}
	<svg viewBox="0 0 {W} {H}" width={W} height={H} role={interactive ? 'group' : 'img'}
		aria-label="Gradient check at θ₀: the error of central differences against the pulled-back gradient falls like h squared and then rises like 1 over h; for every parameter at h = {powTxt(sel)}. {readout ? readout.worst : ''}">
		<!-- shared log axis (left) -->
		{#each ticks as e (e)}
			<line class="rule" x1={lx0 - 4} x2={lx0 + lw} y1={Y(e)} y2={Y(e)} />
			<text class="lbl lbl--micro" x={lx0 - 8} y={Y(e) + 3.5} text-anchor="end">{powTxt(e)}</text>
		{/each}
		{#each Array.from({ length: 21 }, (_, k) => -k) as e (e)}
			<line class="axis" x1={lx0 - 3} x2={lx0} y1={Y(e)} y2={Y(e)} />
		{/each}
		<text class="lbl" x={lx0 - 8} y={top - 40}>|difference quotient − pulled-back gradient|</text>

		<!-- LEFT: error against h -->
		{#each HS as e (e)}
			<line class="axis" x1={X(e)} x2={X(e)} y1={top + plotH} y2={top + plotH + 4} />
			{#if e % 2 === -1 || e === -13}<text class="lbl lbl--micro" x={X(e)} y={top + plotH + 16} text-anchor="middle">{powTxt(e)}</text>{/if}
		{/each}
		<text class="lbl" x={lx0 + lw / 2} y={top + plotH + 34} text-anchor="middle">step <tspan class="m">h</tspan>, shrinking →</text>
		{#if v}
			<!-- the selected h: a hairline band -->
			<rect class="fx-gc__band" x={X(sel) - 7} y={top} width="14" height={plotH} />
			<!-- rounding: u ℓ / h, computed from the loss and the unit roundoff -->
			<line class="fx-gc__guide" x1={X(-1)} x2={X(-13)} y1={Y(roundLine(-1))} y2={Y(roundLine(-13))} />
			<text class="lbl lbl--micro fx-gc__gl" x={X(-1.2)} y={Y(roundLine(-1.2)) + 15}>rounding, <tspan class="m m--s">u ℓ</tspan>/<tspan class="m m--s">h</tspan></text>
			<!-- truncation: slope 2 through the first curve's first point -->
			{@const c0 = v.curves[0].err[0]}
			<line class="fx-gc__guide" x1={X(-1)} x2={X(-6)} y1={Y(c0 + 0.6)} y2={Y(c0 + 0.6 - 10)} />
			<text class="lbl lbl--micro fx-gc__gl" x={X(-2.2) + 6} y={Y(c0 + 0.6 - 2.4) - 4}>∝ <tspan class="m m--s">h</tspan>²</text>
			<g class="marks">
				{#each v.curves as c (c.i)}
					<polyline class="fx-gc__curve" points={c.err.map((y, k) => (y == null ? '' : `${X(HS[k])},${Y(y)}`)).join(' ')} />
					{#each c.err as y, k (k)}
						{#if y != null}<circle class="fx-gc__pt" cx={X(HS[k])} cy={Y(y)} r="2" />{/if}
					{/each}
				{/each}
			</g>
			<text class="lbl lbl--micro" x={lx0} y={top - 22}>one parameter of each of {#each CURVES as n, k (n)}{@const q = pretty(n)}<tspan class="m m--s">{q.s}</tspan>{#if q.sub}<tspan class="m m--s sub" dy="3">{q.sub}</tspan><tspan dy="-3"></tspan>{/if}{k < CURVES.length - 2 ? ', ' : k === CURVES.length - 2 ? ' and ' : ''}{/each}:</text>
			<text class="lbl lbl--micro" x={lx0} y={top - 8}>the entry with the largest gradient</text>
			<!-- clickable decades with a full check -->
			{#if interactive}
			{#each STRIPS as e (e)}
				<g class="fx-gc__pick" class:is-on={e === sel}>
					<circle cx={X(e)} cy={top + plotH + 46} r="4.5" />
					{#if interactive}
						<rect class="hit" x={X(e) - 14} y={top} width="28" height={plotH + 58} role="button" tabindex="0" aria-label="h = {powTxt(e)}" onclick={() => (sel = e)} onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); (sel = e); } }} />
					{/if}
				</g>
			{/each}
			{/if}
			{#if interactive}
				<rect class="hit" x={X(STRIPS[0]) - 10} y={top + plotH + 36} width={X(STRIPS.at(-1)) - X(STRIPS[0]) + 20} height="20" style="pointer-events: none" role="slider" tabindex="0"
					aria-label="step h for the check of every parameter" aria-valuemin="-10" aria-valuemax="-2" aria-valuenow={sel} aria-valuetext="h = {powTxt(sel)}" onkeydown={key} />
			{/if}
			{#if interactive}<text class="lbl lbl--micro" x={lx0} y={top + plotH + 66}>circles: the steps at which every parameter is checked</text>{/if}
		{/if}

		<!-- RIGHT: every parameter at the selected h -->
		{#each ticks as e (e)}
			<line class="rule" x1={rx0 - 4} x2={rx0 + rw} y1={RY(e)} y2={RY(e)} />
			<text class="lbl lbl--micro" x={rx0 - 8} y={RY(e) + 3.5} text-anchor="end">{powTxt(e)}</text>
		{/each}
		<text class="lbl lbl--head" x={rx0} y={rTop - 22}>all {specimen.paramCount} parameters at <tspan class="m">h</tspan> = {powTxt(sel)}</text>
		{#if v}
			<line class="fx-gc__guide" x1={rx0} x2={rx0 + rw} y1={RY(roundLine(sel))} y2={RY(roundLine(sel))} />
			{@const ly = RY(roundLine(sel))}
			{@const top = Math.min(ly - 5, ...stripDots.filter((d) => d.x > rx0 + rw - 80).map((d) => d.y - 9))}
			{#if top < ly - 6}<line class="leader" x1={rx0 + rw - 2} x2={rx0 + rw - 2} y1={top + 3} y2={ly} />{/if}
			<text class="lbl lbl--micro fx-gc__gl fx-gc__halo" x={rx0 + rw} y={top} text-anchor="end">rounding, <tspan class="m m--s">u ℓ</tspan>/<tspan class="m m--s">h</tspan></text>
			<g class="marks">
				{#each stripDots as d (d.i)}
					<circle class="fx-gc__dot" class:is-zero={d.zero} cx={d.x} cy={d.y} r="1.55" />
				{/each}
			</g>
			<!-- the size of the gradient in each array: a covector bar -->
			{#each v.groups as q, gi (q.name)}
				{#if q.gmax != null}
					<line class="fx-gc__gbar" x1={gx[gi].x} x2={gx[gi].x + Math.max(3, gx[gi].w)} y1={RY(q.gmax)} y2={RY(q.gmax)} />
				{/if}
			{/each}
		{:else}
			<text class="lbl lbl--micro" x={rx0 + rw / 2} y={rTop + plotH / 2} text-anchor="middle">differencing {Math.round(job.progress * 100)} % of the {2 * specimen.paramCount} forward passes …</text>
		{/if}
		{#each specimen.layout as l, gi (l.name)}
			{@const n = pretty(l.name)}
			<text class="m fx-gc__name" x={gx[gi].mid} y={rTop + plotH + 18 + (gi % 2) * 15} text-anchor="middle">{n.s}<tspan class="sub" dy="3">{n.sub}</tspan></text>
		{/each}
		<text class="lbl lbl--micro" x={rx0} y={rTop + plotH + 56}>each dot is one parameter; the covector bar</text>
		<text class="lbl lbl--micro" x={rx0} y={rTop + plotH + 70}>over an array is its largest gradient entry</text>
	</svg>
	{#if interactive}
		<p class="readout" aria-live="polite">
			{#if readout}
				<span class="readout__who">{readout.h}</span>
				<span class="readout__part">{readout.worst}</span>
				<span class="readout__part">{readout.rel}</span>
			{/if}
			{#if timing}
				<span class="readout__part">In this browser one forward pass takes {fmt(timing.f, 2)} ms and a forward and backward pass {fmt(timing.fb, 2)} ms, the cost of {fmt(timing.fb / timing.f, 1)} forward passes. Checking every parameter at one step took {2 * specimen.paramCount} forward passes.</span>
			{/if}
			<span class="readout__hint">Click a circle (or use ←→) to check every parameter at that step.</span>
		</p>
	{/if}
</div>

<style>
	.fx-gc :global(.m) { font-family: var(--serif); font-style: italic; font-size: 14px; fill: var(--ink); }
	.fx-gc :global(.m--s) { font-size: 12px; fill: inherit; }
	.fx-gc :global(.sub) { font-size: 10px; }
	.fx-gc__name { font-size: 13px; fill: var(--ink-2); }
	.fx-gc__band { fill: var(--value-tint); opacity: 0.55; }
	.fx-gc__guide { stroke: var(--ink-3); stroke-width: var(--hair); stroke-dasharray: 3 3; }
	.fx-gc__gl { fill: var(--ink-3); }
	.fx-gc__halo { paint-order: stroke; stroke: var(--paper); stroke-width: 3px; stroke-linejoin: round; }
	.fx-gc__curve { fill: none; stroke: var(--ink); stroke-width: var(--line); }
	.fx-gc__pt { fill: var(--ink); }
	.fx-gc__dot { fill: var(--ink-2); transition: cy var(--t-move) var(--ease); }
	.fx-gc__dot.is-zero { fill: var(--paper); stroke: var(--ink-2); stroke-width: var(--hair); }
	.fx-gc__gbar { stroke: var(--covector); stroke-width: 2.2px; }
	.fx-gc__pick circle { fill: var(--paper); stroke: var(--ink-3); stroke-width: 1px; }
	.fx-gc__pick.is-on circle { fill: var(--ink); stroke: var(--ink); }
	.fx-gc__pick { cursor: pointer; }
</style>

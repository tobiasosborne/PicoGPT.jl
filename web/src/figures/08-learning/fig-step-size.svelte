<script module>
	export const meta = { size: 'wide', title: 'How large a step can gradient descent take?' };
</script>

<script>
	// Figure: plain gradient descent on the specimen (full batch, no momentum, no schedule,
	// no clipping, no decay) at three learning rates η, as small multiples sharing axes.
	// Top of each column: the loss (LossChart). Below it: the sharpness Λ(θ_t), the largest
	// eigenvalue of the Hessian of the loss at the current parameters, by power iteration
	// on Hessian–vector products (central differences of the pulled-back gradient,
	// warm-started from the previous checkpoint). The hairline is 2/η: the descent lemma
	// guarantees a decrease only while Λ < 2/η. Gradient descent drives Λ up to that line
	// and holds it there.
	import { getContext } from 'svelte';
	import { measure } from '../primitives/measure.js';
	import { computed } from '../primitives/computed.svelte.js';
	import LossChart from '../primitives/LossChart.svelte';
	import { fmt, layoutWidth } from '../primitives/scales.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const interactive = $derived(mode !== 'print');

	const ETAS = [0.25, 0.5, 1];
	const STEPS = 600;
	const CHECK = [0, 3, 6, 10, 13, 16, 20, 23, 26, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 250, 300, 400, 500, 600];

	/**
	 * Top Hessian eigenvalue at θ by power iteration on Hessian–vector products, each a
	 * central difference of the pulled-back gradient: Hv ≈ (∇ℓ(θ+hv) − ∇ℓ(θ−hv))/2h.
	 * The loss is only piecewise smooth (ReLU), and a difference that straddles a kink is
	 * meaningless; so every estimate is checked by recomputing vᵀHv at a step 3× smaller,
	 * and recomputed at smaller h until the two agree to 2%. Returns {lambda, v, h} or
	 * {lambda: null} if no step size passes (then the point is not drawn).
	 */
	function sharpness(ev, theta, v0, iters) {
		const p = theta.length;
		const hv = (v, h) => {
			const a = Float64Array.from(theta), b = Float64Array.from(theta);
			for (let i = 0; i < p; i++) { a[i] += h * v[i]; b[i] -= h * v[i]; }
			const ga = ev.grad(a), gb = ev.grad(b);
			for (let i = 0; i < p; i++) ga[i] = (ga[i] - gb[i]) / (2 * h);
			return ga;
		};
		const rq = (v, h) => { const w = hv(v, h); let s = 0; for (let i = 0; i < p; i++) s += w[i] * v[i]; return s; };
		for (const h of [1e-5, 1e-6, 1e-7]) {
			let v = v0 ? Float64Array.from(v0) : Float64Array.from({ length: p }, (_, i) => Math.sin(1.7 * i + 0.3));
			let n = Math.hypot(...v);
			for (let i = 0; i < p; i++) v[i] /= n;
			let lambda = 0;
			for (let k = 0; k < iters; k++) {
				const w = hv(v, h);
				lambda = 0;
				for (let i = 0; i < p; i++) lambda += w[i] * v[i];
				n = Math.hypot(...w);
				for (let i = 0; i < p; i++) v[i] = w[i] / n;
			}
			const check = rq(v, h / 3);
			if (Math.abs(check - lambda) <= 0.02 * Math.abs(lambda)) return { lambda, v, h };
		}
		return { lambda: null, v: v0 };
	}

	// svelte-ignore state_referenced_locally
	const job = computed('fig-step-size:v4', function* () {
		const ev = specimen.lab.evaluator();
		const runs = [];
		let v = null;
		for (const eta of ETAS) {
			const run = specimen.lab.run({ optimizer: 'gd', eta, schedule: null, weightDecay: 0, clip: Infinity });
			const sharp = [];
			v = null; // each run starts from θ₀: start the iteration afresh
			for (let k = 0; k <= STEPS; k++) {
				if (CHECK.includes(k)) {
					const r = sharpness(ev, run.theta, v, v ? 24 : 50);
					if (r.lambda != null) { v = r.v; sharp.push([k, r.lambda]); }
					yield (ETAS.indexOf(eta) + k / STEPS) / (ETAS.length + 1);
				}
				if (k < STEPS) run.step();
			}
			const loss = run.loss.slice();
			loss.push(run.currentLoss());
			runs.push({ eta, loss, sharp });
		}
		// the curvature where the book's AdamW run ends
		const trained = sharpness(ev, specimen.theta, null, 60).lambda;
		return { runs, trained };
	});

	const data = $derived(job.value);

	let width = $state(868);
	const W = $derived(layoutWidth(width, mode, 340));
	const narrow = $derived(W < 640);
	const gap = $derived(narrow ? 10 : 26);
	const cols = $derived(narrow ? 1 : 3);
	const colW = $derived((W - 10 - gap * (cols - 1)) / cols); // 10 px for the last tick label
	const lossH = 190, sharpH = 128, head = 34;
	const blockH = $derived(head + lossH + 12 + sharpH + 28);
	const H = $derived((narrow ? ETAS.length : 1) * blockH);
	const pos = (c) => (narrow ? { x: 0, y: c * blockH } : { x: c * (colW + gap), y: 0 });

	// the sharpness strip uses the LossChart's own x mapping (compact: L = 26, R = 6)
	const Lc = 26, Rc = 6;
	const xs = (k) => Lc + (colW - Lc - Rc) * (Math.log1p(k) / Math.log1p(STEPS));
	const lamMax = $derived(data ? Math.ceil(Math.max(...data.runs.flatMap((r) => r.sharp.map((s) => s[1])), ...ETAS.map((e) => 2 / e)) / 4) * 4 : 10);
	const ys = (lam, y0) => y0 + sharpH * (1 - Math.max(0, lam) / lamMax);
	const lamStep = $derived(lamMax > 12 ? 4 : 2);
	const lamTicks = $derived(Array.from({ length: Math.floor(lamMax / lamStep) + 1 }, (_, i) => i * lamStep));
	const lossMax = $derived(data ? Math.ceil(Math.max(Math.log(specimen.config.V), ...data.runs.flatMap((r) => r.loss.filter(Number.isFinite))) * 2 + 0.25) / 2 : 2.5);

	let cursor = $state(null); // a step
	const nearest = (k) => CHECK.reduce((a, b) => (Math.abs(b - k) < Math.abs(a - k) ? b : a), 0);
	const onpoint = (k) => (cursor = k == null ? null : nearest(k));

	const readout = $derived.by(() => {
		if (!data) return null;
		if (cursor == null) {
			return {
				who: 'Gradient descent from the same θ₀ at three learning rates.',
				parts: data.runs.map((r) => `η = ${r.eta}: Λ ends at ${fmt(r.sharp.at(-1)[1], 2)}, against 2/η = ${fmt(2 / r.eta, 0)}`).join('; '),
				extra: `At θ₀, Λ = ${fmt(data.runs[0].sharp[0][1], 2)}.` + (data.trained != null ? ` Where the book’s AdamW run ends, Λ = ${fmt(data.trained, 1)}: plain gradient descent there would need η below ${fmt(2 / data.trained, 3)}.` : '')
			};
		}
		const lam = (r) => r.sharp.find((q) => q[0] === cursor)?.[1];
		return {
			who: `Step ${cursor}.`,
			parts: data.runs.map((r) => `η = ${r.eta}: loss ${fmt(r.loss[cursor], 3)}, ${lam(r) != null ? `Λ = ${fmt(lam(r), 2)}` : 'Λ not resolved (a kink)'} (2/η = ${fmt(2 / r.eta, 0)})`).join('; '),
			extra: ''
		};
	});
</script>

<div class="fx fx-ss" use:measure={(w) => (width = w)}>
	{@html job.tag}
	{#if !data}
		<svg viewBox="0 0 {W} 70" width={W} height="70" role={interactive ? 'group' : 'img'} aria-label="computing three gradient-descent runs and their curvature">
			<text class="lbl lbl--micro" x="0" y="22">running gradient descent three times and measuring the curvature along the way…</text>
			<line class="rule" x1="0" x2={W} y1="36" y2="36" />
			<line class="fx-ss__progress" x1="0" x2={W * job.progress} y1="36" y2="36" />
		</svg>
	{:else}
		<svg viewBox="0 0 {W} {H}" width={W} height={H} role={interactive ? 'group' : 'img'} aria-label="{readout.who} {readout.parts}">
			{#each data.runs as r, c (r.eta)}
				{@const p = pos(c)}
				{@const y0 = p.y + head + lossH + 12}
				<g transform="translate({p.x} {p.y})">
					<text class="lbl lbl--head" x={Lc} y="13"><tspan class="m">η</tspan> = {r.eta}</text>
					{#if c === 0}<text class="lbl lbl--micro" x={Lc + 52} y="13">plain gradient descent from θ₀</text>{/if}
				</g>
				<LossChart x={p.x} y={p.y + head - 12} width={colW} height={lossH} compact xScale="log" xMax={STEPS}
					series={[{ values: r.loss, style: 'main' }]} yMax={lossMax} lnV={Math.log(specimen.config.V)} floor={specimen.lab.floor}
					refLabels={c === 0 || narrow} cursor={cursor} onpoint={interactive ? onpoint : null} yLabel={c === 0 || narrow ? 'loss' : ''} xLabel="" />
				<!-- sharpness -->
				<g transform="translate({p.x} 0)">
					<line class="axis" x1={Lc} x2={Lc} y1={y0} y2={y0 + sharpH} />
					<line class="axis" x1={Lc} x2={colW - Rc} y1={y0 + sharpH} y2={y0 + sharpH} />
					{#each lamTicks as t (t)}
						<line class="axis" x1={Lc - 3.5} x2={Lc} y1={ys(t, y0)} y2={ys(t, y0)} />
						<text class="lbl lbl--micro" x={Lc - 6} y={ys(t, y0) + 3.5} text-anchor="end">{t}</text>
					{/each}
					{#each [0, 1, 10, 100, STEPS] as k (k)}
						<line class="axis" x1={xs(k)} x2={xs(k)} y1={y0 + sharpH} y2={y0 + sharpH + 3.5} />
						<text class="lbl lbl--micro" x={xs(k)} y={y0 + sharpH + 13} text-anchor="middle">{k}</text>
					{/each}
					<text class="lbl lbl--micro" x={colW - Rc} y={y0 + sharpH + 25} text-anchor="end">log(1 + step)</text>
					{#if c === 0 || narrow}<text class="fx-ss__lam" x={Lc - 6} y={y0 - 5} text-anchor="end"><tspan class="m">Λ</tspan></text>{/if}
					<!-- the stability line 2/η -->
					<line class="fx-ss__limit" x1={Lc} x2={colW - Rc} y1={ys(2 / r.eta, y0)} y2={ys(2 / r.eta, y0)} />
					<text class="lbl fx-ss__limittext" x={colW - Rc} y={ys(2 / r.eta, y0) - 5} text-anchor="end">2/<tspan class="m">η</tspan> = {fmt(2 / r.eta, 0)}</text>
					<polyline class="fx-ss__line" points={r.sharp.map(([k, l]) => `${xs(k)},${ys(l, y0)}`).join(' ')} />
					{#each r.sharp as [k, l] (k)}
						<circle class="fx-ss__dot" class:is-on={cursor === k} cx={xs(k)} cy={ys(l, y0)} r={cursor === k ? 3.4 : 2.3} />
					{/each}
					{#if c === 0}
						{@const first = r.sharp[0]}
						<text class="lbl lbl--micro" x={xs(first[0]) + 5} y={ys(first[1], y0) - 6}>at θ₀: {fmt(first[1], 2)}</text>
					{/if}
					{#if cursor != null}
						<line class="fx-ss__cursor" x1={xs(cursor)} x2={xs(cursor)} y1={y0} y2={y0 + sharpH} />
					{/if}
				</g>
			{/each}
		</svg>
		{#if interactive}
			<p class="readout" aria-live="polite">
				<span class="readout__who">{readout.who}</span>
				<span class="readout__part">{readout.parts}.</span>
				{#if readout.extra}<span class="readout__part">{readout.extra}</span>{/if}
				<span class="readout__hint">Point along a loss curve to read every run at that step.</span>
			</p>
		{/if}
	{/if}
</div>

<style>
	.fx-ss :global(.m) { font-family: var(--serif); font-style: italic; }
	.fx-ss__lam { font-size: 15px; fill: var(--ink); }
	.fx-ss__limit { stroke: var(--ink); stroke-width: 0.9px; stroke-dasharray: 5 3; }
	.fx-ss__limittext { fill: var(--ink); paint-order: stroke; stroke: var(--paper); stroke-width: 3px; stroke-linejoin: round; }
	.fx-ss__line { fill: none; stroke: var(--value); stroke-width: var(--hair); }
	.fx-ss__dot { fill: var(--value); }
	.fx-ss__dot.is-on { fill: var(--paper); stroke: var(--value); stroke-width: 1.4px; }
	.fx-ss__cursor { stroke: var(--ink); stroke-width: var(--hair); }
	.fx-ss__progress { stroke: var(--value); stroke-width: 1.5px; }
</style>

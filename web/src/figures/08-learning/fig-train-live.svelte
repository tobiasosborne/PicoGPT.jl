<script module>
	export const meta = { size: 'wide', title: 'The specimen learning, live' };
	export { default as Key } from './fig-train-live.key.svelte';
</script>

<script>
	// Figure: press train and the specimen learns, in the browser, step by step.
	// Every step is the engine's forward pass, backward pass and an AdamW (or plain
	// gradient-descent) update of all p parameters: specimen.lab.run(opts), which with the
	// default recipe and seed 0 reproduces the book's run (artifacts/specimen.json).
	//   left:   the loss curve (LossChart), with greedy samples of "to be or" in the margin
	//           at checkpoints; below it the learning rate η_k, whose axis carries a handle
	//   right:  every parameter array as a Hinton diagram (one shared scale), in forward
	//           order; below, what one character of one window reads (Arcs, both heads)
	// State is a pure function of (seed, optimiser, schedule, η, number of updates).
	// Changing any setting starts again from θ₀; the run just left becomes a dashed ghost.
	// Print: the loss curve of the book's run with its samples, and the parameters and
	// attention at four steps as small multiples.
	import { getContext, onDestroy, untrack } from 'svelte';
	import { measure } from '../primitives/measure.js';
	import { computed } from '../primitives/computed.svelte.js';
	import LossChart from '../primitives/LossChart.svelte';
	import MatrixGlyph from '../primitives/MatrixGlyph.svelte';
	import Arcs from '../primitives/Arcs.svelte';
	import { fmt, showChar, clamp, layoutWidth, reducedMotion } from '../primitives/scales.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = getContext('specimen')();
	const lab = specimen.lab;
	const { T, V } = specimen.config;
	const lnV = Math.log(V);
	const recipe = lab.recipe;
	const interactive = $derived(mode !== 'print');

	// the window and reader whose attention is drawn
	// svelte-ignore state_referenced_locally
	const wStart = clamp((+attrs.window || 1) - 1, 0, specimen.windows.length - 1);
	// svelte-ignore state_referenced_locally
	const reader = clamp((+attrs.reader || T) - 1, 0, T - 1);
	const wText = specimen.corpus.slice(wStart, wStart + T);
	const wIds = specimen.ids(wText);
	const PROMPT = specimen.corpus.slice(0, T);
	const CHECK = [0, 5, 10, 20, 30, 50, 200, 1000, 5000];
	const STOPS = [200, 1000, 5000];
	const ev = lab.evaluator();
	const bk = specimen.layout.find((l) => l.name.endsWith('bk'));
	// one scale for every parameter, fixed for the book: the largest |θᵢ| at θ₀ or trained
	const vmax = Math.max(...specimen.theta0.map(Math.abs), ...specimen.theta.map(Math.abs));

	// ── settings (the reader's) and the run ─────────────────────────────────────────
	let seed = $state(0);
	let optName = $state('adamw');
	let sched = $state('cosine');
	let eta = $state(recipe.eta);
	const ETA_MIN = 1e-4, ETA_MAX = 3;

	function options() {
		return {
			theta0: lab.initTheta(seed),
			optimizer: optName,
			eta,
			schedule: sched === 'cosine' ? { ...recipe.schedule, etaMin: recipe.schedule.etaMin * (eta / recipe.eta) } : null
		};
	}
	const isBook = () => seed === 0 && optName === 'adamw' && sched === 'cosine' && Math.abs(eta - recipe.eta) < 1e-12;
	const isBookNow = $derived(seed === 0 && optName === 'adamw' && sched === 'cosine' && Math.abs(eta - recipe.eta) < 1e-12);

	let run = null; // not reactive: the engine's buffers
	let k = $state(0);
	let losses = $state.raw([]);
	let thetaNow = $state.raw(Float64Array.from(specimen.theta0));
	let attn = $state.raw(null);
	let samples = $state.raw([]);
	let ghost = $state.raw(null); // the run just abandoned: {values, label}
	let diverged = $state(false);
	let playing = $state(false);
	let gnorm = $state(NaN), rateNow = $state(NaN);

	function snapshot() {
		const cur = run.currentLoss();
		losses = [...run.loss, cur];
		thetaNow = Float64Array.from(run.theta);
		attn = ev.look(run.theta, wIds).A;
		k = run.t;
		gnorm = run.gradNorm.at(-1) ?? NaN;
		rateNow = run.lr.at(-1) ?? NaN;
		if (!Number.isFinite(cur)) diverged = true;
	}
	function sampleIfCheckpoint() {
		if (CHECK.includes(run.t) && !samples.some((s) => s.step === run.t)) {
			const cur = run.loss[run.t] ?? run.currentLoss();
			if (Number.isFinite(cur)) samples = [...samples, { step: run.t, loss: cur, text: lab.greedy(run.theta, PROMPT, 10), prompt: PROMPT }];
		}
	}
	function restart(keepGhost = true) {
		if (keepGhost && run && run.t > 0) ghost = { values: losses.slice(), label: 'previous run' };
		run = lab.run(options());
		samples = [];
		diverged = false;
		sampleIfCheckpoint();
		snapshot();
	}
	function advance(n) {
		for (let i = 0; i < n; i++) {
			const l = run.step();
			if (!Number.isFinite(l)) { diverged = true; break; }
			sampleIfCheckpoint();
		}
	}

	// the static states: print shows the book's run (its own Run); interactive starts at θ₀
	function bookRun() {
		const r = lab.run({});
		const smp = [];
		for (let s = 0; s <= 200; s++) {
			if (CHECK.includes(s)) smp.push({ step: s, loss: r.currentLoss(), text: lab.greedy(r.theta, PROMPT, 10), prompt: PROMPT });
			if (s < 200) r.step();
		}
		return { losses: [...r.loss, r.currentLoss()], samples: smp };
	}
	const printed = $derived(mode === 'print' ? bookRun() : null);
	// the book's samples, set in ghost ink at checkpoints the reader's run has not reached
	const bookSamples = computed(`fig-train-live:samples:${PROMPT}:${CHECK.join(',')}`, function* () {
		const r = lab.run({});
		const out = [];
		for (let s = 0; s <= 200; s++) {
			if (CHECK.includes(s)) { out.push({ step: s, loss: r.currentLoss(), text: lab.greedy(r.theta, PROMPT, 10), prompt: PROMPT }); yield s / 200; }
			if (s < 200) r.step();
		}
		return out;
	});
	const shownSamples = $derived.by(() => {
		const live = samples;
		const ghosts = attrs.t == null && isBookNow && bookSamples.value ? bookSamples.value.filter((g) => g.step <= xMax && !live.some((l) => l.step === g.step)).map((g) => ({ ...g, ghost: true })) : [];
		return [...live, ...ghosts];
	});
	restart(false);
	// Film time follows the log step axis, 0…200. Seeking backwards reconstructs
	// the seeded run, so setT is independent of frame order and wall-clock speed.
	$effect(() => {
		if (attrs.t == null) return;
		const target = Math.round(201 ** clamp(+attrs.t, 0, 1) - 1);
		untrack(() => {
			if (target < run.t) restart(false);
			if (target !== run.t) { advance(target - run.t); snapshot(); }
		});
	});

	// ── play loop ───────────────────────────────────────────────────────────────────
	let raf = 0;
	const nextStop = () => STOPS.find((s) => s > k) ?? Infinity;
	function frame() {
		if (!playing) return;
		const stop = nextStop();
		const t0 = performance.now();
		// about one step per frame early on (the log axis spreads them out), more later
		const want = Math.max(1, Math.round(k / 80));
		let done = 0;
		while (done < want && run.t < stop && !diverged && performance.now() - t0 < 9) { advance(1); done++; }
		snapshot();
		if (run.t >= stop || diverged) { playing = false; return; }
		raf = requestAnimationFrame(frame);
	}
	function play() {
		if (playing) { playing = false; cancelAnimationFrame(raf); return; }
		if (diverged || k >= STOPS.at(-1)) restart();
		if (reducedMotion()) {
			// no animation: jump to the next checkpoint
			const target = Math.min(nextStop(), CHECK.find((c) => c > k) ?? nextStop());
			while (run.t < target && !diverged) advance(1);
			snapshot();
			return;
		}
		playing = true;
		raf = requestAnimationFrame(frame);
	}
	function stepOnce() { playing = false; cancelAnimationFrame(raf); if (!diverged) { advance(1); snapshot(); } }
	function reset() { playing = false; cancelAnimationFrame(raf); restart(); }
	function change(fn) { const was = playing; playing = false; cancelAnimationFrame(raf); fn(); restart(); if (was) play(); }
	onDestroy(() => { playing = false; if (typeof cancelAnimationFrame !== 'undefined') cancelAnimationFrame(raf); });

	// ── geometry ────────────────────────────────────────────────────────────────────
	let width = $state(868);
	const W = $derived(layoutWidth(width, mode));
	const narrow = $derived(W < 700);
	const leftW = $derived(narrow ? W : Math.round(W * 0.55));
	const rightX = $derived(narrow ? 0 : leftW + 24);
	const rightW = $derived(narrow ? W : W - leftW - 24);
	const sampW = $derived(narrow ? 0 : 160);
	const xMax = $derived(k > 1000 ? 5000 : k > 200 ? 1000 : 200);
	const chartH = $derived(!interactive ? 260 : narrow ? 240 : 368);
	const etaY = $derived(chartH + 10);
	const etaH = 118;
	const leftH = $derived(etaY + etaH + 10);

	// the learning-rate strip shares the chart's x-axis geometry (LossChart: L = 34, R = sampW + 10)
	const eL = 34;
	const ePW = $derived(leftW - eL - sampW - 10);
	// (LossChart's right padding is margin + 10)
	const ex = (s) => eL + (ePW * Math.log1p(Math.max(0, s))) / Math.log1p(xMax);
	const eTop = $derived(etaY + 16), eBot = $derived(etaY + etaH - 22);
	const ey = (v) => eBot - ((Math.log10(clamp(v, ETA_MIN, ETA_MAX)) - Math.log10(ETA_MIN)) / (Math.log10(ETA_MAX) - Math.log10(ETA_MIN))) * (eBot - eTop);
	const yToEta = (py) => 10 ** (Math.log10(ETA_MIN) + ((eBot - clamp(py, eTop, eBot)) / (eBot - eTop)) * (Math.log10(ETA_MAX) - Math.log10(ETA_MIN)));
	const planned = $derived.by(() => {
		const o = options();
		const pts = [];
		for (let s = 1; s <= xMax; s = s < 30 ? s + 1 : Math.ceil(s * 1.06)) pts.push([s, o.schedule ? lab.scheduleRate(s, { eta: o.eta, ...o.schedule }) : o.eta]);
		return pts;
	});
	const pastPath = $derived.by(() => {
		if (!run) return '';
		let d = '';
		const lr = losses.length ? run.lr : [];
		for (let i = 0; i < lr.length; i++) d += `${i ? 'L' : 'M'}${ex(i + 1).toFixed(1)} ${ey(lr[i]).toFixed(1)}`;
		return d;
	});
	const planPath = $derived(planned.map(([s, v], i) => `${i ? 'L' : 'M'}${ex(s).toFixed(1)} ${ey(v).toFixed(1)}`).join(''));

	// parameter field: forward order, two lines (cells in units of c)
	const groupsOf = () => [
		{ line: 0, items: ['E'], label: 'E' },
		{ line: 0, items: ['gamma1', 'beta1'], label: 'γ_1 β_1' },
		{ line: 0, items: ['Wq', 'bq'], label: 'W_q b_q' },
		{ line: 0, items: ['Wk', 'bk'], label: 'W_k b_k' },
		{ line: 0, items: ['Wv', 'bv'], label: 'W_v b_v' },
		{ line: 0, items: ['Wo', 'bo'], label: 'W_o b_o' },
		{ line: 1, items: ['gamma2', 'beta2'], label: 'γ_2 β_2' },
		{ line: 1, items: ['W1', 'b1'], label: 'W_1 b_1' },
		{ line: 1, items: ['W2', 'b2'], label: 'W_2 b_2', col: 'right' },
		{ line: 1, items: ['gammaf', 'betaf'], label: 'γ_f β_f', col: 'right', row: 1 },
		{ line: 1, items: ['Wout', 'bout'], label: 'W_out b_out', col: 'right', row: 1 }
	];
	const byName = Object.fromEntries(specimen.layout.map((l) => [l.name.split('.').at(-1), l]));
	const shapeOf = (n) => { const s = byName[n].shape; return s.length === 2 ? s : [s[0], 1]; };
	const field = $derived.by(() => {
		const GI = 0.5, GG = 1.7, LAB = 2.2; // gaps (cells): within a group, between groups, label band
		const placeGroups = (gs, x0, y0) => {
			let x = x0, hmax = 0;
			const out = [];
			for (const g of gs) {
				let gx = x;
				const its = g.items.map((n) => { const [r, cc] = shapeOf(n); const it = { n, r, c: cc, x: gx, y: y0 + LAB }; gx += cc + GI; return it; });
				const gw = gx - GI - x;
				// the name must fit too: about 6.4 px per visible character, at a nominal 7 px cell
				const lw = (g.label.replace(/[_ ]/g, '').length * 6.4 + 6) / 7;
				out.push({ ...g, x, w: gw, y: y0, its });
				hmax = Math.max(hmax, LAB + Math.max(...its.map((i) => i.r)));
				x += Math.max(gw, lw) + GG;
			}
			return { out, w: x - GG - x0, h: hmax };
		};
		const gs = groupsOf();
		const l0 = placeGroups(gs.filter((g) => g.line === 0), 0, 0);
		const y1 = l0.h + 2.2;
		const left1 = placeGroups(gs.filter((g) => g.line === 1 && !g.col), 0, y1);
		const rx = left1.w + GG;
		const r0 = placeGroups(gs.filter((g) => g.col && !g.row), rx, y1);
		const r1 = placeGroups(gs.filter((g) => g.col && g.row), rx, y1 + r0.h + 1.4);
		const all = [...l0.out, ...left1.out, ...r0.out, ...r1.out];
		const wCells = Math.max(l0.w, rx + Math.max(r0.w, r1.w));
		const hCells = Math.max(y1 + left1.h, y1 + r0.h + 1.4 + r1.h);
		return { groups: all, wCells, hCells };
	});
	const cell = $derived(Math.min(narrow ? 6.4 : 8, (rightW - 4) / field.wCells));
	const fieldH = $derived(field.hCells * cell);
	const valuesOf = (theta, n) => { const l = byName[n]; return theta.subarray ? theta.subarray(l.offset, l.offset + l.size) : theta.slice(l.offset, l.offset + l.size); };

	// attention of one reader
	const arcsTop = $derived(narrow ? leftH + fieldH + 44 : fieldH + 62);
	const arcStep = $derived((rightW - 16) / T);
	const arcXs = $derived(Array.from({ length: T }, (_, j) => 8 + arcStep * (j + 0.5)));
	const arcUp = $derived(Math.min(60, (arcStep * (T - 1)) / 2 * 0.5 + 8));
	const arcBase = $derived(arcsTop + arcUp + 30);
	const H = $derived(narrow ? arcBase + arcUp + 34 : Math.max(leftH, arcBase + arcUp + 30));

	// η handle
	let dragging = $state(false);
	function svgPoint(svg, e) { const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY; return pt.matrixTransform(svg.getScreenCTM().inverse()); }
	function eDown(e) { dragging = true; e.currentTarget.setPointerCapture(e.pointerId); eMove(e); }
	function eMove(e) {
		if (!dragging) return;
		let v = yToEta(svgPoint(e.currentTarget.ownerSVGElement, e).y);
		if (Math.abs(Math.log(v / recipe.eta)) < 0.08) v = recipe.eta;
		v = Number(v.toPrecision(2));
		if (v !== eta) change(() => (eta = v));
	}
	function eUp() { dragging = false; }
	function eKey(e) {
		const f = e.shiftKey ? 2 : 1.25;
		if (e.key === 'ArrowUp') { change(() => (eta = Number(Math.min(ETA_MAX, eta * f).toPrecision(2)))); e.preventDefault(); }
		if (e.key === 'ArrowDown') { change(() => (eta = Number(Math.max(ETA_MIN, eta / f).toPrecision(2)))); e.preventDefault(); }
		if (e.key === 'Home' || e.key === '0') { change(() => (eta = recipe.eta)); e.preventDefault(); }
	}
	const etaTicks = [1e-4, 1e-3, 1e-2, 1e-1, 1];
	const sup = (d) => String(d).replace(/-/, '⁻').replace(/[0-9]/g, (c) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+c]);
	const sci = (x) => {
		if (!Number.isFinite(x)) return '—';
		if (x === 0) return '0';
		const e = Math.floor(Math.log10(Math.abs(x)));
		if (e >= -2) return fmt(x, e >= 0 ? 2 : 4);
		return `${fmt(x / 10 ** e, 1)} × 10${sup(e)}`;
	};

	// readout
	const lossNow = $derived(losses.at(-1));
	const bkMax = $derived(Math.max(...valuesOf(thetaNow, 'bk').map(Math.abs)));
	const optLabel = $derived(optName === 'adamw' ? 'AdamW' : 'gradient descent');
	const ghostSeries = $derived.by(() => {
		const out = attrs.t == null ? [{ values: specimen.loss, label: 'the book’s run', labelAt: 12, style: 'ghost' }] : [];
		if (ghost) out.push({ values: ghost.values, label: ghost.label, labelAt: Math.min(ghost.values.length - 1, 30), style: 'dash' });
		out.push({ values: losses, style: 'main' });
		return out;
	});
	const fieldGroups = $derived(field.groups);
	const sub = (s) => s.split(/(_\w+|[₀-₉]+)/);
</script>

{#snippet params(theta, x0, y0, c, labels = true, small = false)}
	<g transform="translate({x0} {y0})">
		{#each fieldGroups as g (g.label)}
			{#if labels && (!small || g.items[0].length > 1)}
				<text class="fx-live__pname" class:fx-live__pname--small={small} x={g.x * c} y={(g.y + 1.55) * c}
					>{#each (small ? g.label.split(' ').slice(0, 1) : g.label.split(' ')) as part, pi (pi)}{@const m = part.match(/^([^_]+)_?(.*)$/)}{pi ? ' ' : ''}<tspan class="fx-live__sym">{m[1]}</tspan>{#if m[2]}<tspan class="fx-live__sub" dy="2.5">{m[2]}</tspan><tspan dy="-2.5"> </tspan>{/if}{/each}</text
				>
			{/if}
			{#each g.its as it (it.n)}
				<MatrixGlyph get={(i, j) => valuesOf(theta, it.n)[i + it.r * j]} m={it.r} n={it.c} x={it.x * c} y={it.y * c} cell={c} {vmax} />
			{/each}
		{/each}
	</g>
{/snippet}

{#snippet reads(A, y, xs, up, small = false)}
	{#if A}
		<Arcs {xs} y={y - (small ? 7 : 10)} A={A[0]} {reader} dir={-1} wmax={small ? 5 : 7} hmax={up} aspect={0.5} loop={small ? 5 : 7} labels={!small} labelAt={0.25} />
		<Arcs {xs} y={y + (small ? 4 : 6)} A={A[1]} {reader} dir={1} wmax={small ? 5 : 7} hmax={up} aspect={0.5} loop={small ? 5 : 7} labels={!small} labelAt={0.25} />
	{/if}
	{#each Array.from(wText) as ch, j (j)}
		<text class="tok fx-live__char" class:tok--reader={j === reader} class:fx-live__char--small={small} x={xs[j]} y={y + (small ? 3 : 5)} text-anchor="middle">{showChar(ch)}</text>
	{/each}
{/snippet}

<div class="fx fx-live" data-step={k} use:measure={(w) => (width = w)}>
	{#if interactive}
		{@html bookSamples.tag}
		<svg viewBox="0 0 {W} {H}" width={W} height={H} role={interactive ? 'group' : 'img'}
			aria-label="Training the specimen: step {k}, loss {fmt(lossNow ?? NaN, 4)} nats; uniform guessing {fmt(lnV, 2)}, entropy floor {fmt(lab.floor, 4)}.">
			<LossChart x={0} y={0} width={leftW} height={chartH} series={ghostSeries} {xMax} {lnV} floor={lab.floor} samples={narrow ? [] : shownSamples} rows={narrow ? 0 : CHECK.filter((c) => c <= xMax).length} margin={sampW} yMax={2.5} xLabel="" />

			<!-- the learning rate: η_k for the steps taken (solid) and planned (dotted); drag its axis -->
			<g class="fx-live__eta">
				<text class="lbl lbl--micro" x={eL - 6} y={eTop - 8} text-anchor="end">η<tspan class="fx-live__k" dy="2">k</tspan></text>
				<line class="axis" x1={eL} x2={eL} y1={eTop} y2={eBot} />
				<line class="axis" x1={eL} x2={eL + ePW} y1={eBot} y2={eBot} />
				{#each etaTicks as v (v)}
					<line class="axis" x1={eL - 3} x2={eL} y1={ey(v)} y2={ey(v)} />
					{#if v !== 1e-3 && v !== 1e-1}<text class="lbl lbl--micro" x={eL - 6} y={ey(v) + 3.5} text-anchor="end">{v === 1 ? '1' : `10${sup(Math.log10(v))}`}</text>{/if}
				{/each}
				<path class="fx-live__plan" d={planPath} />
				<path class="fx-live__past" d={pastPath} />
				<text class="lbl lbl--micro" x={eL + ePW} y={eBot + 14} text-anchor="end">η of each step: taken (solid), planned (dotted)</text>
				<g transform="translate({eL} {ey(eta)})">
					<circle class="handle" class:handle--active={dragging} r="6" />
					<text class="lbl fx-live__etalabel" x="12" y="-7"><tspan class="m">η</tspan> = {sci(eta)}</text>
				</g>
				<rect class="handle-hit" x={eL - 22} y={eTop - 8} width="44" height={eBot - eTop + 16} role="slider" tabindex="0"
					aria-label="learning rate η" aria-valuemin={ETA_MIN} aria-valuemax={ETA_MAX} aria-valuenow={eta} aria-valuetext="η = {sci(eta)}"
					onpointerdown={eDown} onpointermove={eMove} onpointerup={eUp} onkeydown={eKey} ondblclick={() => change(() => (eta = recipe.eta))} />
			</g>

			<!-- the parameters -->
			<g transform="translate({rightX} {narrow ? leftH + 22 : 16})">
				<text class="lbl lbl--micro" x="0" y="-6">every parameter, one square each (area ∝ |θᵢ|, open: negative)</text>
				{@render params(thetaNow, 0, 6, cell)}
			</g>

			<!-- what one character reads -->
			<g transform="translate({rightX} 0)">
				<text class="lbl lbl--micro" x="8" y={arcsTop}>what ‘{showChar(wText[reader])}’ reads: head 1 above, head 2 below</text>
				{@render reads(attn, arcBase, arcXs, arcUp)}
			</g>
		</svg>

		{#if attrs.t != null}
			<p class="film-status label">AdamW · seed 0 · step {k} / 200 · loss {fmt(lossNow, 4)} nats</p>
		{/if}

		<div class="fx-live__controls label" role="group" aria-label="training controls">
			<button class="scrub__play" onclick={play} aria-label={playing ? 'Pause' : 'Train'}>
				{#if playing}
					<svg viewBox="0 0 12 12" width="10" height="10" aria-hidden="true"><rect x="2.6" y="2" width="2.2" height="8" /><rect x="7.2" y="2" width="2.2" height="8" /></svg>
				{:else}
					<svg viewBox="0 0 12 12" width="10" height="10" aria-hidden="true"><path d="M3.4 1.8 L10 6 L3.4 10.2 Z" /></svg>
				{/if}
			</button>
			<button class="fx-live__word" onclick={stepOnce}>one step</button>
			<button class="fx-live__word" onclick={reset}>reset</button>
			<span class="fx-live__sep" aria-hidden="true"></span>
			<button class="fx-live__word" class:is-on={optName === 'adamw'} aria-pressed={optName === 'adamw'} onclick={() => optName !== 'adamw' && change(() => (optName = 'adamw'))}>AdamW</button>
			<button class="fx-live__word" class:is-on={optName === 'gd'} aria-pressed={optName === 'gd'} onclick={() => optName !== 'gd' && change(() => (optName = 'gd'))}>gradient descent</button>
			<span class="fx-live__sep" aria-hidden="true"></span>
			<button class="fx-live__word" class:is-on={sched === 'cosine'} aria-pressed={sched === 'cosine'} onclick={() => sched !== 'cosine' && change(() => (sched = 'cosine'))}>warmup + cosine</button>
			<button class="fx-live__word" class:is-on={sched === 'constant'} aria-pressed={sched === 'constant'} onclick={() => sched !== 'constant' && change(() => (sched = 'constant'))}>constant η</button>
			<span class="fx-live__sep" aria-hidden="true"></span>
			<span class="fx-live__seed">
				<button class="fx-live__word" aria-label="previous seed" disabled={seed === 0} onclick={() => change(() => (seed = Math.max(0, seed - 1)))}>‹</button>
				<span>seed {seed}</span>
				<button class="fx-live__word" aria-label="next seed" onclick={() => change(() => (seed = seed + 1))}>›</button>
			</span>
		</div>

		<p class="readout" aria-live="polite">
			<span class="readout__who">Step {k}{isBook() ? ', the book’s run' : ''}.</span>
			{#if diverged}
				<span class="readout__part">The loss is no longer a number: at η = {sci(eta)} {optLabel} has diverged. Drag η down, or reset.</span>
			{:else}
				<span class="readout__part">Loss {fmt(lossNow, 4)} nats (uniform guessing {fmt(lnV, 2)}, floor {fmt(lab.floor, 4)}){k ? `; the last step had ‖g‖ = ${sci(gnorm)} and η = ${sci(rateNow)}` : ''}.</span>
				<span class="readout__part">The key bias b<sub>k</sub>, whose gradient is zero, {bkMax === 0 ? 'has not moved' : `has moved by at most ${sci(bkMax)}`}.</span>
			{/if}
			<span class="readout__hint">Press ▶ to train ({optLabel}, {sched === 'cosine' ? 'warmup and cosine over 200 steps' : 'constant η'}, seed {seed}{seed === 0 ? ': the book’s θ₀' : ''}). Drag η on its axis; changing anything starts again from θ₀.</span>
		</p>
	{:else}
		<svg viewBox="0 0 {W} {chartH}" width={W} height={chartH} role={interactive ? 'group' : 'img'} aria-label="The recorded training loss from {fmt(specimen.loss[0], 2)} to {fmt(specimen.loss.at(-1), 4)} nats in {lab.steps} steps, with samples.">
			<LossChart x={0} y={0} width={W} height={chartH} series={[{ values: printed.losses, style: 'main' }]} xMax={lab.steps} {lnV} floor={lab.floor} samples={printed.samples} margin={240} yMax={2.5} />
		</svg>
	{/if}
</div>

<style>
	.film-status { font-size: 12px; color: var(--ink-2); margin-top: 16px; font-variant-numeric: tabular-nums; }
	.fx-live :global(.m) { font-family: var(--serif); font-style: italic; font-size: 14px; fill: var(--ink); }
	.fx-live__pname { font-family: var(--serif); font-size: 12.5px; fill: var(--ink-2); }
	.fx-live__pname--small { font-size: 9px; }
	.fx-live__sym { font-style: italic; }
	.fx-live__sub { font-size: 9px; font-style: italic; }
	.fx-live__char { font-size: 20px; }
	.fx-live__char--small { font-size: 13px; }
	.fx-live__plan { fill: none; stroke: var(--ink-3); stroke-width: var(--hair); stroke-dasharray: 1.5 2.5; }
	.fx-live__past { fill: none; stroke: var(--value); stroke-width: var(--line); }
	.fx-live__etalabel { fill: var(--ink); font-size: 12px; }
	.fx-live__k { font-size: 8px; font-style: italic; }
	.fx-live__controls { display: flex; flex-wrap: wrap; align-items: center; gap: 0.35rem 0.9rem; margin-top: 0.5rem; max-width: var(--main); color: var(--ink-2); }
	.fx-live__word { font: inherit; background: none; border: 0; padding: 0.15rem 0; color: var(--ink-3); cursor: pointer; }
	.fx-live__word:hover:not(:disabled) { color: var(--value); }
	.fx-live__word.is-on { color: var(--ink); font-weight: 600; text-decoration: underline; text-underline-offset: 3px; text-decoration-thickness: 1px; }
	.fx-live__word:disabled { opacity: 0.35; cursor: default; }
	.fx-live__sep { width: 1px; height: 1em; background: var(--rule); }
	.fx-live__seed { display: inline-flex; gap: 0.45rem; align-items: center; color: var(--ink); }
	.fx-live__seed .fx-live__word { color: var(--ink-2); font-size: 1.15em; padding: 0 0.2rem; }
</style>

<script module>
	export const meta = { size: 'wide', title: 'Adam at the floor' };
</script>

<script>
	// Figure: the slingshot. Two runs of plain Adam from the book's θ₀ (β₁ = 0.9, β₂ = 0.95,
	// ε = 1e-8, no weight decay, no clipping), 4000 updates each:
	//   left    constant η = 0.02
	//   right   η rising over 20 steps to 0.02, then half a cosine down to 0.002
	// Rows, sharing their axes across the two runs:
	//   η_k (linear), the excess loss ℓ − floor (log), the length of the update ‖Δθ‖ (log).
	// Long runs are drawn as min–max envelopes per bin of steps (every spike survives);
	// the spikes after the first descent are found and labelled with their step.
	// The 8000 updates run once on the server; the page hydrates from the numbers
	// (computed.svelte.js), and client-side navigation recomputes them in slices.
	import { getContext } from 'svelte';
	import { measure } from '../primitives/measure.js';
	import { computed } from '../primitives/computed.svelte.js';
	import LossChart from '../primitives/LossChart.svelte';
	import { fmt, clamp, layoutWidth } from '../primitives/scales.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = getContext('specimen')();
	const lab = specimen.lab;
	const floor = lab.floor;
	const interactive = $derived(mode !== 'print');

	const STEPS = 4000, BIN = 8, ETA = 0.02;
	const RUNS = [
		{ key: 'constant', title: 'constant learning rate', schedule: null },
		{ key: 'cosine', title: 'warmup and cosine', schedule: { warmup: 20, total: STEPS, etaMin: ETA / 10 } }
	];
	const base = { optimizer: 'adamw', eta: ETA, beta1: 0.9, beta2: 0.95, eps: 1e-8, weightDecay: 0, clip: Infinity };

	const job = computed(`fig-slingshot:${STEPS}:${BIN}:${ETA}:${floor}`, function* () {
		const out = {};
		for (const [r, spec] of RUNS.entries()) {
			const run = lab.run({ ...base, schedule: spec.schedule });
			for (let k = 0; k < STEPS; k++) { run.step(); if (k % 100 === 0) yield (r + k / STEPS) / RUNS.length; }
			const nb = STEPS / BIN;
			const bins = { exLo: [], exHi: [], uLo: [], uHi: [], off: [], g: [], u: [] };
			for (let b = 0; b < nb; b++) {
				let lo = Infinity, hi = -Infinity, ulo = Infinity, uhi = -Infinity, pk = b * BIN;
				for (let k = b * BIN; k < (b + 1) * BIN; k++) {
					const ex = run.loss[k] - floor;
					if (ex < lo) lo = ex;
					if (ex > hi) { hi = ex; pk = k; }
					ulo = Math.min(ulo, run.updateNorm[k]); uhi = Math.max(uhi, run.updateNorm[k]);
				}
				bins.exLo.push(lo); bins.exHi.push(hi);
				bins.uLo.push(ulo); bins.uHi.push(uhi); bins.off.push(pk - b * BIN);
				bins.g.push(run.gradNorm[pk]); bins.u.push(run.updateNorm[pk]);
			}
			// spikes: after the first descent, a bin whose worst loss rises above 0.05 nats over
			// the floor, separated from the previous spike by a calm stretch
			const spikes = [];
			let calm = 0;
			for (let b = 0; b < nb; b++) {
				const k = b * BIN + bins.off[b];
				if (k > 250 && bins.exHi[b] > 0.02) {
					if (calm >= 6 || !spikes.length) spikes.push({ bin: b, ex: bins.exHi[b] });
					else if (bins.exHi[b] > spikes.at(-1).ex) Object.assign(spikes.at(-1), { bin: b, ex: bins.exHi[b] });
					calm = 0;
				} else calm++;
			}
			out[spec.key] = { ...bins, spikes, last: run.loss[STEPS - 1] - floor, minEx: Math.min(...bins.exLo.slice(Math.floor(500 / BIN))) };
		}
		return out;
	}, { digits: 3 });

	const spikeStep = (d, sp) => sp.bin * BIN + d.off[sp.bin];
	const binSteps = Array.from({ length: STEPS / BIN }, (_, b) => b * BIN + BIN / 2);
	let width = $state(868);
	let hover = $state(null); // { run, bin }
	const W = $derived(layoutWidth(width, mode));
	const narrow = $derived(W < 620);
	const colW = $derived(narrow ? W : (W - 28) / 2);
	const colX = (i) => (narrow ? 0 : i * (colW + 28));
	const rowY = (i) => (narrow ? i * 520 : 0);
	const yEta = 30, hEta = 56, yEx = yEta + hEta + 26, hEx = 236, yU = yEx + hEx + 26, hU = 150;
	const H = $derived((narrow ? 520 + yU + hU : yU + hU) + 8);

	const exRange = [-6, 0];
	const uRange = $derived.by(() => {
		if (!job.value) return [-3, 0];
		const all = RUNS.flatMap((r) => [...job.value[r.key].uLo, ...job.value[r.key].uHi]).filter((v) => v > 0);
		return [Math.floor(Math.log10(Math.min(...all))), Math.ceil(Math.log10(Math.max(...all)))];
	});
	const sup = (d) => String(d).replace(/-/, '⁻').replace(/[0-9]/g, (c) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+c]);
	const sci = (x) => {
		if (!(x > 0)) return fmt(x, 3);
		const e = Math.floor(Math.log10(x));
		return e >= -2 ? fmt(x, e >= -1 ? 3 : 4) : `${fmt(x / 10 ** e, 1)} × 10${sup(e)}`;
	};

	// geometry shared with LossChart (L = 34, R = 10, top pad 16, bottom 26)
	const cL = 34;
	const X = (k) => cL + ((colW - cL - 10) * k) / STEPS;
	const exY = (v) => yEx + 16 + (hEx - 42) * (1 - clamp((Math.log10(Math.max(v, 1e-12)) - exRange[0]) / (exRange[1] - exRange[0]), 0, 1));

	function point(e, i) {
		if (!job.value) return;
		const svg = e.currentTarget.ownerSVGElement;
		const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
		const p = pt.matrixTransform(svg.getScreenCTM().inverse());
		const k = ((p.x - colX(i) - cL) / (colW - cL - 10)) * STEPS;
		const bin = clamp(Math.floor(k / BIN), 0, STEPS / BIN - 1);
		// snap to a spike if one is near
		const sp = job.value[RUNS[i].key].spikes.find((s) => Math.abs(s.bin - bin) <= 3);
		hover = { run: i, bin: sp ? sp.bin : bin };
	}
	function key(e, i) {
		if (!job.value) return;
		const sp = job.value[RUNS[i].key].spikes;
		if (!sp.length) return;
		const cur = hover?.run === i ? sp.findIndex((s) => s.bin === hover.bin) : -1;
		if (e.key === 'ArrowRight') { hover = { run: i, bin: sp[Math.min(sp.length - 1, cur + 1)].bin }; e.preventDefault(); }
		if (e.key === 'ArrowLeft') { hover = { run: i, bin: sp[Math.max(0, cur - 1)].bin }; e.preventDefault(); }
		if (e.key === 'Escape') hover = null;
	}
	const at = $derived.by(() => {
		if (!hover || !job.value) return null;
		const d = job.value[RUNS[hover.run].key], b = hover.bin;
		const r = RUNS[hover.run], k = b * BIN + d.off[b];
		return { run: r, k, ex: d.exHi[b], g: d.g[b], u: d.u[b], lr: r.schedule ? lab.scheduleRate(k + 1, { eta: ETA, ...r.schedule }) : ETA };
	});
	const etaPath = (r) => {
		const pts = [];
		for (let k = 1; k <= STEPS; k += k < 40 ? 1 : 20) pts.push([k, r.schedule ? lab.scheduleRate(k, { eta: ETA, ...r.schedule }) : ETA]);
		return pts.map(([k, v], i) => `${i ? 'L' : 'M'}${X(k).toFixed(1)} ${(yEta + hEta - 6 - (hEta - 16) * (v / ETA)).toFixed(1)}`).join('');
	};
</script>

<div class="fx fx-sling" use:measure={(w) => (width = w)}>
	{@html job.tag}
	<svg viewBox="0 0 {W} {H}" width={W} height={H} role={interactive ? 'group' : 'img'}
		aria-label={job.value
			? `Adam at a constant learning rate: after reaching the floor the loss jumps back up ${job.value.constant.spikes.length} times, at steps ${job.value.constant.spikes.map((sp) => spikeStep(job.value.constant, sp)).join(', ')}. With the cosine schedule: ${job.value.cosine.spikes.length} such ${job.value.cosine.spikes.length === 1 ? 'jump' : 'jumps'}${job.value.cosine.spikes.length ? `, at step${job.value.cosine.spikes.length === 1 ? '' : 's'} ${job.value.cosine.spikes.map((sp) => spikeStep(job.value.cosine, sp)).join(', ')}` : ''}.`
			: 'Adam at the floor, computing'}>
		{#each RUNS as r, i (r.key)}
			{@const d = job.value?.[r.key]}
			<g transform="translate({colX(i)} {rowY(i)})">
				<text class="lbl lbl--head" x={cL} y={12}>{r.title}</text>
				<!-- η_k -->
				<line class="axis" x1={cL} x2={cL} y1={yEta + 10} y2={yEta + hEta - 6} />
				<text class="lbl lbl--micro" x={cL - 6} y={yEta + 13.5} text-anchor="end">{fmt(ETA, 2)}</text>
				<text class="lbl lbl--micro" x={cL - 6} y={yEta + hEta - 2.5} text-anchor="end">0</text>
				<line class="axis" x1={cL} x2={colW - 10} y1={yEta + hEta - 6} y2={yEta + hEta - 6} />
				<path class="fx-sling__eta" d={etaPath(r)} />
				<text class="lbl lbl--micro" x={colW - 10} y={yEta + 6} text-anchor="end"><tspan class="m">η</tspan><tspan class="m sub" dy="2">k</tspan></text>

				{#if d}
					<LossChart x={0} y={yEx} width={colW} height={hEx} xScale="linear" xMax={STEPS} yScale="excess" yLo={exRange[0]} yHi={exRange[1]}
						series={[{ values: [], band: { steps: binSteps, lo: d.exLo, hi: d.exHi } }]} yLabel="" xLabel="" />
					<text class="lbl fx-sling__row" x={cL} y={yEx + 2}>loss above the floor, <tspan class="m">ℓ</tspan> − {fmt(floor, 4)}, in nats</text>
					{#each d.spikes as sp (sp.bin)}
						<text class="lbl lbl--micro fx-sling__spike" x={X(spikeStep(d, sp))} y={exY(sp.ex) - 6} text-anchor="middle">{spikeStep(d, sp)}</text>
					{/each}
					<LossChart x={0} y={yU} width={colW} height={hU} xScale="linear" xMax={STEPS} yScale="excess" yLo={uRange[0]} yHi={uRange[1]}
						series={[{ values: [], band: { steps: binSteps, lo: d.uLo, hi: d.uHi } }]} yLabel="" xLabel="update" />
					<text class="lbl fx-sling__row" x={cL} y={yU + 2}>length of the update, ‖Δ<tspan class="m">θ</tspan>‖</text>
					{#if at && at.run.key === r.key}
						<line class="fx-sling__cursor" x1={X(at.k)} x2={X(at.k)} y1={yEta + 10} y2={yU + hU - 26} />
					{/if}
				{:else}
					<text class="lbl lbl--micro" x={cL + 8} y={yEx + 60}>running {STEPS} updates of Adam… {Math.round((job.progress ?? 0) * 100)} %</text>
				{/if}
				{#if interactive}
					<rect class="hit fx-sling__hit" x={cL} y={yEta} width={colW - cL - 10} height={yU + hU - 26 - yEta} role="slider" tabindex="0"
						aria-label="spikes of the run with {r.title}; ← and → move between them" aria-valuemin="0" aria-valuemax={STEPS} aria-valuenow={at?.run.key === r.key ? at.k : 0}
						onpointermove={(e) => point(e, i)} onpointerdown={(e) => point(e, i)} onpointerleave={() => (hover = null)} onkeydown={(e) => key(e, i)} />
				{/if}
			</g>
		{/each}
	</svg>
	{#if interactive}
		<p class="readout" aria-live="polite">
			{#if at}
				<span class="readout__who">{at.run.title[0].toUpperCase() + at.run.title.slice(1)}, update {at.k}.</span>
				<span class="readout__part">The loss is {sci(at.ex)} nats above the floor; the gradient has length {sci(at.g)} and the update {sci(at.u)}, at η = {sci(at.lr)}.</span>
			{:else if job.value}
				<span class="readout__part">After the first descent, the run at constant η jumps away from the floor {job.value.constant.spikes.length} times; the run with the cosine schedule {job.value.cosine.spikes.length === 0 ? 'never does' : job.value.cosine.spikes.length === 1 ? 'once' : `${job.value.cosine.spikes.length} times`}{job.value.cosine.spikes.length ? `, at step ${job.value.cosine.spikes.map((sp) => spikeStep(job.value.cosine, sp)).join(', ')}, while η is still large` : ''}.</span>
				<span class="readout__hint">Point at a spike, or focus a run and use ← and →.</span>
			{:else}
				<span class="readout__part">Running {STEPS} updates of Adam, twice…</span>
			{/if}
		</p>
	{/if}
</div>

<style>
	.fx-sling :global(.m) { font-family: var(--serif); font-style: italic; font-size: 13px; fill: var(--ink); }
	.fx-sling :global(.sub) { font-size: 9px; }
	.fx-sling__eta { fill: none; stroke: var(--value); stroke-width: var(--line); }
	.fx-sling__spike { fill: var(--ink); font-weight: 600; paint-order: stroke; stroke: var(--paper); stroke-width: 3px; }
	.fx-sling__row { fill: var(--ink-2); font-size: 11.5px; }
	.fx-sling__cursor { stroke: var(--ink); stroke-width: var(--hair); }
	.fx-sling__hit { cursor: crosshair; }
</style>

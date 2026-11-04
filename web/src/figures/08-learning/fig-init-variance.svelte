<script module>
	export const meta = { size: 'wide', title: 'Why these initial variances' };
	export { default as Key } from './fig-init-variance.key.svelte';
</script>

<script>
	// Figure: the size of every intermediate of the forward pass at initialisation, for
	// three choices of the initial weights: the book's variances, and every weight
	// matrix except E with its variance divided by 9 or multiplied by 9 (standard
	// deviation ÷3, ×3). Columns = the three schemes; rows = the stations of the forward
	// pass, top to bottom (forward flows down). One dot per seed (seed 0 is the book's
	// own θ₀); the line joins the medians. Below: how evenly attention reads (normalised
	// entropy, 1 = uniform) and the initial loss against ln V.
	import { getContext } from 'svelte';
	import { measure } from '../primitives/measure.js';
	import { computed } from '../primitives/computed.svelte.js';
	import { fmt, clamp, layoutWidth } from '../primitives/scales.js';
	import { INIT_KEY, initJob, SCHEMES, SEEDS, ROWS } from './initvar.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const interactive = $derived(mode !== 'print');

	// svelte-ignore state_referenced_locally
	const job = computed(INIT_KEY, initJob(specimen));

	const data = $derived(job.value);
	const lnV = $derived(Math.log(specimen.config.V));
	const median = (a) => { const s = a.slice().sort((x, y) => x - y); const n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };

	// geometry
	let width = $state(868);
	const W = $derived(layoutWidth(width, mode, 340));
	const narrow = $derived(W < 640);
	const labelW = $derived(narrow ? 46 : 150);
	const gap = $derived(narrow ? 10 : 22);
	const colW = $derived((W - labelW - 2 * gap) / 3);
	const colX = (c) => labelW + c * (colW + gap);
	const LO = -2.5, HI = 1.5; // log10 RMS
	const xr = (c, v) => colX(c) + ((clamp(Math.log10(v), LO, HI) - LO) / (HI - LO)) * colW;
	const top = 50, rowH = 30;
	const yr = (k) => top + k * rowH + rowH / 2;
	const yH = $derived(top + ROWS.length * rowH + 34);
	const yL = $derived(yH + 50);
	const H = $derived(yL + 40);
	// loss axis: shared across columns, from the data
	const lossRange = $derived.by(() => {
		if (!data) return [1.5, 3];
		const all = data.flat().map((v) => v.loss);
		return [Math.min(lnV, ...all) - 0.1, Math.max(lnV, ...all) + 0.1];
	});
	const xl = (c, v) => colX(c) + ((v - lossRange[0]) / (lossRange[1] - lossRange[0])) * colW;
	const xh = (c, v) => colX(c) + clamp(v, 0, 1) * colW;

	let hover = $state(-1); // row index, ROWS.length = entropy, +1 = loss
	const rowWords = (k) => (k < ROWS.length ? ROWS[k][3] : k === ROWS.length ? 'evenness of attention' : 'initial loss');
	const rowVal = (v, k) => (k < ROWS.length ? v[ROWS[k][0]] : k === ROWS.length ? v.H : v.loss);
	const readout = $derived.by(() => {
		if (!data) return null;
		const k = hover >= 0 ? hover : ROWS.length + 1;
		return { k, words: rowWords(k), parts: SCHEMES.map((s, c) => `${s.name}: ${fmt(median(data[c].map((v) => rowVal(v, k))), k === ROWS.length ? 2 : k > ROWS.length ? 3 : 2)}`) };
	});
	const dec = [-2, -1, 0, 1];
	const decLabel = (e) => (e === -2 ? '0.01' : e === -1 ? '0.1' : e === 0 ? '1' : '10');
</script>

<div class="fx fx-init" use:measure={(w) => (width = w)}>
	{@html job.tag}
	{#if !data}
		<svg viewBox="0 0 {W} 60" width={W} height="60" role={interactive ? 'group' : 'img'} aria-label="computing initial activations">
			<line class="rule" x1="0" x2={W} y1="30" y2="30" />
			<line class="fx-init__progress" x1="0" x2={W * job.progress} y1="30" y2="30" />
		</svg>
	{:else}
		<svg viewBox="0 0 {W} {H}" width={W} height={H} role={interactive ? 'group' : 'img'}
			aria-label="Root-mean-square sizes at initialisation. Initial loss, median over {SEEDS} seeds: {readout.parts.join('; ')}; ln V = {fmt(lnV, 3)}">
			{#each SCHEMES as s, c (c)}
				<text class="lbl lbl--head" x={colX(c)} y="14">{narrow ? s.short : s.name}</text>
				<!-- log axis of sizes -->
				{#each dec as e (e)}
					<line class={e === 0 ? 'fx-init__one' : 'rule'} x1={xr(c, 10 ** e)} x2={xr(c, 10 ** e)} y1={top - 4} y2={top + ROWS.length * rowH} />
					{#if !narrow || e !== -2}<text class="lbl lbl--micro" x={xr(c, 10 ** e)} y={top - 9} text-anchor="middle">{decLabel(e)}</text>{/if}
				{/each}
				<!-- the median profile, top to bottom -->
				<polyline class="fx-init__profile" points={ROWS.map(([key], k) => `${xr(c, median(data[c].map((v) => v[key])))},${yr(k)}`).join(' ')} />
				{#each ROWS as [key], k (key)}
					{#each data[c] as v, si (si)}
						<circle class="fx-init__dot" class:is-book={si === 0 && c === 1} cx={xr(c, v[key])} cy={yr(k) + (si - (SEEDS - 1) / 2) * 1.1} r="2.3" />
					{/each}
					{#if !narrow && (v => v > 10 ** HI || v < 10 ** LO)(median(data[c].map((v) => v[key])))}
						<text class="lbl lbl--micro" x={colX(c) + colW} y={yr(k) - 6} text-anchor="end">off scale</text>
					{/if}
				{/each}
				<!-- evenness of attention: normalised entropy, 0 = one source, 1 = uniform -->
				<line class="axis" x1={colX(c)} x2={colX(c) + colW} y1={yH} y2={yH} />
				{#each [0, 0.5, 1] as u (u)}
					<line class="axis" x1={xh(c, u)} x2={xh(c, u)} y1={yH} y2={yH + 4} />
					<text class="lbl lbl--micro" x={xh(c, u)} y={yH + 15} text-anchor={u === 0 ? 'start' : u === 1 ? 'end' : 'middle'}>{narrow ? fmt(u, u === 0.5 ? 1 : 0) : u === 0 ? 'one source' : u === 1 ? 'uniform' : '0.5'}</text>
				{/each}
				{#each data[c] as v, si (si)}
					<circle class="fx-init__dot" class:is-book={si === 0 && c === 1} cx={xh(c, v.H)} cy={yH - 5 - si * 0.9} r="2.3" />
				{/each}
				<!-- initial loss, against uniform guessing -->
				<line class="axis" x1={colX(c)} x2={colX(c) + colW} y1={yL} y2={yL} />
				<line class="fx-init__lnv" x1={xl(c, lnV)} x2={xl(c, lnV)} y1={yL - 14} y2={yL + 4} />
				{#if c === 0}<text class="lbl lbl--micro" x={xl(c, lnV) + 3} y={yL - 16}>ln <tspan class="m">V</tspan>{narrow ? '' : ', uniform guessing'}</text>{/if}
				{#each [Math.ceil(lossRange[0] * 2) / 2, Math.floor(lossRange[1] * 2) / 2] as u (u)}
					<line class="axis" x1={xl(c, u)} x2={xl(c, u)} y1={yL} y2={yL + 4} />
					<text class="lbl lbl--micro" x={xl(c, u)} y={yL + 15} text-anchor="middle">{fmt(u, 1)}</text>
				{/each}
				{#each data[c] as v, si (si)}
					<circle class="fx-init__dot" class:is-book={si === 0 && c === 1} cx={xl(c, v.loss)} cy={yL - 5 - si * 0.9} r="2.3" />
				{/each}
			{/each}
			<!-- row labels -->
			{#each ROWS as [key, sym, sub, words], k (key)}
				<g class="fx-init__row" class:is-on={hover === k}>
					<text class="fx-init__sym" x={narrow ? labelW - 6 : 34} y={yr(k) + 5} text-anchor="end"><tspan class="m">{sym}</tspan>{#if sub}<tspan class="m sub" dy="3">{sub}</tspan>{/if}{#if key === 'attn'}<tspan class="m" dy="-3">o</tspan>{/if}</text>
					{#if !narrow}<text class="lbl" x="42" y={yr(k) + 4}>{words}</text>{/if}
				</g>
			{/each}
			{#if narrow}
				<text class="lbl lbl--micro" x={labelW - 6} y={yH + 3} text-anchor="end">attn</text>
				<text class="lbl lbl--micro" x={labelW - 6} y={yL + 3} text-anchor="end">loss</text>
			{:else}
				<text class="lbl" x="42" y={yH - 1}>attention</text>
				<text class="lbl lbl--micro" x="42" y={yH + 12}>evenness of reading</text>
				<text class="lbl" x="42" y={yL - 1}>initial loss</text>
				<text class="lbl lbl--micro" x="42" y={yL + 12}>nats</text>
				<text class="lbl lbl--micro" x="0" y={top - 9}>size (rms)</text>
			{/if}
			{#if interactive}
				{#each [...ROWS.map((_, k) => [k, yr(k) - rowH / 2, rowH]), [ROWS.length, yH - 20, 36], [ROWS.length + 1, yL - 20, 36]] as [k, y, h] (k)}
					<rect class="hit" x="0" {y} width={W} height={h} role="button" tabindex="0" aria-label={rowWords(k)}
						onpointerenter={() => (hover = k)} onfocus={() => (hover = k)} onpointerleave={() => (hover = -1)} onblur={() => (hover = -1)} onclick={() => (hover = k)} onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); (hover = k); } }} />
				{/each}
			{/if}
		</svg>
		{#if interactive}
			<p class="readout" aria-live="polite">
				<span class="readout__who">{readout.words[0].toUpperCase() + readout.words.slice(1)}, median of {SEEDS} seeds:</span>
				<span class="readout__part">{readout.parts.join('; ')}{readout.k === ROWS.length + 1 ? `; ln V = ${fmt(lnV, 3)}` : ''}.</span>
				<span class="readout__hint">Point at a row. The heavier dot in the middle column is the book’s own θ₀.</span>
			</p>
		{/if}
	{/if}
</div>

<style>
	.fx-init :global(.m) { font-family: var(--serif); font-style: italic; }
	.fx-init :global(.sub) { font-size: 0.72em; }
	.fx-init__sym { font-size: 15px; fill: var(--ink); }
	.fx-init__dot { fill: var(--value); mix-blend-mode: var(--overprint); opacity: 0.8; }
	.fx-init__dot.is-book { fill: var(--paper); stroke: var(--ink); stroke-width: 1.2px; opacity: 1; mix-blend-mode: normal; }
	.fx-init__profile { fill: none; stroke: var(--value); stroke-width: var(--line); opacity: 0.6; }
	.fx-init__one { stroke: var(--ink-3); stroke-width: 0.8px; }
	.fx-init__lnv { stroke: var(--ink); stroke-width: 1px; }
	.fx-init__row.is-on .fx-init__sym, .fx-init__row.is-on .lbl { fill: var(--ink); font-weight: 600; }
	.fx-init__progress { stroke: var(--value); stroke-width: 1.5px; }
</style>

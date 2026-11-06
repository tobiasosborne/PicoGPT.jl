<script module>
	export const meta = { size: 'wide', title: 'Is there a band of depths?' };
</script>

<script>
	// Figure: the paper's per-depth signatures of a workspace band, on the Shakespeare model.
	// One row per statistic, one column per depth (shared axis at the bottom). In each row the
	// J-lens is a short bar in covector ink, its control an open ring in ink:
	//   (a) top-1 agreement with the model's own next character — control: the logit lens;
	//   (b) mean excess kurtosis of the readout's logits — control: the logit lens;
	//   (c) the number of singular directions of Wout J_ℓ holding 90 % of Σσ² (no control:
	//       the readout's own count is the value at the last depth);
	//   (d) occupancy, the first k at which a J-lens vector improves the fit less than a random
	//       one (chapter 4);
	//   (e) the share of ‖x‖² in the span of the 65 lens vectors — control: random 65-dim spans.
	// A workspace band would show as a middle stretch of depths that differs from both ends.
	// Numbers: Shakespeare export lenses.metrics, jacobians.readout_singular_values, j_space.
	import { getContext } from 'svelte';
	import { measure } from '#figures/primitives/measure.js';
	import DepthAxis from '../primitives/DepthAxis.svelte';
	import { fmt, layoutWidth } from '#figures/primitives/scales.js';
	import { at, depthLabel, depthScale } from '../primitives/jdata.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const S = $derived(getContext('artifacts')()?.jspace?.shakespeare ?? null);
	const interactive = $derived(mode !== 'print');
	let width = $state(868);
	let focus = $state(-1);

	const n = $derived(S?.depth.length ?? 0);
	const rank = $derived(S ? S.jacobians.readout_singular_values.shape[0] : 0);
	/** singular directions of Wout J_j needed for 90 % of the sum of squares */
	function effDim(j) {
		const sv = S.jacobians.readout_singular_values;
		const s2 = Array.from({ length: sv.shape[0] }, (_, r) => at(sv, r, j) ** 2);
		const total = s2.reduce((a, b) => a + b, 0);
		let acc = 0, k = 0;
		while (k < s2.length && acc < 0.9 * total) acc += s2[k++];
		return k;
	}
	const rows = $derived.by(() => {
		if (!S) return [];
		const m = S.lenses.metrics, J = S.j_space;
		const kurt = m.flatMap((q) => [q.j_lens.mean_excess_kurtosis, q.logit_lens.mean_excess_kurtosis]);
		const kmax = Math.max(0.2, ...kurt.map(Math.abs));
		return [
			{ key: 'agree', title: 'agreement with the model', note: 'top-1, J-lens vs model', ring: 'logit lens',
				lo: 0, hi: 1, ticks: [0, 0.5, 1], j: m.map((q) => q.j_lens.model_agreement), c: m.map((q) => q.logit_lens.model_agreement), f: (v) => fmt(v, 2) },
			{ key: 'kurt', title: 'excess kurtosis', note: 'of the readout’s logits', ring: 'logit lens',
				lo: -kmax, hi: kmax, ticks: [0], j: m.map((q) => q.j_lens.mean_excess_kurtosis), c: m.map((q) => q.logit_lens.mean_excess_kurtosis), f: (v) => fmt(v, 2) },
			{ key: 'dim', title: 'effective dimension', note: `directions for 90 % of Wout J (of ${rank})`, ring: null,
				lo: 0, hi: rank, ticks: [0, rank], j: S.depth.map((_, j) => effDim(j)), c: null, f: (v) => fmt(v, 0) },
			{ key: 'occ', title: 'occupancy', note: 'k where J stops beating random', ring: null,
				lo: 0, hi: rank, ticks: [0, rank], j: J.map((q) => q.occupancy ?? NaN), c: null, f: (v) => (Number.isFinite(v) ? fmt(v, 0) : 'none') },
			{ key: 'span', title: 'share of ‖x‖² in the span', note: `of the ${rank} lens vectors`, ring: `random ${rank}-dim span`,
				lo: 0, hi: 1, ticks: [0, 0.5, 1], j: J.map((q) => q.span_fraction), c: J.map((q) => q.random_span_fraction), f: (v) => fmt(v, 2) }
		];
	});

	// geometry
	const W = $derived(layoutWidth(width, mode, 340));
	const narrow = $derived(W < 620);
	const padL = $derived(narrow ? 30 : 196);
	const padR = $derived(narrow ? 32 : 30);
	const xs = $derived(depthScale(n, padL + 12, W - padR - 12));
	const hRow = $derived(narrow ? 54 : 58);
	const gapRow = $derived(narrow ? 58 : 22);
	const top = 14;
	const y0 = (r) => top + (narrow ? 32 : 0) + r * (hRow + gapRow); // top of row r
	const yv = (r, v) => { const R = rows[r]; return y0(r) + hRow - ((v - R.lo) / (R.hi - R.lo)) * hRow; };
	const yAxis = $derived(y0(rows.length - 1) + hRow + 18);
	const H = $derived(yAxis + 64);

	const readout = $derived.by(() => {
		if (focus < 0 || !rows.length) return null;
		return `Depth ${depthLabel(S.depth[focus])}: ` + rows.map((R) => `${R.title} ${R.f(R.j[focus])}${R.c ? ` (${R.ring} ${R.f(R.c[focus])})` : ''}`).join('; ') + '.';
	});
</script>

<div class="fx fx-band" use:measure={(w) => (width = w)}>
	{#if !S}
		<p class="fig__pending">The Shakespeare J-lens export is missing.</p>
	{:else}
		<svg viewBox="0 0 {W} {H}" width={W} height={H} role={interactive ? 'group' : 'img'} aria-label="Per-depth statistics of the J-lens on the Shakespeare model. {rows.map((R) => `${R.title}: ${R.j.map(R.f).join(', ')}`).join('. ')}">
			{#if focus >= 0}
				<rect class="band__wash" x={xs(focus) - (xs(1) - xs(0)) / 2} y={top - 6} width={xs(1) - xs(0)} height={yAxis - top + 2} />
			{/if}
			{#each rows as R, r (R.key)}
				{@const yt = y0(r)}
				<!-- row label -->
				{#if narrow}
					<text class="lbl lbl--head" x={4} y={yt - 24}>{R.title}</text>
					<text class="lbl lbl--micro" x={4} y={yt - 11}>{R.note}{R.ring ? `; rings: ${R.ring}` : ''}</text>
				{:else}
					<text class="lbl lbl--head" x={padL - 34} y={yt + 14} text-anchor="end">{R.title}</text>
					<text class="lbl lbl--micro" x={padL - 34} y={yt + 28} text-anchor="end">{R.note}</text>
					{#if R.ring}
						<text class="lbl lbl--micro" x={padL - 34} y={yt + 42} text-anchor="end">rings: {R.ring}</text>
					{/if}
				{/if}
				<!-- the row's scale -->
				{#each R.ticks as t (t)}
					<line class={t === 0 && R.lo < 0 ? 'axis' : 'rule'} x1={padL} x2={W - padR} y1={yv(r, t)} y2={yv(r, t)} />
					<text class="lbl lbl--micro" x={padL - 6} y={yv(r, t) + 3} text-anchor="end">{R.f(t)}</text>
				{/each}
				{#if R.lo < 0}<line class="rule" x1={padL} x2={padL} y1={yt} y2={yt + hRow} />{/if}
				<!-- the control, then the J-lens -->
				{#if R.c}
					{#each R.c as v, j (j)}
						<circle class="band__ring" cx={xs(j)} cy={yv(r, v)} r="3.2" />
					{/each}
				{/if}
				{#each R.j as v, j (j)}
					{#if Number.isFinite(v)}
						<line class="band__stalk" x1={xs(j)} x2={xs(j)} y1={yv(r, Math.max(R.lo, Math.min(R.hi, 0)))} y2={yv(r, v)} />
						<rect class="band__bar" x={xs(j) - 7} y={yv(r, v) - 1.6} width="14" height="3.2" />
						<text class="lbl lbl--micro lbl--covector band__val" x={xs(j) + 9} y={yv(r, v) + (R.c && R.c[j] > v ? 11 : -4)}>{R.f(v)}</text>
					{:else}
						<text class="lbl lbl--micro" x={xs(j)} y={yt + hRow - 4} text-anchor="middle">none</text>
					{/if}
				{/each}
			{/each}
			<DepthAxis depths={S.depth} kinds={S.depth_kind} x={xs(0)} width={xs(n - 1) - xs(0)} y={yAxis} sel={focus} onpick={interactive ? (j) => (focus = focus === j ? -1 : j) : null} />
			{#if interactive}
				{#each S.depth as _, j (j)}
					<rect class="hit" x={xs(j) - (xs(1) - xs(0)) / 2} y={top} width={xs(1) - xs(0)} height={yAxis - top} role="button" tabindex="-1" aria-label="depth {depthLabel(S.depth[j])}"
						onpointerenter={() => (focus = j)} onpointerleave={() => (focus = -1)} onkeydown={() => {}} />
				{/each}
			{/if}
		</svg>
		{#if interactive}
			<p class="readout" aria-live="polite">
				{#if readout}<span class="readout__part">{readout}</span>{:else}<span class="readout__hint">Point at a depth for its numbers.</span>{/if}
			</p>
		{/if}
	{/if}
</div>

<style>
	.band__bar { fill: var(--covector); }
	.band__stalk { stroke: var(--covector); stroke-width: var(--hair); opacity: 0.6; }
	.band__ring { fill: var(--paper); stroke: var(--ink-2); stroke-width: 1px; }
	.band__wash { fill: var(--value-tint); opacity: 0.5; }
	.band__val { font-size: 10px; }
</style>

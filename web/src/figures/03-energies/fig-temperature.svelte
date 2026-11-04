<script module>
	export const meta = { size: 'wide', title: 'One distribution at seven temperatures' };
</script>

<script>
	// Figure: what temperature does to a distribution. For each context (attrs.contexts =
	// "window:position,…", 1-based) one row: the path of softmax(z/τ) through the simplex
	// (projected exactly as the frontispiece projects it), and the energy ladder at
	// τ → 0, ¼, ½, 1, 2, 4, τ → ∞, all on one energy scale. Entropy under each ladder.
	// Static; pointing at a ladder marks its point on the path.
	import { getContext } from 'svelte';
	import { measure } from '../primitives/measure.js';
	import EnergyLadder, { gibbs, entropy } from '../primitives/EnergyLadder.svelte';
	import { fmt, showChar, clamp, layoutWidth } from '../primitives/scales.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const T = $derived(specimen.config.T);
	const V = $derived(specimen.config.V);
	const interactive = $derived(mode !== 'print');
	let width = $state(868);
	let hover = $state(-1);

	const TAUS = [
		{ t: 0, label: '→ 0' },
		{ t: 0.25, label: '¼' },
		{ t: 0.5, label: '½' },
		{ t: 1, label: '1' },
		{ t: 2, label: '2' },
		{ t: 4, label: '4' },
		{ t: Infinity, label: '→ ∞' }
	];
	const contexts = $derived(
		String(attrs.contexts ?? '1:1,3:1')
			.split(',')
			.map((s) => s.split(':').map((v) => +v - 1))
			.map(([w, p]) => {
				const start = clamp(w, 0, specimen.corpus.length - T), pos = clamp(p, 0, T - 1);
				const tr = specimen.windowTrace(start);
				return { start, pos, z: tr.Z[pos], ctx: tr.chars.slice(0, pos + 1).map(showChar).join('') };
			})
	);
	// one energy scale for every ladder
	const allE = $derived(contexts.flatMap((c) => c.z.map((v) => -v)));
	const Emin = $derived(Math.floor(Math.min(...allE)) - 1);
	const Emax = $derived(Math.ceil(Math.max(...allE)) + 1);

	// geometry
	const W = $derived(layoutWidth(width, mode, 340));
	const narrow = $derived(interactive && W < 760);
	const simW = $derived(narrow ? 150 : 128);
	const perRow = $derived(narrow ? 3 : TAUS.length);
	const colW = $derived(narrow ? (W - 8) / perRow : (W - simW - 10) / TAUS.length);
	const ladH = $derived(narrow ? 160 : 176);
	const rowH = $derived(narrow ? 34 + simW + Math.ceil(TAUS.length / 3) * (ladH + 64) : ladH + 92);
	const H = $derived(contexts.length * rowH + 8);
	const pScale = $derived(interactive ? Math.max(22, Math.min(30, colW - 78)) : 14);
	function cell(k) {
		// position of the k-th ladder within a row
		if (!narrow) return { x: simW + 10 + k * colW + 22, y: 46 };
		const r = Math.floor(k / perRow), c = k % perRow;
		return { x: 4 + c * colW + 22, y: 34 + simW + 30 + r * (ladH + 64) };
	}

	// the simplex: vertex a at the a-th corner of a regular V-gon, clockwise from the top
	const corner = (a, R, cx, cy) => [cx + R * Math.sin((2 * Math.PI * a) / V), cy - R * Math.cos((2 * Math.PI * a) / V)];
	const project = (p, R, cx, cy) => p.reduce(([x, y], pa, a) => { const [u, v] = corner(a, R, cx, cy); return [x + pa * u, y + pa * v]; }, [0, 0]);
	const pathTaus = Array.from({ length: 220 }, (_, k) => 10 ** (2 - (5.5 * k) / 219)); // 100 … 3·10⁻⁴
	function simplexPath(z, R, cx, cy) {
		const pts = [[cx, cy], ...pathTaus.map((t) => project(gibbs(z, t), R, cx, cy)), project(gibbs(z, 0), R, cx, cy)];
		return 'M' + pts.map(([x, y]) => `${x.toFixed(2)} ${y.toFixed(2)}`).join('L');
	}
</script>

<div class="fx fx-temp" use:measure={(w) => (width = w)}>
	<svg viewBox="0 0 {W} {H}" width={W} height={H} role="img" aria-label="The Gibbs distribution of {contexts.length} contexts at temperatures from 0 to infinity: entropies {contexts.map((c) => TAUS.map((t) => fmt(entropy(gibbs(c.z, t.t)), 2)).join(', ')).join('; ')}">
		{#each contexts as c, r (r)}
			{@const y0 = r * rowH}
			{@const R = simW / 2 - 18}
			{@const cx = simW / 2 + (narrow ? 4 : 0)}
			{@const cy = y0 + 46 + simW / 2 - 4}
			<g transform="translate(0 0)">
				<text class="lbl lbl--head" x={narrow ? 4 : 0} y={y0 + 16}>after ‘<tspan class="fx-temp__ctx">{c.ctx}</tspan>’</text>
				<text class="lbl lbl--micro" x={narrow ? 4 : 0} y={y0 + 30}>window {c.start + 1}, position {c.pos + 1}</text>

				<!-- the simplex and the path of p(τ) -->
				<g class="fx-temp__simplex">
					<path class="fx-temp__poly" d={'M' + Array.from({ length: V + 1 }, (_, a) => corner(a % V, R, cx, cy).map((v) => v.toFixed(2)).join(' ')).join('L')} />
					{#each specimen.vocab as ch, a (a)}
						{@const [vx, vy] = corner(a, R + 10, cx, cy)}
						<text class="tok fx-temp__vtx" x={vx} y={vy + 4.5} text-anchor="middle">{showChar(ch)}</text>
					{/each}
					<path class="fx-temp__path" d={simplexPath(c.z, R, cx, cy)} />
					{#each TAUS as t, k (k)}
						{@const [px, py] = project(gibbs(c.z, t.t), R, cx, cy)}
						<circle class="fx-temp__dot" class:is-on={k === hover} cx={px} cy={py} r={k === hover ? 4 : t.t === 1 ? 3.2 : 2.3} />
						{#if t.t === 1 || k === hover}
							<text class="lbl lbl--micro fx-temp__dotl" x={px + 6} y={py - 5}>τ {t.t === Infinity ? '→ ∞' : t.t === 0 ? '→ 0' : '= ' + t.label}</text>
						{/if}
					{/each}
				</g>

				<!-- the ladders -->
				{#each TAUS as t, k (k)}
					{@const g = cell(k)}
					{@const p = gibbs(c.z, t.t)}
					<g class="fx-temp__col" class:is-on={k === hover} onpointerenter={() => interactive && (hover = k)} onpointerleave={() => interactive && (hover = -1)} role="presentation">
						<text class="lbl fx-temp__tau" x={g.x - 4} y={y0 + g.y - 14}><tspan class="m">τ</tspan> {t.t === 0 || t.t === Infinity ? t.label : '= ' + t.label}</text>
						<EnergyLadder z={c.z} labels={specimen.vocab} tau={t.t} {p} x={g.x} y={y0 + g.y} height={ladH} {Emin} {Emax} {pScale} lead={interactive ? 15 : 10} gap={14} barH={6} small pLabels="big" track={false} ticks={5} />
						<text class="lbl lbl--micro" x={g.x - 4} y={y0 + g.y + ladH + 18}><tspan class="m">H</tspan> = {fmt(entropy(p), 2)}</text>
						{#if interactive}<rect class="hit" x={g.x - 20} y={y0 + g.y - 30} width={colW} height={ladH + 56} />{/if}
					</g>
				{/each}
			</g>
		{/each}
	</svg>
</div>

<style>
	.fx-temp :global(.m) { font-family: var(--serif); font-style: italic; fill: var(--ink); font-size: 14px; }
	.fx-temp__ctx { font-family: var(--serif); font-size: 15px; }
	.fx-temp__poly { fill: none; stroke: var(--rule); stroke-width: var(--hair); }
	.fx-temp__vtx { font-size: 13px; fill: var(--ink-2); }
	.fx-temp__path { fill: none; stroke: var(--value); stroke-width: var(--line); }
	.fx-temp__dot { fill: var(--value); stroke: var(--paper); stroke-width: 1px; }
	.fx-temp__dotl { fill: var(--ink); }
	.fx-temp__tau { fill: var(--ink); }
	.fx-temp__col.is-on .fx-temp__tau { font-weight: 650; }
</style>

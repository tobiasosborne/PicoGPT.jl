<script module>
	/** The Gibbs distribution of logits z at temperature τ: p_a ∝ exp(z_a/τ), computed
	 *  stably. τ = 0 is the frozen limit (uniform on the tied maxima), τ = ∞ uniform. */
	export function gibbs(z, tau = 1) {
		const m = Math.max(...z);
		if (tau === Infinity) return z.map(() => 1 / z.length);
		if (!(tau > 0)) {
			const g = z.map((v) => (v === m ? 1 : 0));
			const n = g.reduce((a, b) => a + b, 0);
			return g.map((v) => v / n);
		}
		const w = z.map((v) => Math.exp((v - m) / tau));
		const s = w.reduce((a, b) => a + b, 0);
		return w.map((v) => v / s);
	}
	/** Entropy in nats (0 ln 0 = 0). */
	export const entropy = (p) => -p.reduce((s, q) => s + (q > 0 ? q * Math.log(q) : 0), 0);
	/** Free energy F = −τ ln Σ exp(−E_a/τ) with E = −z: ≤ the ground energy, ≥ it − τ ln V. */
	export function freeEnergy(z, tau = 1) {
		const m = Math.max(...z);
		if (!(tau > 0)) return -m;
		if (tau === Infinity) return -Infinity;
		return -m - tau * Math.log(z.reduce((s, v) => s + Math.exp((v - m) / tau), 0));
	}
	/** Place labels at desired y's with at least `gap` between neighbours, each group
	 *  centred on the mean of its members' wishes. Returns placed y per input index. */
	export function dodge(ys, gap) {
		const order = ys.map((y, i) => ({ y, i })).sort((a, b) => a.y - b.y);
		let groups = order.map((o) => ({ members: [o], top: o.y }));
		for (let pass = 0; pass < 50; pass++) {
			let merged = false;
			const out = [];
			for (const g of groups) {
				const prev = out.at(-1);
				if (prev && prev.top + prev.members.length * gap > g.top + 1e-9) {
					prev.members.push(...g.members);
					const mean = prev.members.reduce((s, m) => s + m.y, 0) / prev.members.length;
					prev.top = mean - ((prev.members.length - 1) * gap) / 2;
					merged = true;
				} else out.push({ members: [...g.members], top: g.top });
			}
			groups = out;
			if (!merged) break;
		}
		const placed = new Array(ys.length);
		for (const g of groups) g.members.forEach((m, k) => (placed[m.i] = g.top + k * gap));
		return placed;
	}
</script>

<script>
	// The energy ladder: a distribution on a finite set drawn as the occupation of energy
	// levels (DESIGN.md §5). Outcome a is a rung at its energy E_a = −z_a on a vertical axis
	// (up = higher energy = less likely); its occupation p_a is a bar along the rung. The
	// rungs sit at their true energies on the rail; their labels and bars sit on shelves,
	// moved apart just enough to be read when levels (nearly) coincide, joined by leaders.
	// The τ-ruler stands on the lowest rung: a rung kτ above it is e^k times less occupied.
	// F marks the free energy −τ ln 𝒵. A pure drawing: the figure owns the state and passes
	// handlers for dragging (onrung, onrail, onruler) if it wants them.
	import { fmt, fmtProb, showChar, clamp } from './scales.js';

	let {
		z = [],
		labels = [],
		tau = 1,
		p = null, // optional override (limits τ = 0, ∞)
		x = 0, // the rail
		y = 0, // top of the axis
		height = 300,
		Emin = -10,
		Emax = 10,
		pScale = 200, // px for p = 1
		gap = 19, // smallest distance between shelves
		lead = 34, // rung + leader length, rail → label
		barH = 8,
		showZ = false,
		showF = false,
		ruler = true,
		axis = true,
		ticks = null, // tick step in energy units (null: chosen)
		pLabels = 'all', // 'all' | 'big' (≥ .05) | 'none'
		track = true, // faint full-scale track behind each bar
		highlight = -1, // emphasise one outcome (e.g. the target)
		clip = false, // hide rungs outside [Emin, Emax] (an arrow marks those above)
		active = -1, // the rung being handled
		small = false,
		onrung = null, // (a, PointerEvent) on pointerdown
		onrungkey = null, // (a, KeyboardEvent)
		onrail = null,
		onrailkey = null,
		onruler = null,
		onrulerkey = null
	} = $props();

	const V = $derived(z.length);
	const P = $derived(p ?? gibbs(z, tau));
	const E = $derived(z.map((v) => -v));
	const E0 = $derived(Math.min(...E));
	const shown = $derived(E.map((e) => !clip || (e >= Emin && e <= Emax)));
	const above = $derived(clip ? E.filter((e) => e > Emax).length : 0);
	const yOf = (e) => y + ((Emax - e) / (Emax - Emin)) * height;
	const yTrue = $derived(E.map((e) => yOf(e)));
	const yShelf = $derived(dodge(yTrue.map((v, a) => (shown[a] ? v : -1e6 - a * gap)), Math.max(gap, small ? 18 : 23)));
	const xLabel = $derived(x + lead + 2);
	const xBar = $derived(x + lead + (small ? 14 : 18) + (showZ ? 44 : 0));
	const F = $derived(showF ? freeEnergy(z, tau) : null);

	const tickStep = $derived(ticks ?? (Emax - Emin > 24 ? 5 : Emax - Emin > 10 ? 2 : 1));
	const tickList = $derived.by(() => {
		const out = [];
		for (let e = Math.ceil(Emin / tickStep) * tickStep; e <= Emax + 1e-9; e += tickStep) out.push(e);
		return out;
	});

	// the τ-ruler: from the ground rung up; multiples of τ while they fit
	const xr = $derived(x - (small ? 12 : 20));
	const rulerTop = $derived(yOf(Math.min(Emax, E0 + tau)));
	const rulerOver = $derived(E0 + tau > Emax);
	const multiples = $derived.by(() => {
		const out = [];
		if (!(tau > 0) || !Number.isFinite(tau) || small) return out;
		for (let k = 2; k <= 4 && E0 + k * tau <= Emax; k++) if ((height / (Emax - Emin)) * tau >= 7) out.push(k);
		return out;
	});
	const fs = $derived(small ? 14 : 18);
</script>

<g class="ladder" class:ladder--small={small}>
	{#if axis}
		<line class="ladder__rail" x1={x} x2={x} y1={y} y2={y + height} />
		{#each tickList as e (e)}
			<line class="axis" x1={x - (e === 0 ? 5 : 3)} x2={x} y1={yOf(e)} y2={yOf(e)} />
		{/each}
		{#if !small && Emin < 0 && Emax > 0}<text class="lbl lbl--micro" x={x - 7} y={yOf(0) + 3.5} text-anchor="end">0</text>{/if}
		{#if above}<path class="ladder__arrow" d="M{x - 3} {y + 4} L{x} {y - 1} L{x + 3} {y + 4}" />{/if}
	{/if}

	{#if onrail}
		<rect class="handle-hit ladder__railhit" x={xr - 34} y={y - 6} width={xr + 20 - (xr - 34)} height={height + 12} role="slider" tabindex="0" aria-label="shift every energy by the same amount" aria-valuenow={0} onpointerdown={onrail} onkeydown={onrailkey} />
	{/if}

	{#if ruler && tau > 0 && Number.isFinite(tau)}
		<g class="ladder__ruler">
			{#if multiples.length}<line class="ladder__rulerfaint" x1={xr} x2={xr} y1={rulerTop} y2={yOf(E0 + multiples.at(-1) * tau)} />{/if}
			<line class="ladder__rulerline" x1={xr} x2={xr} y1={yOf(E0)} y2={rulerTop} />
			<line class="ladder__rulertick" x1={xr - 3} x2={xr + 3} y1={yOf(E0)} y2={yOf(E0)} />
			{#each multiples as k (k)}
				<line class="ladder__rulertick ladder__rulertick--faint" x1={xr - 2} x2={xr + 2} y1={yOf(E0 + k * tau)} y2={yOf(E0 + k * tau)} />
				{#if !small && (height / (Emax - Emin)) * tau >= 15}<text class="lbl lbl--micro ladder__k" x={xr - 6} y={yOf(E0 + k * tau) + 3.5} text-anchor="end">{k}τ</text>{/if}
			{/each}
			{#if rulerOver}
				<path class="ladder__arrow" d="M{xr - 3} {y + 5} L{xr} {y} L{xr + 3} {y + 5}" />
			{:else}
				<line class="ladder__rulertick" x1={xr - 4} x2={xr + 4} y1={rulerTop} y2={rulerTop} />
			{/if}
			{#if !small}<text class="ladder__tau" x={xr - (onruler ? 10 : 6)} y={onruler ? rulerTop + 5 : Math.min(yOf(E0) - 2, (yOf(E0) + rulerTop) / 2 + 5)} text-anchor="end">τ</text>{/if}
			{#if onruler}
				<circle class="handle" class:handle--active={active === 'ruler'} cx={xr} cy={rulerTop} r="5.5" />
				<rect class="handle-hit" x={xr - 12} y={rulerTop - 12} width="24" height="24" role="slider" tabindex="0" aria-label="temperature τ" aria-valuenow={tau} aria-valuetext="τ = {fmt(tau, 2)}" onpointerdown={onruler} onkeydown={onrulerkey} />
			{/if}
		</g>
	{/if}

	{#if F != null && Number.isFinite(F) && F >= Emin}
		<g class="ladder__F">
			<path d="M{x - 1} {yOf(F)} l-6 -3.5 l0 7 z" />
			{#if !small}<text class="ladder__Fl" x={x - 10} y={yOf(F) + 5} text-anchor="end">𝓕</text>{/if}
		</g>
	{/if}

	{#each E as e, a (a)}
		{#if shown[a]}
		{@const yt = yTrue[a]}
		{@const ys = yShelf[a]}
		{@const pa = P[a]}
		<g class="ladder__rung" class:is-hi={a === highlight} class:is-active={a === active}>
			<line class="ladder__step" x1={x} x2={x + lead * 0.45} y1={yt} y2={yt} />
			<line class="ladder__lead" x1={x + lead * 0.45} x2={x + lead - 2} y1={yt} y2={ys} />
			<text class="tok ladder__char" x={xLabel} y={ys + fs * 0.33} style:font-size="{fs}px">{showChar(labels[a])}</text>
			{#if showZ}<text class="lbl lbl--micro ladder__z" x={xBar - 8} y={ys + 3.5} text-anchor="end">{fmt(z[a], 2)}</text>{/if}
			{#if track}<line class="ladder__track" x1={xBar} x2={xBar + pScale} y1={ys} y2={ys} />{/if}
			<rect class="bar--value ladder__bar" x={xBar} y={ys - barH / 2} width={Math.max(0, pa * pScale)} height={barH} />
			{#if pLabels === 'all' || (pLabels === 'big' && pa >= 0.05)}
				<text class="lbl lbl--micro ladder__p" class:lbl--strong={pa >= 0.1} x={xBar + pa * pScale + 5} y={ys + 3.5}>{fmtProb(pa, 2)}</text>
			{/if}
			{#if onrung}
				<rect class="handle-hit ladder__hit" x={x + 2} y={Math.min(yt, ys) - gap / 2} width={xBar - x + Math.max(40, pa * pScale + 34)} height={Math.abs(yt - ys) + gap} role="slider" tabindex="0" aria-label="energy of ‘{showChar(labels[a])}’" aria-valuenow={e} aria-valuetext="E = {fmt(e, 2)}, p = {fmtProb(pa, 3, true)}" onpointerdown={(ev) => onrung(a, ev)} onkeydown={(ev) => onrungkey?.(a, ev)} />
			{/if}
		</g>
		{/if}
	{/each}
</g>

<style>
	.ladder__step { stroke: var(--ink); stroke-width: 1.4px; }
	.ladder__rail { stroke: var(--ink-2); stroke-width: 1px; }
	.ladder__lead { stroke: var(--ink-3); stroke-width: var(--hair); }
	.ladder__track { stroke: var(--rule-2); stroke-width: var(--hair); }
	.ladder__char { fill: var(--ink); }
	.ladder__rung.is-hi .ladder__char { font-weight: 650; }
	.ladder__rung.is-active .ladder__step { stroke: var(--value); stroke-width: var(--heavy); }
	.ladder__rung.is-active .ladder__char { fill: var(--value-deep); }
	.ladder__bar { transition: width var(--t-quick) var(--ease); }
	.ladder__rulerline { stroke: var(--ink); stroke-width: 2px; }
	.ladder__rulerfaint { stroke: var(--ink-3); stroke-width: var(--hair); }
	.ladder__rulertick { stroke: var(--ink); stroke-width: 1px; }
	.ladder__rulertick--faint { stroke: var(--ink-3); }
	.ladder__arrow { fill: none; stroke: var(--ink); stroke-width: 1px; }
	.ladder__tau { font-family: var(--serif); font-style: italic; font-size: 15px; fill: var(--ink); }
	.ladder__k { font-style: normal; }
	.ladder__F path { fill: var(--ink); }
	.ladder__Fl { font-family: var(--math); font-style: italic; font-size: 14px; fill: var(--ink); }
	.ladder__railhit { cursor: ns-resize; }
	.ladder__hit { cursor: ns-resize; }
	:global([data-mode='print']) .ladder__bar { transition: none; }
	@media (prefers-reduced-motion: reduce) { .ladder__bar { transition: none; } }
</style>

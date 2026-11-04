<script module>
	export const meta = { size: 'wide', title: 'Softmax as the occupation of energy levels' };
	export { default as Key } from './fig-energy-ladder.key.svelte';
</script>

<script>
	// Figure: the model's logits for one context as an energy ladder (E_a = −z_a) and their
	// Gibbs distribution p_a = e^{−E_a/τ}/𝒵. Direct manipulation:
	//   drag a rung       → that logit changes; every bar moves (normalisation); a rung
	//                       dragged level with another snaps to an exact tie
	//   drag the rail     → every energy shifts by the same amount; 𝒵 and F change, p does not
	//   drag the τ-handle → temperature (the ruler's length is τ, in units of energy)
	//   click the corpus  → another context (window, position)
	// Static state: the trained model's own logits at τ = 1.
	import { getContext } from 'svelte';
	import { measure } from '../primitives/measure.js';
	import EnergyLadder, { gibbs, entropy } from '../primitives/EnergyLadder.svelte';
	import CorpusStrip from '../primitives/CorpusStrip.svelte';
	import { fmt, showChar, clamp, layoutWidth } from '../primitives/scales.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const T = $derived(specimen.config.T);
	const V = $derived(specimen.config.V);
	const interactive = $derived(mode !== 'print');

	// svelte-ignore state_referenced_locally
	let start = $state(clamp((+attrs.window || 3) - 1, 0, specimen.corpus.length - specimen.config.T));
	// svelte-ignore state_referenced_locally
	let pos = $state(clamp((+attrs.position || 1) - 1, 0, specimen.config.T - 1));
	let tau = $state(1);
	// The film cools the same Gibbs distribution; readers still own the handle.
	$effect(() => { if (attrs.t != null) tau = 8 * (0.25 / 8) ** clamp(+attrs.t, 0, 1); });
	let shift = $state(0); // added to every logit
	let dz = $state(null); // per-logit changes made by dragging (null: none)
	let active = $state(-1);
	let width = $state(868);

	const tr = $derived(specimen.windowTrace(start));
	const zModel = $derived(tr.Z[pos]);
	const z = $derived(zModel.map((v, a) => v + (dz?.[a] ?? 0) + shift));
	const p = $derived(gibbs(z, tau));
	const H = $derived(entropy(p));
	const lnZ = $derived.by(() => {
		const m = Math.max(...z);
		return m / tau + Math.log(z.reduce((s, v) => s + Math.exp((v - m) / tau), 0));
	});
	const F = $derived(-tau * lnZ);
	const modified = $derived(dz != null || shift !== 0 || tau !== 1);
	// a context change resets the hand-made changes
	let lastCtx = '';
	$effect(() => {
		const key = `${start}:${pos}`;
		if (key !== lastCtx) { lastCtx = key; dz = null; shift = 0; }
	});

	// geometry
	const W = $derived(layoutWidth(width, mode, 340));
	const narrow = $derived(W < 700);
	const Emin = -14, Emax = 14;
	const top = $derived(narrow ? 112 : 108);
	const height = $derived(narrow ? 330 : 370);
	const xRail = $derived(narrow ? 58 : 110);
	const pScale = $derived(narrow ? Math.max(90, W - xRail - 34 - 62 - 60) : 250);
	const xNum = $derived(narrow ? 18 : Math.min(W - 250, xRail + 34 + 62 + pScale + 78));
	const yNum = $derived(narrow ? top + height + 40 : top + 30);
	const Hsvg = $derived(narrow ? yNum + 210 : top + height + 26);
	const kE = $derived((Emax - Emin) / height); // energy per px

	// dragging: the figure owns the state; the ladder reports pointerdown
	let svgEl = $state();
	let drag = null;
	function pointY(e) {
		const pt = svgEl.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
		return pt.matrixTransform(svgEl.getScreenCTM().inverse()).y;
	}
	function begin(kind, e, a = -1) {
		if (!interactive) return;
		e.preventDefault();
		svgEl.setPointerCapture(e.pointerId);
		drag = { kind, a, y0: pointY(e), z0: z.slice(), shift0: shift, tau0: tau };
		active = kind === 'rung' ? a : kind;
	}
	const E0 = $derived(Math.min(...z.map((v) => -v)));
	function onmove(e) {
		if (!drag) return;
		const yy = pointY(e);
		const dE = -(yy - drag.y0) * kE; // up = higher energy
		if (drag.kind === 'rung') setEnergy(drag.a, -drag.z0[drag.a] + dE);
		else if (drag.kind === 'rail') setShift(drag.shift0 - dE);
		else if (drag.kind === 'ruler') {
			const yTop = top;
			let t = yy >= yTop ? Emax - (yy - yTop) * kE - E0 : (Emax - E0) * Math.exp((yTop - yy) / 45);
			if (Math.abs(Math.log(Math.max(1e-9, t))) < 0.035) t = 1; // snap to the model's τ = 1
			tau = clamp(t, 0.02, 100);
		}
	}
	function onup() { drag = null; active = -1; }
	function setEnergy(a, E) {
		E = clamp(E, Emin + 0.2, Emax - 0.2);
		const Es = z.map((v) => -v);
		for (let b = 0; b < V; b++) if (b !== a && Math.abs(Es[b] - E) < 0.2) E = Es[b]; // snap: an exact tie
		const next = (dz ?? new Array(V).fill(0)).slice();
		next[a] = -E - zModel[a] - shift;
		dz = next;
	}
	function setShift(c) {
		const zz = zModel.map((v, a) => v + (dz?.[a] ?? 0));
		const lo = Math.max(...zz.map((v) => -v - (Emax - 0.2))); // keep every rung on the axis
		const hi = Math.min(...zz.map((v) => -v - (Emin + 0.2)));
		shift = clamp(c, -hi, -lo);
		if (Math.abs(shift) < 0.04) shift = 0;
	}
	function rungKey(a, e) {
		const d = e.shiftKey ? 1 : 0.1;
		if (e.key === 'ArrowUp') { setEnergy(a, -z[a] + d); e.preventDefault(); }
		if (e.key === 'ArrowDown') { setEnergy(a, -z[a] - d); e.preventDefault(); }
	}
	function railKey(e) {
		const d = e.shiftKey ? 1 : 0.25;
		if (e.key === 'ArrowUp') { setShift(shift - d); e.preventDefault(); }
		if (e.key === 'ArrowDown') { setShift(shift + d); e.preventDefault(); }
	}
	function rulerKey(e) {
		const f = e.shiftKey ? 1.5 : 1.15;
		if (e.key === 'ArrowUp') { tau = clamp(tau * f, 0.02, 100); e.preventDefault(); }
		if (e.key === 'ArrowDown') { tau = clamp(tau / f, 0.02, 100); e.preventDefault(); }
		if (e.key === 'Enter' || e.key === 'Home') { tau = 1; e.preventDefault(); }
	}
	function reset() { dz = null; shift = 0; tau = 1; }

	// numbers in words
	// a positive number given by its log10: plain below 10⁵, else mantissa × 10^e
	const sci = (lg10) => {
		if (lg10 < 5) return { m: fmt(10 ** lg10, lg10 < 2 ? 3 : 1), e: null };
		const e = Math.floor(lg10);
		return { m: `${fmt(10 ** (lg10 - e), 2)} × 10`, e: String(e).replace('-', '−') };
	};
	const lg10Z = $derived(lnZ / Math.LN10);
	const zz = $derived(sci(lg10Z));
	const focusA = $derived(active >= 0 ? active : p.indexOf(Math.max(...p)));
	const ctxText = $derived(tr.chars.slice(0, pos + 1).map(showChar).join(''));
	const ranked = $derived(p.map((q, a) => ({ q, a })).sort((u, v) => v.q - u.q));
	const ties = $derived.by(() => {
		const m = Math.max(...z);
		return z.filter((v) => v === m).length;
	});
	const readout = $derived.by(() => {
		const top3 = ranked.slice(0, 3).map(({ q, a }) => `‘${showChar(specimen.vocab[a])}’ ${fmt(q, 3)}`).join(', ');
		const parts = [`At τ = ${fmt(tau, 2)}${tau === 1 ? ' (the model’s own)' : ''}: ${top3}; entropy ${fmt(H, 2)} of a possible ${fmt(Math.log(V), 2)} nats.`];
		if (shift !== 0) parts.push(`Every energy shifted by ${fmt(-shift, 2)}: the partition function and F changed, the probabilities did not.`);
		if (dz != null) parts.push(ties > 1 ? `${ties} rungs are tied for lowest: cool the ladder and they share the probability equally.` : 'Logits changed by hand.');
		return parts;
	});
	const axisLabelX = $derived(xRail - (narrow ? 22 : 30));
</script>

<div class="fx fx-ladder" use:measure={(w) => (width = w)}>
	<svg
		bind:this={svgEl}
		viewBox="0 0 {W} {Hsvg}"
		width={W}
		height={Hsvg}
		role="img"
		aria-label="Energy ladder for the character after ‘{ctxText}’: {ranked.slice(0, 3).map(({ q, a }) => `${showChar(specimen.vocab[a])} ${fmt(q, 2)}`).join(', ')} at τ = {fmt(tau, 2)}"
		onpointermove={onmove}
		onpointerup={onup}
		onpointercancel={onup}
	>
		<CorpusStrip corpus={specimen.corpus} {T} bind:start bind:pos maxStart={specimen.corpus.length - T} x={narrow ? 8 : xRail - 30} y={24} width={narrow ? W - 16 : Math.min(520, W - xRail)} {interactive} label="context" />
		<text class="lbl" x={narrow ? 8 : xRail - 30} y={60}>the next character after ‘<tspan class="fx-ladder__ctx">{ctxText}</tspan>’ (window {start + 1}, position {pos + 1})</text>

		<!-- column heads -->
		<text class="lbl" x={xRail} y={top - 22} text-anchor="middle">energy <tspan class="m">E</tspan> = −<tspan class="m">z</tspan></text>
		<text class="lbl lbl--micro" x={xRail} y={top - 8} text-anchor="middle">less likely ↑</text>
		<text class="lbl lbl--micro" x={xRail} y={top + height + 17} text-anchor="middle">more likely ↓</text>
		<text class="lbl" x={xRail + 34 + 62 - 8} y={top - 22} text-anchor="end">logit <tspan class="m">z</tspan></text>
		<text class="lbl" x={xRail + 34 + 62 + 2} y={top - 22}>probability <tspan class="m">p</tspan></text>

		<EnergyLadder
			{z}
			labels={specimen.vocab}
			{tau}
			x={xRail}
			y={top}
			{height}
			{Emin}
			{Emax}
			{pScale}
			showZ
			showF
			ticks={2}
			{active}
			onrung={interactive ? (a, e) => begin('rung', e, a) : null}
			onrungkey={rungKey}
			onrail={interactive ? (e) => begin('rail', e) : null}
			onrailkey={railKey}
			onruler={interactive ? (e) => begin('ruler', e) : null}
			onrulerkey={rulerKey}
		/>

		<!-- the numbers of the Gibbs distribution, live -->
		<g class="fx-ladder__nums" transform="translate({xNum} {yNum})">
			<text class="lbl lbl--head" x="0" y="0">at <tspan class="m">τ</tspan> = {fmt(tau, 2)}</text>
			<text class="fx-ladder__eq" x="0" y="32"><tspan class="mz">𝒵</tspan> = Σ<tspan class="m sub" dy="4">a</tspan><tspan dy="-4"> exp(−</tspan><tspan class="m">E</tspan><tspan class="m sub" dy="4">a</tspan><tspan dy="-4">/</tspan><tspan class="m">τ</tspan>) = {zz.m}{#if zz.e}<tspan class="sup" dy="-7">{zz.e}</tspan>{/if}</text>
			<text class="fx-ladder__eq" x="0" y="60"><tspan class="m">F</tspan> = −<tspan class="m">τ</tspan> ln <tspan class="mz">𝒵</tspan> = {fmt(F, 2)}</text>
			<text class="fx-ladder__eq" x="0" y="86"><tspan class="m">H</tspan> = −Σ <tspan class="m">p</tspan> ln <tspan class="m">p</tspan> = {fmt(H, 2)} nats</text>
			<text class="lbl lbl--micro" x="0" y="104">of at most ln {V} = {fmt(Math.log(V), 2)}</text>
			<text class="fx-ladder__eq" x="0" y="138"><tspan class="m">p</tspan>(‘<tspan class="fx-ladder__ctx">{showChar(specimen.vocab[focusA])}</tspan>’) = exp(<tspan class="m">z</tspan>/<tspan class="m">τ</tspan>) / <tspan class="mz">𝒵</tspan></text>
			<text class="fx-ladder__eq" x="0" y="160"><tspan dx="38">= exp({fmt(z[focusA] / tau, 2)}) / <tspan class="mz">𝒵</tspan> = {fmt(p[focusA], 3)}</tspan></text>
			{#if interactive && modified}
				<text class="lbl fx-ladder__reset" x="0" y="196" role="button" tabindex="0" onclick={reset} onkeydown={(e) => (e.key === 'Enter' || e.key === ' ') && reset()}>↺ back to the model’s own numbers</text>
			{/if}
		</g>
	</svg>
	{#if interactive}
		<p class="readout" aria-live="polite">
			<span class="readout__who">After ‘{ctxText}’.</span>
			{#each readout as r, k (k)}<span class="readout__part">{r}</span>{/each}
			<span class="readout__hint">Drag a rung to change its logit, drag the axis to shift every energy at once, drag the τ-handle on the ruler. Click a character of the line above to change the context.</span>
		</p>
	{/if}
</div>

<style>
	.fx-ladder :global(.m) { font-family: var(--serif); font-style: italic; fill: var(--ink); }
	.fx-ladder :global(.sub) { font-size: 0.72em; }
	.fx-ladder :global(.sup) { font-size: 0.72em; }
	.fx-ladder :global(.sup2) { font-size: 0.6em; }
	.fx-ladder :global(.mz) { font-family: var(--math); font-size: 15px; fill: var(--ink); }
	.fx-ladder__ctx { font-family: var(--serif); fill: var(--ink); font-size: 14px; }
	.fx-ladder__eq { font-family: var(--sans); font-size: 13px; fill: var(--ink-2); font-variant-numeric: tabular-nums lining-nums; }
	.fx-ladder__eq .m { font-size: 15px; }
	.fx-ladder__reset { cursor: pointer; fill: var(--value-deep); }
	.fx-ladder__reset:hover { text-decoration: underline; }
</style>

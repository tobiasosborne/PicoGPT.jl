<script module>
	export const meta = { size: 'wide', title: 'The J-space part of a concept, and the rest of it' };
</script>

<script>
	// Figure: is the J-space part of a concept vector the part that drives the output?
	// For each character a and depth ℓ: the concept vector m_a (mean column before a, minus
	// the mean column), its J-component m_a^J (nearest point of the J-space, k = 3, exact),
	// and the remainder r_a = m_a − m_a^J. Each of v_a, m_a^J, r_a is rescaled to ‖m_a‖ and
	// added to the column at every position whose next character is not a; the bar is the
	// mean change of log p_a there, in nats. Hatched (covector ink): the lens vector v_a;
	// solid: the J-component; open: the remainder. Under each character: the share of ‖m_a‖²
	// in the J-component, and cos(v_a, r_a), zero when v_a is one of the fit's vectors.
	// Numbers: J-lens export interventions.by_depth[j].concept_mean_delta_logp, concept_vectors.
	import { getContext } from 'svelte';
	import { measure } from '#figures/primitives/measure.js';
	import { fmt, fmtDense, showChar, maxAbs, layoutWidth } from '#figures/primitives/scales.js';
	import { at, fibre, depthLabel, dot, norm } from './jdata.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const J = $derived(getContext('artifacts')()?.jspace?.specimen ?? null);
	const interactive = $derived(mode !== 'print');
	let width = $state(868);
	let focus = $state(null); // {j, a}

	const vocab = $derived(J?.vocab ?? []);
	const V = $derived(vocab.length);
	// component order in the export: j_k3, remainder, lens_vector; drawn as lens, J, remainder
	const KIND = $derived.by(() => {
		const names = J?.interventions.concept_components ?? [];
		const ix = (k) => names.indexOf(k);
		return [{ key: 'lens_vector', i: ix('lens_vector'), cls: 'lens', label: 'v_a' }, { key: 'j_k3', i: ix('j_k3'), cls: 'j', label: 'J-part' }, { key: 'remainder', i: ix('remainder'), cls: 'rest', label: 'rest' }];
	});

	const rows = $derived.by(() => {
		if (!J) return [];
		return J.depth.map((d, j) => {
			const B = J.interventions.by_depth[j];
			return {
				d, j,
				chars: vocab.map((_, a) => {
					const vec = (k) => fibre(B.concept_vectors, [null, a, k], 0);
					const comp = vec(KIND[1].i), rest = vec(KIND[2].i), v = vec(KIND[0].i);
					const m = comp.map((x, q) => x + rest[q]);
					const nr = norm(rest);
					return {
						a,
						delta: KIND.map((K) => at(B.concept_mean_delta_logp, a, K.i)),
						share: dot(comp, comp) / dot(m, m),
						cosRest: nr > 1e-12 ? dot(v, rest) / (norm(v) * nr) : 0
					};
				})
			};
		});
	});
	const dmax = $derived(maxAbs(rows.map((r) => r.chars.map((c) => c.delta))));

	// geometry
	const W = $derived(layoutWidth(width, mode, 340));
	const narrow = $derived(W < 620);
	const padL = $derived(narrow ? 6 : 96);
	const padR = 6;
	const cw = $derived((W - padL - padR) / Math.max(1, V));
	const bw = $derived(Math.min(16, cw * 0.2));
	const bx = (a, k) => padL + cw * a + cw / 2 + (k - 1) * (bw + 2) - bw / 2;
	const hBar = $derived(narrow ? 54 : 70);
	const rowH = $derived(hBar + (narrow ? 76 : 74));
	const top = 30;
	const base = (j) => top + 18 + j * rowH + hBar + (narrow ? 14 : 0);
	const H = $derived(base(2) + 52);
	const ys = (v) => (v / dmax) * hBar;

	const readout = $derived.by(() => {
		if (!focus || !rows.length) return null;
		const c = rows[focus.j].chars[focus.a];
		return `depth ${depthLabel(rows[focus.j].d)}, ‘${showChar(vocab[focus.a])}’: mean change of log p = ${fmt(c.delta[0], 2)} nats with v_a, ${fmt(c.delta[1], 2)} with the J-component, ${fmt(c.delta[2], 2)} with the remainder (all at the length of the concept vector); the J-component holds ${fmt(100 * c.share, 0)} % of ‖m_a‖²; cos(v_a, r_a) = ${fmt(c.cosRest, 2)}.`;
	});
	const ticks = $derived([0, 5, 10].filter((t) => t <= dmax));
</script>

<div class="fx fx-priv" use:measure={(w) => (width = w)}>
	{#if !J}
		<p class="fig__pending">The J-lens export (jspace/artifacts/specimen-jlens.json) is missing.</p>
	{:else}
		<svg viewBox="0 0 {W} {H}" width={W} height={H} role={interactive ? 'group' : 'img'} aria-label="Mean change of log p_a, in nats, when writing the lens vector, the J-component or the remainder of the concept vector at equal length: {rows.map((r) => `depth ${depthLabel(r.d)}: ${r.chars.map((c) => `${showChar(vocab[c.a])} ${fmt(c.delta[0], 1)}/${fmt(c.delta[1], 1)}/${fmt(c.delta[2], 1)}`).join(', ')}`).join('; ')}">
			<!-- what the three bars are, once -->
			<g transform="translate({narrow ? 6 : padL} {12})">
				<rect class="priv__bar priv__bar--lens" x="0" y="-9" width="9" height="11" />
				<text class="lbl lbl--micro" x="14" y="1">lens vector <tspan class="priv__m">v</tspan><tspan class="priv__m priv__sub" dy="2">a</tspan></text>
				<rect class="priv__bar priv__bar--j" x={narrow ? 104 : 118} y="-9" width="9" height="11" />
				<text class="lbl lbl--micro" x={narrow ? 118 : 132} y="1">J-component of <tspan class="priv__m">m</tspan><tspan class="priv__m priv__sub" dy="2">a</tspan></text>
				<rect class="priv__bar priv__bar--rest" x={narrow ? 228 : 268} y="-9" width="9" height="11" />
				<text class="lbl lbl--micro" x={narrow ? 242 : 282} y="1">remainder</text>
			</g>
			{#each rows as r, j (j)}
				{@const b0 = base(j)}
				{#if narrow}
					<text class="lbl lbl--head" x={6} y={b0 - hBar - 6}>depth {depthLabel(r.d)}</text>
				{:else}
					<text class="lbl lbl--head" x={4} y={b0 - hBar + 10}>depth {depthLabel(r.d)}</text>
					<text class="lbl lbl--micro" x={4} y={b0 - hBar + 24}>Δ ln p, nats</text>
					{#each ticks as t (t)}
						<line class="rule" x1={padL - 8} x2={W - padR} y1={b0 - ys(t)} y2={b0 - ys(t)} />
						<text class="lbl lbl--micro" x={padL - 12} y={b0 - ys(t) + 3} text-anchor="end">{t}</text>
					{/each}
				{/if}
				<line class="axis" x1={padL} x2={W - padR} y1={b0} y2={b0} />
				{#each r.chars as c (c.a)}
					{#each KIND as K, k (K.key)}
						{@const h = ys(c.delta[k])}
						<rect class="priv__bar priv__bar--{K.cls}" x={bx(c.a, k)} y={h >= 0 ? b0 - h : b0} width={bw} height={Math.abs(h)} />
					{/each}
					<text class="tok priv__ch" x={padL + cw * c.a + cw / 2} y={b0 + 19} text-anchor="middle">{showChar(vocab[c.a])}</text>
					<text class="lbl lbl--micro" x={padL + cw * c.a + cw / 2} y={b0 + 32} text-anchor="middle">{fmt(100 * c.share, 0)}%</text>
					<text class="lbl lbl--micro" class:priv__nonzero={Math.abs(c.cosRest) >= 0.005} x={padL + cw * c.a + cw / 2} y={b0 + 44} text-anchor="middle">{fmtDense(c.cosRest, 2)}</text>
					{#if interactive}
						<rect class="hit" x={padL + cw * c.a} y={b0 - hBar} width={cw} height={hBar + 48} role="button" tabindex="0" aria-label="depth {depthLabel(r.d)}, ‘{showChar(vocab[c.a])}’"
							onpointerenter={() => (focus = { j, a: c.a })} onfocus={() => (focus = { j, a: c.a })} onclick={() => (focus = { j, a: c.a })} onkeydown={() => {}} />
					{/if}
				{/each}
				{#if !narrow}
					<text class="lbl lbl--micro" x={padL - 12} y={b0 + 32} text-anchor="end">J share of ‖m‖²</text>
					<text class="lbl lbl--micro" x={padL - 12} y={b0 + 44} text-anchor="end">cos(v, r)</text>
				{/if}
			{/each}
		</svg>
		{#if interactive}
			<p class="readout" aria-live="polite">
				{#if readout}<span class="readout__part">{readout}</span>{:else}<span class="readout__hint">Point at a character to read its three numbers.</span>{/if}
			</p>
		{/if}
	{/if}
</div>

<style>
	.priv__bar--lens { fill: url(#hatch-covector); stroke: var(--covector); stroke-width: var(--hair); }
	.priv__bar--j { fill: var(--value); }
	.priv__bar--rest { fill: var(--paper); stroke: var(--value); stroke-width: 1px; }
	.priv__ch { font-size: 16px; fill: var(--ink); }
	.priv__m { font-family: var(--serif); font-style: italic; font-size: 13px; fill: var(--ink); }
	.priv__sub { font-size: 9.5px; }
	.priv__nonzero { fill: var(--ink); font-weight: 650; }
</style>

<script module>
	export const meta = { size: 'wide', title: 'The J-coordinates of one column, as k grows' };
</script>

<script>
	// Figure: what are the J-coordinates of a column, as k grows?
	// One column of the specimen at its three depths (small multiples). In each panel, row k
	// (k = 1 … V) is the exact nearest point of the J-space with sparsity k: every k-subset
	// of the V lens vectors tried, nonnegative least squares on each. A dot in the column of
	// character a means v_a is in the winning subset; its area ∝ the length ‖c_a v_a‖ of that
	// term of x_J (one scale for the whole figure); a ring marks a chosen vector whose
	// coefficient came out zero. At the right of each row, the share of ‖x‖² the J-space
	// misses, d_k(x)²/‖x‖² (bar, value ink), with the same share for random frames of V unit
	// vectors (tick, the mean of 20 frames): the control.
	// Data: jspace.specimen.j_space[j].{selected, coefficients, distance, norm, random_fraction}
	// and lenses.vectors; column n = t + (b−1)T, window b starting at corpus position b − 1.
	// Interactive: click a character (or drag the window) to choose the column; point at a
	// row for its numbers. Print: the stated column.
	import { getContext } from 'svelte';
	import { measure } from '#figures/primitives/measure.js';
	import CorpusStrip from '#figures/primitives/CorpusStrip.svelte';
	import { fmt, showChar, clamp, layoutWidth, areaRadius } from '#figures/primitives/scales.js';
	import { at, depthLabel } from '../primitives/jdata.js';
	import { norm } from '../primitives/cone.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const js = $derived(getContext('artifacts')()?.jspace?.specimen ?? null);
	const T = $derived(specimen.config.T), V = $derived(specimen.config.V), D = $derived(specimen.config.D);
	const vocab = $derived(specimen.vocab);
	const interactive = $derived(mode !== 'print');

	// svelte-ignore state_referenced_locally
	let start = $state(clamp((+attrs.window || 1) - 1, 0, 9));
	// svelte-ignore state_referenced_locally
	let pos = $state(clamp((+attrs.column || 4) - 1, 0, 7));
	let hover = $state(null); // { j, k }
	let width = $state(868);
	const W = $derived(layoutWidth(width, mode, 330));
	const narrow = $derived(W < 700);

	const n = $derived(start * T + pos);
	const tr = $derived(specimen.windowTrace(start));
	const depths = $derived(js?.depth ?? []);
	// per depth: rows k = 1..V with the subset, coefficients, term lengths, and the shares
	const panels = $derived(js ? depths.map((d, j) => {
		const S = js.j_space[j];
		const vlen = Array.from({ length: V }, (_, a) => norm(Array.from({ length: D }, (_, q) => at(js.lenses.vectors, a, q, j))));
		const xn = S.norm[n];
		const rows = Array.from({ length: V }, (_, kk) => {
			const terms = [];
			for (let slot = 0; slot <= kk; slot++) {
				const a = at(S.selected, slot, kk, n) - 1, c = at(S.coefficients, slot, kk, n);
				if (a >= 0) terms.push({ a, c, len: c * vlen[a] });
			}
			return { k: kk + 1, terms, frac: at(S.distance, kk, n) ** 2 / xn ** 2, random: at(S.random_fraction, kk, n) };
		});
		return { j, d, xn, rows };
	}) : []);
	const lenMax = $derived(Math.max(1e-9, ...panels.flatMap((p) => p.rows.flatMap((r) => r.terms.map((t) => t.len))))); // one scale

	// layout
	const cols = $derived(narrow ? 1 : panels.length);
	const gap = $derived(narrow ? 0 : 30);
	const panelW = $derived((W - gap * (cols - 1)) / cols);
	const cell = $derived(Math.min(narrow ? 30 : 27, (panelW - 34 - (narrow ? 110 : 80)) / V));
	const barW = $derived(panelW - 34 - cell * V - 14);
	const top0 = 64;
	const head = 44;
	const panelH = $derived(head + V * cell + 26);
	const px = (i) => (narrow ? 0 : i * (panelW + gap));
	const py = (i) => (narrow ? top0 + i * (panelH + 14) : top0);
	const H = $derived(narrow ? top0 + panels.length * (panelH + 14) : top0 + panelH + 8);
	const rDot = $derived(cell * 0.46);

	const vh = (a) => `<i>v</i><sub>${showChar(vocab[a])}</sub>`;
	const pct = (f) => `${fmt(100 * f, 0)}%`;
	const readout = $derived.by(() => {
		if (!panels.length) return null;
		const h = hover ?? { j: 1, k: 2 };
		const P = panels[h.j], r = P.rows[h.k - 1];
		const terms = r.terms.filter((t) => t.c > 0).map((t) => `${fmt(t.c)} ${vh(t.a)}`).join(' + ') || '0';
		return {
			who: `‘${showChar(tr.chars[pos])}’ at position ${pos + 1} of “${tr.text}”, depth ${depthLabel(P.d)}, k = ${r.k}:`,
			body: `<i>x</i><sub>J</sub> = ${terms}; the J-space misses ${pct(r.frac)} of ‖<i>x</i>‖² (random frames of ${V} directions: ${pct(r.random)}).`
		};
	});
	const aria = $derived(panels.map((P) => `depth ${depthLabel(P.d)}: ${P.rows.map((r) => `k=${r.k} {${r.terms.filter((t) => t.c > 0).map((t) => showChar(vocab[t.a])).join(',')}} misses ${pct(r.frac)}`).join('; ')}`).join('. '));
</script>

<div class="fx fx-dec" use:measure={(w) => (width = w)}>
	{#if js}
		<svg viewBox="0 0 {W} {H}" width={W} height={H} role={interactive ? 'group' : 'img'} aria-label="J-space decompositions of ‘{showChar(tr.chars[pos])}’ at position {pos + 1} of “{tr.text}”. {aria}">
			<CorpusStrip corpus={specimen.corpus} {T} bind:start bind:pos x={narrow ? 8 : 34} y={22} width={Math.min(W - 60, 340)} {interactive} label="the column x" />
			{#each panels as P, i (P.j)}
				{@const X0 = px(i) + 34}
				{@const Y0 = py(i)}
				{@const bx = X0 + V * cell + 14}
				<g class="fx-dec__panel">
					<text class="lbl lbl--head" x={px(i)} y={Y0 + 10}>depth {depthLabel(P.d)}</text>
					<text class="lbl lbl--micro" x={px(i) + 64} y={Y0 + 10}>‖x‖ = {fmt(P.xn)}</text>
					{#each vocab as ch, a (a)}
						<text class="tok fx-dec__ch" x={X0 + a * cell + cell / 2} y={Y0 + head - 10} text-anchor="middle">{showChar(ch)}</text>
					{/each}
					<text class="lbl lbl--micro" x={bx} y={Y0 + head - 10}>share missed</text>
					{#each P.rows as r (r.k)}
						{@const y = Y0 + head + (r.k - 1) * cell + cell / 2}
						{@const on = hover && hover.j === P.j && hover.k === r.k}
						<text class="lbl lbl--micro fx-dec__k" class:lbl--strong={on} x={X0 - 8} y={y + 3.5} text-anchor="end">k = {r.k}</text>
						{#each vocab as _, a (a)}
							<circle class="fx-dec__grid" cx={X0 + a * cell + cell / 2} cy={y} r="0.8" />
						{/each}
						{#each r.terms as t (t.a)}
							{#if t.c > 0}
								<circle class="fx-dec__dot" cx={X0 + t.a * cell + cell / 2} cy={y} r={areaRadius(t.len, lenMax, rDot)} />
							{:else}
								<circle class="fx-dec__zero" cx={X0 + t.a * cell + cell / 2} cy={y} r={rDot * 0.32} />
							{/if}
						{/each}
						<line class="rule" x1={bx} x2={bx + barW} y1={y} y2={y} />
						<rect class="bar--value" x={bx} y={y - cell * 0.2} width={barW * r.frac} height={cell * 0.4} />
						<line class="fx-dec__rand" x1={bx + barW * r.random} x2={bx + barW * r.random} y1={y - cell * 0.36} y2={y + cell * 0.36} />
						{#if r.k === V}
							<text class="lbl lbl--micro" x={bx + barW * r.random} y={y + cell * 0.36 + 10} text-anchor="middle">random</text>
						{/if}
						{#if interactive}
							<rect class="hit" x={px(i)} y={y - cell / 2} width={panelW} height={cell} role="button" tabindex="0" aria-label="depth {depthLabel(P.d)}, k = {r.k}" onpointerenter={() => (hover = { j: P.j, k: r.k })} onfocus={() => (hover = { j: P.j, k: r.k })} />
						{/if}
					{/each}
					<text class="lbl lbl--micro" x={bx} y={Y0 + head + V * cell + 12}>0</text>
					<text class="lbl lbl--micro" x={bx + barW} y={Y0 + head + V * cell + 12} text-anchor="end">all</text>
				</g>
			{/each}
		</svg>
		{#if interactive && readout}
			<p class="readout" aria-live="polite">
				<span class="readout__who">{readout.who}</span>
				<span class="readout__part">{@html readout.body}</span>
				<span class="readout__hint">Click a character for another column; point at a row for its numbers.</span>
			</p>
		{/if}
	{:else}
		<p class="fig__pending">The J-lens export (jspace/artifacts/specimen-jlens.json) is not available.</p>
	{/if}
</div>

<style>
	.fx-dec__ch { font-size: 14px; fill: var(--ink-2); }
	.fx-dec__grid { fill: var(--ink-3); opacity: 0.5; }
	.fx-dec__dot { fill: var(--value); }
	.fx-dec__zero { fill: none; stroke: var(--value); stroke-width: var(--hair); }
	.fx-dec__rand { stroke: var(--ink); stroke-width: 1.2px; }
</style>

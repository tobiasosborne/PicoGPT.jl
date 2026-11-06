<script module>
	export const meta = { size: 'wide', title: 'The J-lens vectors of the specimen, against the readout rows' };
	export { default as Key } from './fig-j-lens-vectors.key.svelte';
</script>

<script>
	// Figure: what are the J-lens vectors, and how do they differ from the readout's rows?
	// Rows: the depths 0, ½, 1 of the specimen, and at the bottom the readout's own rows w_a
	// (the logit lens's covectors, the same at every depth). Columns: the V characters. Each
	// glyph is a covector in ℝ^D as a spoke glyph in covector ink (bar tips; open bar =
	// negative component), all on ONE scale. Under each lens vector, its cosine with the
	// readout row of the same character. At depth 1 (= L) the rest of the model is the
	// identity, J_L = I, and the two rows coincide.
	// Data: jspace.specimen.lenses.{vectors[a, q, j], logit_vectors[a, q, j]}.
	// Interactive: point at a glyph for its numbers. Print: the same, static.
	import { getContext } from 'svelte';
	import { measure } from '#figures/primitives/measure.js';
	import SpokeGlyph from '#figures/primitives/SpokeGlyph.svelte';
	import { fmt, fmtDense, showChar, maxAbs, spokeScale, layoutWidth } from '#figures/primitives/scales.js';
	import { at, depthLabel } from '../primitives/jdata.js';
	import { dot, norm } from '../primitives/cone.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const js = $derived(getContext('artifacts')()?.jspace?.specimen ?? null);
	const V = $derived(specimen.config.V), D = $derived(specimen.config.D);
	const vocab = $derived(specimen.vocab);
	const interactive = $derived(mode !== 'print');
	let hover = $state(null); // { r, a }
	let width = $state(868);
	const W = $derived(layoutWidth(width, mode, 330));
	const narrow = $derived(W < 600);

	const depths = $derived(js?.depth ?? []);
	const vec = (t, a, j) => Array.from({ length: D }, (_, q) => at(t, a, q, j));
	// rows: one per depth (lens vectors), then the readout rows
	const rows = $derived(js ? [
		...depths.map((d, j) => ({ key: `d${j}`, label: `depth ${depthLabel(d)}`, sym: 'v', vs: vocab.map((_, a) => vec(js.lenses.vectors, a, j)), j })),
		{ key: 'w', label: 'readout rows', sym: 'w', vs: vocab.map((_, a) => vec(js.lenses.logit_vectors, a, 0)), j: -1 }
	] : []);
	const W_rows = $derived(rows.at(-1)?.vs ?? []);
	const cos = (u, w) => dot(u, w) / (norm(u) * norm(w));

	// layout: one scale for every glyph
	const padL = $derived(narrow ? 40 : 108);
	const step = $derived((W - padL - 6) / V);
	const rG = $derived(Math.min(narrow ? 19 : 34, step * 0.4));
	const sc = $derived(spokeScale(rG, maxAbs(rows.map((r) => r.vs))));
	const rowH = $derived(rG * 2 + (narrow ? 30 : 36));
	const top = 30;
	const xs = (a) => padL + step * (a + 0.5);
	const ys = (r) => top + rG + 6 + r * rowH + (r === rows.length - 1 ? 10 : 0);
	const H = $derived(rows.length ? ys(rows.length - 1) + rG + 30 : 40);

	const readout = $derived.by(() => {
		if (!rows.length) return null;
		const h = hover ?? { r: 0, a: vocab.indexOf('t') >= 0 ? vocab.indexOf('t') : 0 };
		const row = rows[h.r], v = row.vs[h.a], ch = showChar(vocab[h.a]);
		const big = v.map((x, q) => ({ x, q })).sort((p, q) => Math.abs(q.x) - Math.abs(p.x)).slice(0, 2);
		const comps = big.map((b) => `component ${b.q + 1} is ${fmt(b.x)}`).join(', ');
		if (row.j < 0) return { who: `The readout row w for ‘${ch}’:`, body: `‖w‖ = ${fmt(norm(v))}; its largest ${comps}.` };
		return { who: `The J-lens vector for ‘${ch}’ at ${row.label}:`, body: `‖v‖ = ${fmt(norm(v))}, ${fmt(norm(v) / norm(W_rows[h.a]))} times the readout row's length; its cosine with the readout row is ${fmt(cos(v, W_rows[h.a]))}; its largest ${comps}.` };
	});
	const aria = $derived(rows.slice(0, -1).map((r) => `${r.label}: cosines with the readout rows ${r.vs.map((v, a) => `${showChar(vocab[a])} ${fmt(cos(v, W_rows[a]))}`).join(', ')}`).join('; '));
</script>

<div class="fx fx-lv" use:measure={(w) => (width = w)}>
	{#if js}
		<svg viewBox="0 0 {W} {H}" width={W} height={H} role={interactive ? 'group' : 'img'} aria-label="J-lens vectors of the specimen as covector spoke glyphs. {aria}">
			{#each vocab as ch, a (a)}
				<text class="tok fx-lv__ch" x={xs(a)} y={16} text-anchor="middle">{showChar(ch)}</text>
			{/each}
			{#each rows as row, r (row.key)}
				{#if r === rows.length - 1}
					<line class="rule" x1={0} x2={W} y1={ys(r) - rG - 18} y2={ys(r) - rG - 18} />
				{/if}
				<text class="lbl" class:lbl--head={!narrow} x={0} y={ys(r) - 2}>{narrow ? (row.j < 0 ? 'w' : depthLabel(depths[row.j])) : row.label}</text>
				{#if !narrow}
					<text class="lbl lbl--micro" x={0} y={ys(r) + 13}>{row.j < 0 ? 'logit lens, every depth' : row.j === depths.length - 1 ? 'J = I here' : 'rows of Wout J'}</text>
				{/if}
				{#each row.vs as v, a (a)}
					<SpokeGlyph values={v} x={xs(a)} y={ys(r)} rmax={rG} scale={sc.scale} ink="covector" tip={narrow ? 1.6 : 2.2} weight={hover && hover.r === r && hover.a === a ? 1.6 : 1} />
					{#if row.j >= 0}
						<text class="lbl lbl--micro fx-lv__cos" x={xs(a)} y={ys(r) + rG + 15} text-anchor="middle">{fmtDense(cos(v, W_rows[a]), 2)}</text>
					{/if}
					{#if interactive}
						<rect class="hit" x={xs(a) - step / 2} y={ys(r) - rG - 6} width={step} height={rG * 2 + 22} role="button" tabindex="0" aria-label="{row.label}, ‘{showChar(vocab[a])}’" onpointerenter={() => (hover = { r, a })} onfocus={() => (hover = { r, a })} />
					{/if}
				{/each}
			{/each}
			{#if !narrow}
				<text class="lbl lbl--micro" x={0} y={ys(0) + rG + 15}>cos with w</text>
			{/if}
		</svg>
		{#if interactive && readout}
			<p class="readout" aria-live="polite">
				<span class="readout__who">{readout.who}</span>
				<span class="readout__part">{readout.body}</span>
				<span class="readout__hint">Point at a glyph for its numbers. All glyphs share one scale.</span>
			</p>
		{/if}
	{:else}
		<p class="fig__pending">The J-lens export (jspace/artifacts/specimen-jlens.json) is not available.</p>
	{/if}
</div>

<style>
	.fx-lv__ch { font-size: 17px; fill: var(--ink); }
	.fx-lv__cos { fill: var(--ink-2); }
</style>

<script module>
	export const meta = { size: 'wide', title: 'Replacing the rest of the model by its averaged derivative' };
	export { default as Key } from './fig-j-idea.key.svelte';
</script>

<script>
	// Figure: the idea of the book for one column of the specimen. The column runs down its
	// stem through the three depths ℓ = 0, ½, 1 (spoke glyphs of x_ℓ, value ink): this is
	// the real rest of the model, and at the foot it reaches the readout (the model's own
	// distribution). From each depth a hairline bypass leaves the stem, passes through ONE
	// matrix J_ℓ (the averaged Jacobian, a Hinton glyph) and goes straight to the readout:
	// the lens's distribution at that depth, beside the others. At depth 1 nothing is left
	// to replace, and the bypass gives the model's own readout.
	// Data: the stream from specimen.windowTrace(start) (trained θ); J_ℓ, the lens and model
	// distributions from the J-lens export (jspace.specimen: jacobians.averaged,
	// lenses.j_probs, lenses.model_probs), column n = t + (b−1)T of its batch, whose window b
	// starts at corpus position b − 1 (checked below against the corpus).
	// Interactive: click a character (or drag the window) to choose the column; click a depth.
	// Print: the same picture, all three bypasses drawn alike.
	import { getContext } from 'svelte';
	import { measure } from '#figures/primitives/measure.js';
	import SpokeGlyph from '#figures/primitives/SpokeGlyph.svelte';
	import MatrixGlyph from '#figures/primitives/MatrixGlyph.svelte';
	import CorpusStrip from '#figures/primitives/CorpusStrip.svelte';
	import { fmt, fmtProb, showChar, clamp, maxAbs, spokeScale, layoutWidth } from '#figures/primitives/scales.js';
	import DistBars, { distBarsWidth } from '../primitives/DistBars.svelte';
	import { at, fibre, matrixOf, argmax, depthLabel, tv } from '../primitives/jdata.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const js = $derived(getContext('artifacts')()?.jspace?.specimen ?? null);
	const T = $derived(specimen.config.T);
	const D = $derived(specimen.config.D);
	const V = $derived(specimen.config.V);
	const interactive = $derived(mode !== 'print');

	// svelte-ignore state_referenced_locally
	let start = $state(clamp((+attrs.window || 1) - 1, 0, 9));
	// svelte-ignore state_referenced_locally
	let pos = $state(clamp((+attrs.column || 6) - 1, 0, 7));
	// svelte-ignore state_referenced_locally
	let sel = $state(attrs.depth != null ? clamp(Math.round(2 * +attrs.depth), 0, 2) : 0);
	let width = $state(868);
	const W = $derived(layoutWidth(width, mode, 330));
	const narrow = $derived(W < 560);

	const tr = $derived(specimen.windowTrace(start));
	const n = $derived(start * T + pos); // column of the export's batch (window b = start + 1)
	const depths = $derived(js?.depth ?? [0, 0.5, 1]);
	const J = $derived(depths.length);
	const xs = $derived([tr.X0[pos], tr.Xmid[pos], tr.Xout[pos]]);
	const lens = $derived(js ? depths.map((_, j) => fibre(js.lenses.j_probs, n, j)) : []);
	const model = $derived(js ? fibre(js.lenses.model_probs, n) : tr.probs[pos]);
	const nextChar = $derived(specimen.corpus[start + pos + 1]);
	const target = $derived(specimen.vocab.indexOf(nextChar));
	// the export's batch is the corpus's windows in order; say so if that ever stops being true
	const aligned = $derived(!js || argmax(model) === argmax(tr.probs[pos]));

	// one Hinton scale for the three matrices, one spoke scale for the three stations
	const Js = $derived(js ? depths.map((_, j) => matrixOf(js.jacobians.averaged, j)) : []);
	const jmax = $derived(js ? maxAbs(depths.map((_, j) => Array.from({ length: D * D }, (_, k) => Js[j](k % D, Math.floor(k / D))))) : 1);

	// geometry
	const rG = $derived(narrow ? 15 : 25);
	const padL = $derived(narrow ? 36 : 96);
	const xg = $derived(padL + rG);
	const cell = $derived(narrow ? 3.6 : 6.5);
	const jw = $derived(cell * D);
	const bw = $derived(narrow ? 7.5 : 13);
	const bgap = $derived(narrow ? 2 : 4);
	const barsW = $derived(distBarsWidth(V, bw, bgap));
	const xb = $derived(W - barsW - (narrow ? 6 : 104));
	const xj = $derived(narrow ? xg + rG + 18 : xg + rG + Math.max(40, (xb - xg - rG - jw) * 0.3));
	const top = $derived(narrow ? 96 : 104);
	const rowH = $derived(narrow ? 80 : 104);
	const rowY = (j) => top + j * rowH;
	const barH = $derived(narrow ? 30 : 44);
	const baseY = (j) => rowY(j) + barH / 2 + 2;
	const yModel = $derived(rowY(J));
	const H = $derived(yModel + barH / 2 + 30);
	const sc = $derived(spokeScale(rG, maxAbs(xs)));
	const stripW = $derived(Math.min(W - 24, specimen.corpus.length * (narrow ? 16 : 19)));

	const subOf = (d) => depthLabel(d);
	const pick = (j) => { if (interactive) sel = j; };

	const words = $derived.by(() => {
		if (!js) return null;
		const ctx = tr.chars.slice(0, pos + 1).map(showChar).join('');
		const parts = depths.map((d, j) => {
			const p = lens[j], a = argmax(p);
			return `at depth ${depthLabel(d)} the lens reads ‘${showChar(specimen.vocab[a])}’ (${fmtProb(p[a], 2, true)}), at total variation ${fmt(tv(p, model), 2)} from the model`;
		});
		const m = argmax(model);
		return {
			who: `The column of ‘${showChar(tr.chars[pos])}’ after “${ctx}”, position ${pos + 1} of window ${start + 1}.`,
			lens: parts.join('; ') + '.',
			model: `The model reads ‘${showChar(specimen.vocab[m])}’ (${fmtProb(model[m], 2, true)})${target >= 0 ? `; the text continues with ‘${showChar(nextChar)}’` : ''}.`
		};
	});
</script>

<div class="fx fx-idea" use:measure={(w) => (width = w)}>
	{#if !js}
		<p class="fig__pending">The J-lens export (jspace/artifacts/specimen-jlens.json) is missing.</p>
	{:else}
		<svg viewBox="0 0 {W} {H}" width={W} height={H} role={interactive ? 'group' : 'img'}
			aria-label="One column of the specimen at depths 0, ½ and 1, and the lens readouts from each depth beside the model's. {words?.lens ?? ''} {words?.model ?? ''}">
			<CorpusStrip corpus={specimen.corpus} {T} bind:start bind:pos x={(W - stripW) / 2} y={26} width={stripW} showTarget {interactive} label="the column" />

			<!-- the real rest of the model: the stem, value ink -->
			<path class="fx-idea__stem" d={depths.slice(1).map((_, k) => `M${xg} ${rowY(k) + rG + 3} V${rowY(k + 1) - rG - 3}`).join(' ') + ` M${xg} ${rowY(J - 1) + rG + 3} V${yModel} H${xb - 10}`} />
			<path class="fx-idea__arrow" d="M{xb - 15} {yModel - 3.5} L{xb - 9} {yModel} L{xb - 15} {yModel + 3.5}" />
			{#each depths.slice(1) as d, k (k)}
				{@const ym = (rowY(k) + rowY(k + 1)) / 2}
				<text class="lbl lbl--micro" x={xg - 8} y={ym + 4} text-anchor="end">+ {js.depth_kind[k + 1] === 'attention' ? 'attention' : 'MLP'}</text>
			{/each}
			<text class="lbl lbl--micro" x={xg + 8} y={yModel - 6}>the rest of the model: here, the readout</text>
			<text class="lbl fx-idea__rowlbl" x={xb + barsW + 10} y={yModel + 4}>{narrow ? '' : 'the model'}</text>

			{#each depths as d, j (j)}
				{@const y = rowY(j)}
				{@const on = !interactive || sel === j}
				<!-- the station: the column at depth ℓ -->
				<SpokeGlyph values={xs[j]} x={xg} {y} rmax={rG} scale={sc.scale} tip={narrow ? 1.6 : 2.3} />
				<text class="math fx-idea__x" x={xg - rG - 8} y={y + 1} text-anchor="end">x<tspan class="fx-idea__sub" dy="4">{subOf(d)}</tspan></text>
				{#if !narrow}<text class="lbl lbl--micro" x={xg - rG - 8} y={y + 16} text-anchor="end">depth {depthLabel(d)}</text>{/if}

				<!-- the bypass: J_ℓ, then the readout -->
				<g class="fx-idea__bypass" class:is-on={on}>
					<line x1={xg + rG + 4} x2={xj - 5} y1={y} y2={y} />
					<line x1={xj + jw + 5} x2={xb - 10} y1={y} y2={y} />
					<path class="fx-idea__arrowhead" d="M{xb - 15} {y - 3.5} L{xb - 9} {y} L{xb - 15} {y + 3.5}" />
				</g>
				<MatrixGlyph rows={Array.from({ length: D }, (_, i) => Array.from({ length: D }, (_, q) => Js[j](i, q)))} x={xj} y={y - jw / 2} {cell} vmax={jmax} grid />
				<text class="math math--small fx-idea__J" class:is-on={on} x={xj + jw / 2} y={y - jw / 2 - 6} text-anchor="middle">J<tspan class="fx-idea__sub" dy="3">{subOf(d)}</tspan></text>
				{#if !narrow && j === 0}
					<text class="lbl lbl--micro" x={(xj + jw + xb) / 2} y={y - 6} text-anchor="middle">then the readout</text>
				{/if}

				<DistBars values={lens[j]} x={xb} y={baseY(j)} {bw} gap={bgap} h={barH} vocab={specimen.vocab} chars={specimen.vocab} {target} charSize={narrow ? 10 : 11.5} muted={!on} pLabel={!narrow} />
				{#if !narrow}<text class="lbl fx-idea__rowlbl" class:lbl--strong={interactive && sel === j} x={xb + barsW + 10} y={y + 4}>lens at depth {depthLabel(d)}</text>{/if}

				{#if interactive}
					<rect class="hit" x={padL - 30} y={y - rowH / 2 + 6} width={W - padL + 24} height={rowH - 12} role="button" tabindex="0"
						aria-label="depth {depthLabel(d)}: read the lens from here" onclick={() => pick(j)} onpointerenter={() => pick(j)} onfocus={() => pick(j)}
						onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(j); } }} />
				{/if}
			{/each}

			<DistBars values={model} x={xb} y={yModel + barH / 2 + 2} {bw} gap={bgap} h={barH} vocab={specimen.vocab} chars={specimen.vocab} {target} charSize={narrow ? 10 : 11.5} pLabel={!narrow} />
		</svg>
		{#if interactive}
			<p class="readout" aria-live="polite">
				<span class="readout__who">{words.who}</span>
				<span class="readout__part">{words.lens[0].toUpperCase() + words.lens.slice(1)}</span>
				<span class="readout__part">{words.model}</span>
				{#if !aligned}<span class="readout__part">(The export's column and the window disagree: check the batch order.)</span>{/if}
				<span class="readout__hint">Click a character, or drag the window; point at a depth to follow its bypass.</span>
			</p>
		{/if}
	{/if}
</div>

<style>
	.fx-idea__stem { fill: none; stroke: var(--value); stroke-width: var(--line); }
	.fx-idea__arrow, .fx-idea__arrowhead { fill: none; stroke-width: var(--line); }
	.fx-idea__arrow { stroke: var(--value); }
	.fx-idea__bypass line, .fx-idea__bypass .fx-idea__arrowhead { stroke: var(--ink-3); stroke-width: var(--hair); }
	.fx-idea__bypass.is-on line, .fx-idea__bypass.is-on .fx-idea__arrowhead { stroke: var(--ink); stroke-width: 0.9px; }
	.fx-idea__x { font-size: 16px; }
	.fx-idea :global(.fx-idea__sub) { font-size: 11px; font-style: normal; }
	.fx-idea__J { fill: var(--ink-3); }
	.fx-idea__J.is-on { fill: var(--ink); }
	.fx-idea__rowlbl { fill: var(--ink-2); }
</style>

<script module>
	export const meta = { size: 'wide', title: 'What do the two inks mean?' };
	export { default as Key } from './fig-two-inks.key.svelte';
</script>

<script>
	// Figure: the two inks on one map, the readout z = W_out x + b_out of one column, set as
	// an augmented product: [W | b] (x; 1) = Wx + b. Left, forward, in value ink: x as a row
	// over the columns of W (with a 1 over b), z as a column beside the rows, and the loss
	// of this one prediction, ℓ = −log p_y. Right, backward, in covector ink, in mirror: ℓ̄ = 1,
	// z̄ = p − e_y, x̄ = Wᵀ z̄ in the row, [W̄ | b̄] = z̄ (xᵀ, 1). These are the exact pullbacks of
	// ℓ for this column; the engine's full-batch adjoints (lab.gradTrace) are the same
	// vectors divided by N, and the figure checks that they agree.
	// Clock t ∈ [0, 1]: forward phases, then backward phases. Each array has its own scale.
	import { getContext } from 'svelte';
	import { measure } from '../primitives/measure.js';
	import Scrubber from '../primitives/Scrubber.svelte';
	import { fmt, fmtProb, showChar, clamp, phase, layoutWidth } from '../primitives/scales.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const { D, V, T } = $derived(specimen.config);
	const train = $derived(specimen.lab.train);
	// svelte-ignore state_referenced_locally
	const b = clamp((+attrs.window || 2) - 1, 0, specimen.lab.train.x.length - 1);
	// svelte-ignore state_referenced_locally
	let pos = $state(clamp((+attrs.pos || 1) - 1, 0, specimen.config.T - 1));
	// svelte-ignore state_referenced_locally
	let t = $state(attrs.t != null ? +attrs.t : 1);
	$effect(() => { if (attrs.t != null) t = +attrs.t; });
	let width = $state(868);
	const interactive = $derived(mode !== 'print');

	const gt = $derived(specimen.lab.gradTrace(specimen.theta));
	const win = $derived(gt.windows[b]);
	const P = $derived(new Map(specimen.params().map((p) => [p.name.split('.').at(-1), p.values])));
	const Wo = $derived(P.get('Wout')); // V×D, column-major
	const bo = $derived(P.get('bout'));
	const Wat = (i, j) => Wo[i + V * j];
	const x = $derived(win.val.Xf[pos]);
	const z = $derived(win.val.Z[pos]);
	const p = $derived(win.val.probs[pos]);
	const y = $derived(win.y[pos]); // the engine's prepared batch holds 0-based ids
	const loss = $derived(-Math.log(p[y]));
	const zbar = $derived(p.map((v, a) => v - (a === y ? 1 : 0)));
	const xbar = $derived(Array.from({ length: D }, (_, j) => zbar.reduce((s, v, i) => s + Wat(i, j) * v, 0)));
	const Wbar = (i, j) => (j < D ? zbar[i] * x[j] : zbar[i]); // j = D is the bias column
	// the engine's backward pass, for this column, is the same up to 1/N
	const N = $derived(train.batch.N ?? train.x.length * T);
	const agree = $derived(Math.max(...zbar.map((v, a) => Math.abs(v / N - win.bar.Z[pos][a])), ...xbar.map((v, j) => Math.abs(v / N - win.bar.Xf[pos][j]))) < 1e-12);

	const amax = (arr) => Math.max(1e-12, ...arr.map(Math.abs));
	const sx = $derived(amax(x));
	const sW = $derived(Math.max(amax(Wo), amax(bo)));
	const sz = $derived(amax(z));
	const sxb = $derived(amax(xbar));
	const szb = $derived(amax(zbar));
	const sWb = $derived(Math.max(1e-12, ...Array.from({ length: V * (D + 1) }, (_, k) => Math.abs(Wbar(k % V, Math.floor(k / V))))));

	// clock
	const PHASES = [
		{ at: 0, label: 'x' },
		{ at: 0.12, label: 'z' },
		{ at: 0.4, label: 'ℓ' },
		{ at: 0.52, label: 'z̄' },
		{ at: 0.7, label: 'x̄ and W̄' },
		{ at: 1, label: '' }
	];
	const uZ = $derived(phase(t, 0.12, 0.38)); // rows of z, one by one
	const uL = $derived(phase(t, 0.38, 0.48));
	const uLb = $derived(phase(t, 0.5, 0.56));
	const uZb = $derived(phase(t, 0.54, 0.68));
	const uXb = $derived(phase(t, 0.7, 0.98)); // columns of x̄ and W̄, one by one
	const rowOn = (i) => clamp(uZ * V - i, 0, 1);
	const colOn = (j) => clamp(uXb * (D + 1) - j, 0, 1);
	const activeRow = $derived(uZ > 0 && uZ < 1 ? Math.min(V - 1, Math.floor(uZ * V)) : -1);
	const activeCol = $derived(uXb > 0 && uXb < 1 ? Math.min(D, Math.floor(uXb * (D + 1))) : -1);

	// geometry
	const W = $derived(layoutWidth(width, mode, 340));
	const narrow = $derived(W < 720);
	const panelW = $derived(narrow ? W : (W - 40) / 2);
	const c = $derived(Math.min(26, (panelW - 92) / (D + 3.2)));
	const ox = (k) => (narrow ? 0 : k * (panelW + 40)); // panel origin
	const oy = (k) => (narrow ? k * (top + 12 * c + 160) : 0);
	const top = 62;
	const lx = 34; // left of the matrix within a panel
	const yX = top + 14;
	const yM = $derived(yX + c + 22);
	const xZ = $derived(lx + (D + 1) * c + 0.25 * c + 34);
	const yL = $derived(yM + V * c + 30);
	const H = $derived(narrow ? oy(1) + yL + 94 : yL + 94);
	const sq = (v, s, k = 1) => c * 0.9 * Math.sqrt(Math.min(1, Math.abs(v) / s)) * k;
	const cellX = (j) => lx + c * j + (j === D ? 0.25 * c : 0);

	const readout = $derived(`Predicting position ${pos + 2} of window ${b + 1}, after “${train.x[b].slice(0, pos + 1).map((id) => showChar(specimen.vocab[id - 1])).join('')}”: the text has ‘${showChar(specimen.vocab[y])}’, to which the model gave p = ${fmtProb(p[y], 3, true)}, so ℓ = ${fmt(loss, 3)} nats. Its adjoint z̄ = p − e_y is largest at ${zbar.map((v, a) => ({ v, a })).sort((u, w) => Math.abs(w.v) - Math.abs(u.v)).slice(0, 2).map((o) => `‘${showChar(specimen.vocab[o.a])}’ (${fmt(o.v, 2)})`).join(' and ')}.`);
	const sub = (n) => String(n);
</script>

<div class="fx fx-inks" use:measure={(w) => (width = w)}>
	<svg viewBox="0 0 {W} {H}" width={W} height={H} role={interactive ? 'group' : 'img'} aria-label="The readout map forward in value ink and backward in covector ink. {readout}">
		<!-- the window: choose the column -->
		<g>
			{#each train.x[b] as id, k (k)}
				<text class="tok fx-inks__w" class:is-on={k === pos} x={(narrow ? 20 : W / 2 - 120) + k * 26} y={22} text-anchor="middle">{showChar(specimen.vocab[id - 1])}</text>
				{#if interactive}<rect class="hit" x={(narrow ? 20 : W / 2 - 120) + k * 26 - 13} y="2" width="26" height="30" role="button" tabindex="0" aria-label="column {k + 1}" onclick={() => (pos = k)} onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); (pos = k); } }} />{/if}
			{/each}
			<text class="lbl lbl--micro" x={(narrow ? 20 : W / 2 - 120) + T * 26 - 4} y={22}>→ ‘{showChar(specimen.vocab[y])}’ next</text>
			<line class="fx-inks__under" x1={(narrow ? 20 : W / 2 - 120) + pos * 26 - 8} x2={(narrow ? 20 : W / 2 - 120) + pos * 26 + 8} y1={28} y2={28} />
		</g>

		{#each [0, 1] as k (k)}
			{@const fwd = k === 0}
			<g transform="translate({ox(k)} {oy(k)})" class="fx-inks__panel" class:fx-inks__panel--back={!fwd}>
				<text class="lbl fx-inks__title" x={lx} y={top - 14}>{fwd ? 'forward' : 'backward'}</text>
				<text class="lbl lbl--micro" x={lx + (fwd ? 76 : 90)} y={top - 14}>{fwd ? 'value ink, computed down the page' : 'covector ink, pulled back up the page'}</text>

				<!-- the input row: x (with the 1 of the bias) / x̄ -->
				{#each Array.from({ length: D + 1 }) as _, j (j)}
					{@const X0 = cellX(j)}
					<rect class="fx-inks__slot" x={X0} y={yX} width={c} height={c} />
					{#if j < D}
						{#if fwd}
							{@const s = sq(x[j], sx)}
							<rect class="fx-inks__v" class:neg={x[j] < 0} x={X0 + (c - s) / 2} y={yX + (c - s) / 2} width={s} height={s} />
						{:else}
							{@const s = sq(xbar[j], sxb, colOn(j))}
							{#if s > 0.2}<rect class="fx-inks__c" class:neg={xbar[j] < 0} x={X0 + (c - s) / 2} y={yX + (c - s) / 2} width={s} height={s} />{/if}
						{/if}
					{:else}
						<text class="lbl lbl--micro" x={X0 + c / 2} y={yX + c / 2 + 3.5} text-anchor="middle">1</text>
					{/if}
				{/each}
				<text class="lbl" x={lx - 8} y={yX + c / 2 + 4} text-anchor="end"><tspan class="m" class:fx-inks__cm={!fwd}>x</tspan></text>
				{#if !fwd}<line class="fx-inks__bar" x1={lx - 17} x2={lx - 10} y1={yX + c / 2 - 7} y2={yX + c / 2 - 7} />{/if}

				<!-- the matrix [W | b] / [W̄ | b̄] with row labels (characters) -->
				{#each specimen.vocab as ch, i (i)}
					<text class="tok fx-inks__row" class:is-on={i === y} x={lx - 8} y={yM + c * (i + 0.5) + 4} text-anchor="end">{showChar(ch)}</text>
				{/each}
				<rect class="matrix__frame" x={lx} y={yM} width={D * c} height={V * c} />
				<rect class="matrix__frame" x={cellX(D)} y={yM} width={c} height={V * c} />
				{#each Array.from({ length: V * (D + 1) }) as _, q (q)}
					{@const i = q % V}
					{@const j = Math.floor(q / V)}
					{@const X0 = cellX(j)}
					{#if fwd}
						{@const v = j < D ? Wat(i, j) : bo[i]}
						{@const s = sq(v, sW)}
						<rect class="fx-inks__v" class:neg={v < 0} class:dim={activeRow >= 0 && activeRow !== i} x={X0 + (c - s) / 2} y={yM + c * i + (c - s) / 2} width={s} height={s} />
					{:else}
						{@const v = Wbar(i, j)}
						{@const s = sq(v, sWb, colOn(j))}
						{#if s > 0.2}<rect class="fx-inks__c" class:neg={v < 0} x={X0 + (c - s) / 2} y={yM + c * i + (c - s) / 2} width={s} height={s} />{/if}
					{/if}
				{/each}
				{#if fwd && activeRow >= 0}<rect class="fx-inks__hl" x={lx - 1} y={yM + c * activeRow} width={(D + 1.25) * c + 2} height={c} />{/if}
				{#if !fwd && activeCol >= 0}<rect class="fx-inks__hl fx-inks__hl--c" x={cellX(activeCol) - 1} y={yX - 1} width={c + 2} height={yM - yX + V * c + 2} />{/if}

				<!-- the output column z / z̄ -->
				{#each Array.from({ length: V }) as _, i (i)}
					<rect class="fx-inks__slot" x={xZ} y={yM + c * i} width={c} height={c} />
					{#if fwd}
						{@const s = sq(z[i], sz, rowOn(i))}
						{#if s > 0.2}<rect class="fx-inks__v" class:neg={z[i] < 0} x={xZ + (c - s) / 2} y={yM + c * i + (c - s) / 2} width={s} height={s} />{/if}
					{:else}
						{@const s = sq(zbar[i], szb, uZb)}
						{#if s > 0.2}<rect class="fx-inks__c" class:neg={zbar[i] < 0} x={xZ + (c - s) / 2} y={yM + c * i + (c - s) / 2} width={s} height={s} />{/if}
					{/if}
				{/each}
				<rect class="fx-inks__target" x={xZ - 2} y={yM + c * y - 2} width={c + 4} height={c + 4} />
				<text class="lbl" x={xZ + c + 8} y={yM + 14}><tspan class="m" class:fx-inks__cm={!fwd}>z</tspan></text>
				{#if !fwd}<line class="fx-inks__bar" x1={xZ + c + 8} x2={xZ + c + 15} y1={yM + 1} y2={yM + 1} />{/if}
				<text class="lbl lbl--micro" x={xZ + c + 8} y={yM + 30}>{fwd ? 'scores' : 'p − e'}</text>
				{#if !fwd}<text class="lbl lbl--micro" x={xZ + c + 8} y={yM + 43}>(open: the</text><text class="lbl lbl--micro" x={xZ + c + 8} y={yM + 55}>target, −)</text>{/if}
				<text class="lbl lbl--micro" x={xZ + c + 8} y={yM + c * y + c / 2 + 3.5}>← next</text>
				<text class="lbl fx-inks__eq" x={(cellX(D) + c + xZ) / 2} y={yM + (V * c) / 2 + 5} text-anchor="middle">{fwd ? '=' : '←'}</text>

				<!-- the loss: one number -->
				{#if fwd}
					<g opacity={uL > 0 ? 1 : 0}>
						<line class="fx-inks__flow" x1={xZ + c / 2} x2={xZ + c / 2} y1={yM + V * c + 4} y2={yL - 12} />
						<text class="lbl" x={xZ + c / 2} y={yL + 4} text-anchor="middle"><tspan class="m">ℓ</tspan> = −log <tspan class="m">p</tspan><tspan class="m sub" dy="3">{showChar(specimen.vocab[y])}</tspan><tspan dy="-3"> = {fmt(loss, 3)}</tspan></text>
						<text class="lbl lbl--micro" x={xZ + c / 2} y={yL + 20} text-anchor="middle">one number</text>
					</g>
				{:else}
					<g opacity={uLb > 0 ? 1 : 0}>
						<line class="fx-inks__flow fx-inks__flow--c" x1={xZ + c / 2} x2={xZ + c / 2} y1={yL - 12} y2={yM + V * c + 4} />
						<path class="fx-inks__head" d="M{xZ + c / 2 - 3.5},{yM + V * c + 10} L{xZ + c / 2},{yM + V * c + 4} L{xZ + c / 2 + 3.5},{yM + V * c + 10}" />
						<text class="lbl" x={xZ + c / 2} y={yL + 4} text-anchor="middle"><tspan class="m fx-inks__cm">ℓ</tspan> = 1</text>
						<line class="fx-inks__bar" x1={xZ + c / 2 - 14} x2={xZ + c / 2 - 8} y1={yL - 9} y2={yL - 9} />
						<text class="lbl lbl--micro" x={xZ + c / 2} y={yL + 20} text-anchor="middle">the seed</text>
					</g>
				{/if}
				<text class="lbl lbl--micro" x={lx} y={yL + 4}>{fwd ? '[W  b] (x, 1) = Wx + b' : 'x̄ = Wᵀz̄,   [W̄  b̄] = z̄ (xᵀ, 1)'}</text>
				<text class="lbl lbl--micro" x={lx} y={yL + 18}>{fwd ? 'W = W_out, b = b_out' : 'every value has a covector of its shape'}</text>
				<text class="lbl lbl--micro" x={lx} y={yL + 46}>full cell |entry| = {fwd ? `x ${sx.toPrecision(3)}; W,b ${sW.toPrecision(3)}` : `x̄ ${sxb.toPrecision(3)}; W̄,b̄ ${sWb.toPrecision(3)}`}</text>
				<text class="lbl lbl--micro" x={lx} y={yL + 60}>{fwd ? `z ${sz.toPrecision(3)}` : `z̄ ${szb.toPrecision(3)}`}</text>
			</g>
		{/each}
		{#if !agree}<text class="lbl lbl--covector" x={4} y={H - 4}>check failed: the engine’s adjoints disagree</text>{/if}
	</svg>
	{#if interactive}
		<Scrubber bind:t phases={PHASES} duration={3600} label="forward, then backward" ink={t >= 0.5 ? 'covector' : 'value'} />
		<p class="readout" aria-live="polite">
			<span class="readout__part">{readout}</span>
			<span class="readout__hint">Scrub the computation (or press play); click a character of the window to take another column.</span>
		</p>
	{/if}
</div>

<style>
	.fx-inks :global(.m) { font-family: var(--serif); font-style: italic; font-size: 16px; fill: var(--ink); }
	.fx-inks :global(.sub) { font-size: 10px; }
	.fx-inks :global(.fx-inks__cm) { fill: var(--covector-deep); }
	.fx-inks__title { font-weight: 650; fill: var(--ink); font-variant-caps: all-small-caps; letter-spacing: 0.06em; font-size: 14px; }
	.fx-inks__w { font-size: 18px; fill: var(--ink-3); }
	.fx-inks__w.is-on { fill: var(--ink); font-weight: 650; }
	.fx-inks__under { stroke: var(--ink); stroke-width: 1px; }
	.fx-inks__row { font-size: 14px; fill: var(--ink-2); }
	.fx-inks__row.is-on { fill: var(--ink); font-weight: 650; }
	.fx-inks__slot { fill: none; stroke: var(--rule-2); stroke-width: var(--hair); }
	.fx-inks__v { fill: var(--value); mix-blend-mode: var(--overprint); transition: opacity var(--t-quick); }
	.fx-inks__v.neg { fill: none; stroke: var(--value); stroke-width: 0.8px; }
	.fx-inks__v.dim { opacity: 0.3; }
	.fx-inks__c { fill: url(#hatch-covector); stroke: var(--covector); stroke-width: 0.8px; mix-blend-mode: var(--overprint); }
	.fx-inks__c.neg { fill: none; }
	.fx-inks__hl { fill: none; stroke: var(--ink); stroke-width: 0.8px; }
	.fx-inks__hl--c { stroke: var(--covector-deep); }
	.fx-inks__target { fill: none; stroke: var(--ink-3); stroke-width: var(--hair); stroke-dasharray: 2 2; }
	.fx-inks__eq { font-size: 18px; fill: var(--ink-3); }
	.fx-inks__flow { stroke: var(--value); stroke-width: 1px; }
	.fx-inks__flow--c { stroke: var(--covector); }
	.fx-inks__head { fill: none; stroke: var(--covector); stroke-width: 1px; }
	.fx-inks__bar { stroke: var(--covector-deep); stroke-width: 1.1px; }
</style>

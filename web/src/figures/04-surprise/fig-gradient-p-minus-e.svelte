<script module>
	export const meta = { size: 'wide', title: 'The gradient of the loss with respect to the logits' };
	export { default as Key } from './fig-gradient-p-minus-e.key.svelte';
</script>

<script>
	// Figure: one prediction n, forward (value ink, down) and backward (covector ink, up):
	//   z (the logits, lollipops)  →  p = softmax(z) (bars; the target's e_y outlined)
	//   →  ℓ = −ln p_y;   and back:  N·z̄ = p − e_y (hatched bars about zero, on top).
	// If other windows share this context, their gradients are added (bottom row):
	//   Σ (p − e_y) = n (p − p̂), zero exactly when the model predicts what followed.
	// Interactive: choose the window and position on the corpus line; untrained/trained.
	import { getContext } from 'svelte';
	import { measure } from '../primitives/measure.js';
	import CorpusStrip from '../primitives/CorpusStrip.svelte';
	import { fmt, showChar, clamp, maxAbs, layoutWidth } from '../primitives/scales.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const V = $derived(specimen.config.V);
	const T = $derived(specimen.config.T);
	const vocab = $derived(specimen.vocab);
	const interactive = $derived(mode !== 'print');
	let width = $state(868);

	const maxStart = $derived(specimen.lab.train.x.length - 1);
	// svelte-ignore state_referenced_locally
	let start = $state(clamp((+attrs.window || 3) - 1, 0, specimen.lab.train.x.length - 1));
	// svelte-ignore state_referenced_locally
	let pos = $state(clamp((+attrs.position || 1) - 1, 0, specimen.config.T - 1));
	// svelte-ignore state_referenced_locally
	let trained = $state(attrs.state !== 'untrained');

	const pr = $derived(specimen.predictions(trained ? specimen.theta : specimen.theta0));
	const n = $derived(start * T + pos);
	const z = $derived(pr.Z[n]);
	const p = $derived(pr.probs[n]);
	const y = $derived(pr.y[start][pos] - 1);
	const g = $derived(p.map((v, a) => v - (a === y ? 1 : 0)));
	const loss = $derived(-Math.log(p[y]));
	const ctx = $derived(specimen.corpus.slice(start, start + pos + 1));
	// every prediction with the same context
	const peers = $derived.by(() => {
		const out = [];
		for (let b = 0; b < pr.B; b++) for (let k = 0; k < T; k++) if (specimen.corpus.slice(b, b + k + 1) === ctx) out.push({ b, k, y: pr.y[b][k] - 1 });
		return out;
	});
	const gsum = $derived(p.map((v, a) => peers.length * v - peers.filter((o) => o.y === a).length));
	const zMax = $derived(maxAbs(pr.Z));

	// geometry
	const W = $derived(layoutWidth(width, mode, 340));
	const narrow = $derived(W < 640);
	const L = $derived(narrow ? 10 : 150);
	const R = $derived(narrow ? 34 : 210);
	const step = $derived((W - L - R) / V);
	const xs = $derived(Array.from({ length: V }, (_, a) => L + step * (a + 0.5)));
	const bw = $derived(Math.min(22, step * 0.42));
	const Y = $derived(narrow ? { strip: 26, gbar: 148, z: 258, p: 382, loss: 420, sum: 520 } : { strip: 26, gbar: 128, z: 238, p: 362, loss: 400, sum: 492 });
	const gAmp = 48, zAmp = 42, pAmp = 66;
	const shared = $derived(peers.length > 1);
	const H = $derived(shared ? Y.sum + gAmp + (narrow ? 40 : 34) : Y.loss + 26);
	const xArrow = $derived(W - R + (narrow ? 14 : 40));
	const fmtS = (v) => (Math.abs(v) < 0.005 ? fmt(0, 2) : fmt(v, 2));
</script>

{#snippet covector(vals, y0, label)}
	<line class="axis" x1={xs[0] - step / 2} x2={xs[V - 1] + step / 2} y1={y0} y2={y0} />
	{#each vals as v, a (a)}
		{@const h = Math.max(-1, Math.min(1, v)) * gAmp}
		<rect class="bar--covector" x={xs[a] - bw / 2} y={h > 0 ? y0 - h : y0} width={bw} height={Math.abs(h)} />
		{#if Math.abs(v) >= 0.005}
			<text class="lbl lbl--micro lbl--covector" class:lbl--strong={Math.abs(v) >= 0.1} x={xs[a]} y={v > 0 ? y0 - h - 6 : y0 - h + 14} text-anchor="middle">{fmt(v, 2)}</text>
		{/if}
	{/each}
	{#if !narrow}{@render label()}{/if}
{/snippet}

<div class="fx fx-grad" use:measure={(w) => (width = w)}>
	<svg viewBox="0 0 {W} {H}" width={W} height={H} role="img" aria-label="Prediction after ‘{ctx}’ of ‘{showChar(vocab[y])}’: p = {fmt(p[y], 3)}, loss {fmt(loss, 3)}; gradient {g.map((v, a) => `${showChar(vocab[a])} ${fmt(v, 2)}`).join(', ')}">
		<CorpusStrip corpus={specimen.corpus} {T} bind:start bind:pos {maxStart} x={narrow ? 6 : L - 20} y={Y.strip} width={narrow ? W - 12 : Math.min(520, W - L)} showTarget {interactive} label="prediction" />
		<!-- untrained / trained -->
		{#each [false, true] as tr (tr)}
			{#if interactive}
				<text class="lbl fx-grad__state" class:is-on={trained === tr} x={W - (tr ? 4 : 92)} y={narrow ? Y.strip + 46 : Y.strip} text-anchor="end" role="button" tabindex="0" aria-pressed={trained === tr} onclick={() => (trained = tr)} onkeydown={(e) => (e.key === 'Enter' || e.key === ' ') && (trained = tr)}>{tr ? 'trained' : 'untrained'}</text>
			{:else if trained === tr}
				<text class="lbl fx-grad__state is-on" x={W - 4} y={Y.strip} text-anchor="end">{tr ? 'trained' : 'untrained'}</text>
			{/if}
		{/each}

		<!-- the covector, on top: N·z̄ = p − e_y -->
		{#snippet gLabel()}
			<text class="lbl lbl--covector" x={L - 18} y={Y.gbar - 6} text-anchor="end">gradient</text>
			<text class="lbl lbl--covector" x={L - 18} y={Y.gbar + 10} text-anchor="end"><tspan class="m m--cov">N z̄</tspan> = <tspan class="m m--cov">p</tspan> − <tspan class="m m--cov">e</tspan><tspan class="m m--cov sub" dy="3">y</tspan></text>
		{/snippet}
		{@render covector(g, Y.gbar, gLabel)}

		{#if narrow}
			<text class="lbl lbl--micro lbl--covector" x={xs[0] - step / 2} y={Y.gbar - gAmp - 10}>gradient: p minus the one-hot vector of what came</text>
			<text class="lbl lbl--micro" x={xs[0] - step / 2} y={Y.z - zAmp - 12}>logits z</text>
			<text class="lbl lbl--micro" x={xs[0] - step / 2} y={Y.p - pAmp - 12}>probabilities p; outlined: what came</text>
		{/if}
		<!-- logits -->
		{#if !narrow}
			<text class="lbl" x={L - 18} y={Y.z - 4} text-anchor="end">logits</text>
			<text class="lbl" x={L - 18} y={Y.z + 12} text-anchor="end"><tspan class="m">z</tspan></text>
		{/if}
		<line class="axis" x1={xs[0] - step / 2} x2={xs[V - 1] + step / 2} y1={Y.z} y2={Y.z} />
		{#each z as v, a (a)}
			{@const yy = Y.z - (v / zMax) * zAmp}
			<line class="stem" x1={xs[a]} x2={xs[a]} y1={Y.z} y2={yy} />
			<circle class={v >= 0 ? 'dot--value' : 'dot--open'} cx={xs[a]} cy={yy} r="3.2" />
			<text class="lbl lbl--micro" x={xs[a] + 7} y={yy + (v >= 0 ? -2 : 9)}>{fmt(v, 2)}</text>
		{/each}

		<!-- probabilities and the observed one-hot -->
		{#if !narrow}
			<text class="lbl" x={L - 18} y={Y.p - 34} text-anchor="end">probabilities</text>
			<text class="lbl" x={L - 18} y={Y.p - 18} text-anchor="end"><tspan class="m">p</tspan> = softmax(<tspan class="m">z</tspan>)</text>
			<text class="lbl lbl--micro" x={L - 18} y={Y.p - 2} text-anchor="end">outlined: what came, <tspan class="m">e</tspan><tspan class="m sub" dy="3">y</tspan></text>
		{/if}
		<line class="axis" x1={xs[0] - step / 2} x2={xs[V - 1] + step / 2} y1={Y.p} y2={Y.p} />
		{#each p as v, a (a)}
			{#if a === y}<rect class="fx-grad__onehot" x={xs[a] - bw / 2 - 3} y={Y.p - pAmp} width={bw + 6} height={pAmp} />{/if}
			<rect class="bar--value" x={xs[a] - bw / 2} y={Y.p - v * pAmp} width={bw} height={v * pAmp} />
			{#if v >= 0.005}<text class="lbl lbl--micro" class:lbl--strong={v >= 0.1} x={xs[a]} y={Y.p - v * pAmp - 5} text-anchor="middle">{fmt(v, 2)}</text>{/if}
			<text class="tok fx-grad__ch" class:is-y={a === y} x={xs[a]} y={Y.p + 19} text-anchor="middle">{showChar(vocab[a])}</text>
		{/each}

		<!-- the loss -->
		<text class="lbl" x={xs[0] - step / 2} y={Y.loss + 16}><tspan class="m">ℓ</tspan> = −ln <tspan class="m">p</tspan>(‘<tspan class="fx-grad__tok">{showChar(vocab[y])}</tspan>’) = <tspan class="fx-grad__strong">{fmt(loss, 3)}</tspan> nats</text>

		<!-- forward down, backward up -->
		{#if !narrow}
			<g class="fx-grad__arrows">
				<line class="fx-grad__fwd" x1={xArrow - 12} x2={xArrow - 12} y1={Y.z - 34} y2={Y.loss + 4} />
				<path class="fx-grad__fwdhead" d="M{xArrow - 16} {Y.loss - 4} L{xArrow - 12} {Y.loss + 4} L{xArrow - 8} {Y.loss - 4}" />
				<text class="lbl lbl--value" x={xArrow - 20} y={(Y.z + Y.p) / 2} text-anchor="end">forward</text>
				<line class="fx-grad__bwd" x1={xArrow + 12} x2={xArrow + 12} y1={Y.loss + 4} y2={Y.gbar - 30} />
				<path class="fx-grad__bwdhead" d="M{xArrow + 8} {Y.gbar - 22} L{xArrow + 12} {Y.gbar - 30} L{xArrow + 16} {Y.gbar - 22}" />
				{#each [0.25, 0.5, 0.75] as f (f)}
					<line class="fx-grad__bwdbar" x1={xArrow + 8} x2={xArrow + 16} y1={Y.loss + 4 - f * (Y.loss + 4 - Y.gbar + 30)} y2={Y.loss + 4 - f * (Y.loss + 4 - Y.gbar + 30)} />
				{/each}
				<text class="lbl lbl--covector" x={xArrow + 22} y={(Y.z + Y.p) / 2}>backward</text>
				<text class="lbl lbl--micro" x={xArrow + 22} y={(Y.z + Y.p) / 2 + 15}><tspan class="m">ℓ̄</tspan> = 1</text>
			</g>
		{/if}

		{#if shared}
			{#snippet sumLabel()}
				<text class="lbl lbl--covector" x={L - 18} y={Y.sum - 22} text-anchor="end">summed over the</text>
				<text class="lbl lbl--covector" x={L - 18} y={Y.sum - 8} text-anchor="end">{peers.length} windows with this</text>
				<text class="lbl lbl--covector" x={L - 18} y={Y.sum + 6} text-anchor="end">context: {peers.length}(<tspan class="m m--cov">p</tspan> − <tspan class="m m--cov">p̂</tspan>)</text>
			{/snippet}
			<line class="rule" x1={xs[0] - step / 2} x2={W} y1={Y.loss + 40} y2={Y.loss + 40} />
			{@render covector(gsum, Y.sum, sumLabel)}
			{#if narrow}<text class="lbl lbl--micro lbl--covector" x={xs[0] - step / 2} y={Y.sum - gAmp - 4}>summed over the {peers.length} windows with this context</text>{/if}
			<text class="lbl lbl--micro" x={narrow ? xs[0] - step / 2 : xs[V - 1] + step / 2 + 10} y={narrow ? Y.sum + gAmp + 14 : Y.sum - 18}>the windows: {peers.map((o) => `${o.b + 1}`).join(', ')};</text>
			<text class="lbl lbl--micro" x={narrow ? xs[0] - step / 2 : xs[V - 1] + step / 2 + 10} y={narrow ? Y.sum + gAmp + 28 : Y.sum - 4}>after ‘{Array.from(ctx).map(showChar).join('')}’ came {peers.map((o) => `‘${showChar(vocab[o.y])}’`).join(', ')}</text>
		{/if}
	</svg>
	{#if interactive}
		<p class="readout" aria-live="polite">
			<span class="readout__who">Window {start + 1}, position {pos + 1}, {trained ? 'trained' : 'untrained'}.</span>
			<span class="readout__part">After ‘{Array.from(ctx).map(showChar).join('')}’ came ‘{showChar(vocab[y])}’, given {fmt(p[y], 3)}. Descent raises its logit (gradient {fmtS(g[y])}) and lowers each other logit in proportion to its probability.{#if shared}{' '}Summed over the {peers.length} windows with this context, the pulls nearly cancel when the model predicts what followed: largest remaining component {fmtS(Math.max(...gsum.map(Math.abs)))}.{/if}</span>
			<span class="readout__hint">Click a character of the line to choose the prediction of the next one; drag the bracket to move the window. Switch between untrained and trained at the top right.</span>
		</p>
	{/if}
</div>

<style>
	.fx-grad :global(.m) { font-family: var(--serif); font-style: italic; fill: var(--ink); font-size: 14px; }
	.fx-grad :global(.m--cov) { fill: var(--covector-deep); }
	.fx-grad :global(.sub) { font-size: 10px; }
	.fx-grad__state { cursor: pointer; fill: var(--ink-3); font-weight: 500; }
	.fx-grad__state.is-on { fill: var(--ink); font-weight: 650; text-decoration: underline; text-underline-offset: 3px; }
	.fx-grad__onehot { fill: none; stroke: var(--ink); stroke-width: 1px; }
	.fx-grad__ch { font-size: 17px; fill: var(--ink-2); }
	.fx-grad__ch.is-y { fill: var(--ink); font-weight: 650; }
	.fx-grad__tok { font-family: var(--serif); font-size: 14px; fill: var(--ink); }
	.fx-grad__strong { fill: var(--ink); font-weight: 600; }
	.fx-grad__fwd { stroke: var(--value); stroke-width: 1.4px; }
	.fx-grad__fwdhead { fill: none; stroke: var(--value); stroke-width: 1.4px; }
	.fx-grad__bwd { stroke: var(--covector); stroke-width: 1.4px; }
	.fx-grad__bwdhead { fill: none; stroke: var(--covector); stroke-width: 1.4px; }
	.fx-grad__bwdbar { stroke: var(--covector); stroke-width: 1.2px; }
	:global([data-mode='print']) .fx-grad__state.is-on { text-decoration: none; }
</style>

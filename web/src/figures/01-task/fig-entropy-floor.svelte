<script module>
	export const meta = { size: 'wide', title: 'The floor under the loss' };
</script>

<script>
	// Figure: what can no model learn? The 80 predictions in the layout of
	// fig-training-windows (one row per window, each target in the column of its corpus
	// position). Under each target, a dot whose AREA is the smallest surprise any model
	// can have on it, −log(n_{c,y}/n_c) for its context c (the window prefix). Contexts
	// followed by one character only cost nothing and get a hairline tick. The mean of
	// all 80 is the floor (checked against the adapter's lab.floor). The trained model's
	// own surprise is drawn as a ring at the same scale, so the reader can see it sit on
	// the floor.
	import { getContext } from 'svelte';
	import { measure } from '../primitives/measure.js';
	import { fmt, showChar, layoutWidth, areaRadius } from '../primitives/scales.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const T = $derived(specimen.config.T);
	const train = $derived(specimen.lab.train);
	const B = $derived(train.x.length);
	const m = $derived(specimen.corpus.length);
	const ch = (id) => showChar(specimen.vocab[id - 1]);
	let hover = $state(null);
	let width = $state(868);
	const interactive = $derived(mode !== 'print');

	// per prediction: its context (the window prefix), the best possible surprise, the model's
	const cells = $derived.by(() => {
		const groups = new Map();
		const out = [];
		for (let b = 0; b < B; b++) for (let t = 0; t < T; t++) {
			const ctx = train.x[b].slice(0, t + 1).join(',');
			const y = train.y[b][t];
			if (!groups.has(ctx)) groups.set(ctx, new Map());
			groups.get(ctx).set(y, (groups.get(ctx).get(y) ?? 0) + 1);
			out.push({ b, t, ctx, y });
		}
		const pred = specimen.predictions(specimen.theta);
		for (const c of out) {
			const g = groups.get(c.ctx);
			const nc = [...g.values()].reduce((s, v) => s + v, 0);
			c.best = -Math.log(g.get(c.y) / nc);
			c.followers = [...g.entries()];
			const n = c.b * T + c.t;
			c.model = -Math.log(pred.probs[n][c.y - 1]);
		}
		return out;
	});
	const floor = $derived(cells.reduce((s, c) => s + c.best, 0) / cells.length);
	const agrees = $derived(Math.abs(floor - specimen.lab.floor) < 1e-12);
	const costly = $derived(cells.filter((c) => c.best > 1e-12));
	const vmax = $derived(Math.max(...cells.map((c) => Math.max(c.best, c.model))));
	const trainedLoss = $derived(specimen.loss.at(-1));

	// geometry (as fig-training-windows)
	const W = $derived(layoutWidth(width, mode, 330));
	const narrow = $derived(W < 600);
	const padL = $derived(narrow ? 30 : 86);
	const padR = $derived(narrow ? 8 : 178);
	const cw = $derived((W - padL - padR) / m);
	const xc = (k) => padL + cw * (k + 0.5);
	const rowH = $derived(narrow ? 32 : 40);
	const y0 = 34;
	const yRow = (b) => y0 + rowH * b;
	const rmax = $derived(Math.min(cw * 0.4, rowH * 0.22));
	const H = $derived(y0 + rowH * B + 46);
	const fs = $derived(narrow ? 11 : 13);
	const same = (c) => hover && c.ctx === hover.ctx;

	const readout = $derived.by(() => {
		if (!hover) return null;
		const c = hover;
		const ctx = c.ctx.split(',').map((id) => ch(+id)).join('');
		if (c.followers.length === 1) return `Window ${c.b + 1}, position ${c.t + 1}: the context “${ctx}” is always followed by ‘${ch(c.y)}’, so a model can be certain: the floor here is 0. The trained specimen pays ${fmt(c.model, 4)} nats.`;
		const list = c.followers.map(([y, k]) => `‘${ch(y)}’${k > 1 ? ` ×${k}` : ''}`).join(', ');
		return `Window ${c.b + 1}, position ${c.t + 1}: the context “${ctx}” is followed by ${list} in different windows. The best a model can do is ${fmt(Math.exp(-c.best), 2)} on each: a surprise of ${fmt(c.best, 3)} nats here. The trained specimen pays ${fmt(c.model, 3)}.`;
	});
</script>

<div class="fx fx-floor" use:measure={(w) => (width = w)}>
	<svg viewBox="0 0 {W} {H}" width={W} height={H} role={interactive ? 'group' : 'img'} aria-label="The entropy floor: of {cells.length} predictions, {costly.length} cannot be made certain; the mean of the smallest possible surprises is {fmt(floor, 4)} nats.">
		{#each cells as c (c.b * 64 + c.t)}
			{@const x = xc(c.b + c.t + 1)}
			{@const y = yRow(c.b)}
			{#if c.best > 1e-12}
				<circle class="fx-floor__best" class:is-on={same(c)} cx={x} cy={y + 5 + rmax} r={areaRadius(c.best, vmax, rmax)} />
			{:else}
				<line class="fx-floor__zero" x1={x - 2.5} x2={x + 2.5} y1={y + 5 + rmax} y2={y + 5 + rmax} />
			{/if}
			{#if areaRadius(c.model, vmax, rmax) > 0.8}
				<circle class="fx-floor__model" cx={x} cy={y + 5 + rmax} r={areaRadius(c.model, vmax, rmax)} />
			{/if}
			<text class="tok fx-floor__ch" class:is-dim={hover && !same(c)} class:is-on={same(c)} x={x} y={y} text-anchor="middle" font-size={fs}>{ch(c.y)}</text>
			{#if interactive}
				<rect class="hit" x={x - cw / 2} y={y - rowH * 0.4} width={cw} height={rowH} role="button" aria-label="read this entry; details in the live readout" tabindex="0" onpointerenter={() => (hover = c)} onfocus={() => (hover = c)} onpointerleave={() => (hover = null)} onblur={() => (hover = null)} onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); (hover = c); } }} />
			{/if}
		{/each}
		{#each train.x as _, b (b)}
			<text class="lbl lbl--micro" x={xc(b + 1) - cw * 0.85} y={yRow(b)} text-anchor="end">{narrow ? b + 1 : `window ${b + 1}`}</text>
		{/each}

		<!-- the sum -->
		{#if !narrow}
			{@const xs = W - padR + 22}
			<text class="lbl lbl--head" x={xs} y={y0 + 4}>the floor</text>
			<text class="lbl" x={xs} y={y0 + 22}>mean of the {cells.length} dots</text>
			<text class="lbl fx-floor__num" x={xs} y={y0 + 46}>{fmt(floor, 4)} nats</text>
			<text class="lbl lbl--micro" x={xs} y={y0 + 66}>{costly.length} of the {cells.length} cost something;</text>
			<text class="lbl lbl--micro" x={xs} y={y0 + 80}>every other prediction is</text>
			<text class="lbl lbl--micro" x={xs} y={y0 + 94}>fixed by its context</text>
			{#if Number.isFinite(trainedLoss)}
				<circle class="fx-floor__model" cx={xs + 4} cy={y0 + 126} r="4" />
				<text class="lbl lbl--micro" x={xs + 14} y={y0 + 130}>the trained specimen:</text>
				<text class="lbl lbl--micro" x={xs + 14} y={y0 + 144}>mean {fmt(trainedLoss, 4)} nats</text>
			{/if}
			<circle class="fx-floor__best" cx={xs + 4} cy={y0 + 168} r="4" />
			<text class="lbl lbl--micro" x={xs + 14} y={y0 + 172}>area: the least surprise</text>
			<text class="lbl lbl--micro" x={xs + 14} y={y0 + 186}>any model can have</text>
		{/if}
		{#if !agrees}<text class="lbl lbl--covector" x={padL} y={H - 6}>check failed: the floor disagrees with the adapter’s</text>{/if}
	</svg>
	{#if interactive}
		<p class="readout" aria-live="polite">
			{#if readout}<span class="readout__part">{readout}</span>{:else}<span class="readout__part">{costly.length} of the {cells.length} predictions cannot be certain; their mean least surprise is {fmt(floor, 4)} nats.</span>{/if}
			<span class="readout__hint">Point at a prediction to see its context and what follows it.</span>
		</p>
	{/if}
</div>

<style>
	.fx-floor__best { fill: var(--value); }
	.fx-floor__best.is-on { fill: var(--value-deep); }
	.fx-floor__zero { stroke: var(--rule); stroke-width: var(--hair); }
	.fx-floor__model { fill: none; stroke: var(--ink); stroke-width: 0.7px; }
	.fx-floor__ch { fill: var(--ink-2); }
	.fx-floor__ch.is-on { fill: var(--ink); font-weight: 650; }
	.is-dim { opacity: 0.4; }
	.fx-floor__num { font-size: 17px; fill: var(--ink); font-weight: 560; }
</style>

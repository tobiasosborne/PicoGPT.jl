<script module>
	export const meta = { size: 'wide', title: 'The specimen writes' };
	export { default as Key } from './fig-sampling-tree.key.svelte';
</script>

<script>
	// Figure: the specimen generating text, one character per column. Each column is the
	// distribution p = softmax(z/τ) of the next character stacked as [0, 1] in vocabulary
	// order — the cumulative distribution that the sampler inverts — with the uniform draw u
	// marked; the segment u falls in is the character drawn (solid), the others are the
	// paths not taken. The line through the chosen segments is the path through the tree.
	// Under the text: where the context was never seen in training.
	// Interactive: τ (drag; far left = τ → 0, greedy), "draw again" (new seed), play/scrub.
	// Print: τ → 0, 1 and 2, same seed.
	import { getContext } from 'svelte';
	import { measure } from '../primitives/measure.js';
	import Scrubber from '../primitives/Scrubber.svelte';
	import { gibbs, entropy } from '../primitives/EnergyLadder.svelte';
	import { fmt, fmtProb, showChar, clamp, layoutWidth } from '../primitives/scales.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const T = $derived(specimen.config.T);
	const V = $derived(specimen.config.V);
	const vocab = $derived(specimen.vocab);
	const interactive = $derived(mode !== 'print');
	let width = $state(868);

	// svelte-ignore state_referenced_locally
	const prompt = String(attrs.prompt ?? '␣').replace(/␣/g, ' '); // the source writes a space as ␣
	// svelte-ignore state_referenced_locally
	const count = +(attrs.count ?? 22);
	// svelte-ignore state_referenced_locally
	let tau = $state(attrs.tau != null ? +attrs.tau : 1);
	// svelte-ignore state_referenced_locally
	let seed = $state(+(attrs.seed ?? 1));
	// svelte-ignore state_referenced_locally
	let t = $state(attrs.t != null ? +attrs.t : 1);
	$effect(() => { if (attrs.t != null) t = +attrs.t; });

	// contexts that occur in training: every window prefix (positions are relative to the window)
	const seen = $derived.by(() => {
		const s = new Set();
		for (let b = 0; b < specimen.lab.train.x.length; b++) for (let k = 1; k <= T; k++) s.add(specimen.corpus.slice(b, b + k));
		return s;
	});
	function run(tt) {
		const out = specimen.speak(prompt, count, { temperature: tt, seed });
		return out.steps.map((st) => {
			const p = gibbs(st.z, tt);
			const ctx = st.context.map((i) => vocab[i - 1]).join('');
			return { ...st, p, ctx, seen: seen.has(ctx), H: entropy(p), F: p.reduce((acc, v) => [...acc, (acc.at(-1) ?? 0) + v], []) };
		});
	}
	const rows = $derived(interactive ? [{ tau, steps: run(tau) }] : [0, 1, 2].map((tt) => ({ tau: tt, steps: run(tt) })));
	const shown = $derived(interactive ? Math.round(t * count) : count);

	// geometry
	const W = $derived(layoutWidth(width, mode, 340));
	const narrow = $derived(W < 640);
	const L = $derived(narrow ? 30 : 64);
	const R = 8;
	const nCols = $derived(prompt.length + count);
	const cw = $derived((W - L - R) / nCols);
	const cx = (k) => L + cw * (k + 0.5);
	const colH = $derived(interactive ? (narrow ? 150 : 210) : 120);
	const bw = $derived(Math.min(16, cw * 0.56));
	const top = $derived(interactive ? 76 : 20);
	const rowH = $derived(colH + 76);
	const H = $derived(top + rows.length * rowH + 8);

	// τ track: log10 τ from −1 to 1, plus the greedy stop at the left end
	const tx0 = $derived(L + 70), tx1 = $derived(Math.min(W - 200, L + 470));
	const xT = (tt) => (tt <= 0 ? tx0 - 34 : tx0 + ((Math.log10(tt) + 1) / 2) * (tx1 - tx0));
	let svgEl = $state();
	let dragging = $state(false);
	function px(e) { const pt = svgEl.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY; return pt.matrixTransform(svgEl.getScreenCTM().inverse()).x; }
	function tDown(e) { dragging = true; e.currentTarget.setPointerCapture(e.pointerId); tMove(e); }
	function tMove(e) {
		if (!dragging) return;
		const x = px(e);
		if (x < tx0 - 17) { tau = 0; return; }
		let v = 10 ** (clamp((x - tx0) / (tx1 - tx0), 0, 1) * 2 - 1);
		if (Math.abs(Math.log(v)) < 0.06) v = 1;
		tau = v;
	}
	function tKey(e) {
		const f = e.shiftKey ? 1.6 : 1.15;
		if (e.key === 'ArrowRight' || e.key === 'ArrowUp') { tau = tau === 0 ? 0.1 : clamp(tau * f, 0.1, 10); e.preventDefault(); }
		if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') { tau = tau / f < 0.1 ? 0 : tau / f; e.preventDefault(); }
		if (e.key === 'Home' || e.key === 'Enter') { tau = 1; e.preventDefault(); }
	}
	let hover = $state(-1);
	const hv = $derived(hover >= 0 && hover < rows[0].steps.length ? rows[0].steps[hover] : null);
	const textOf = (row) => prompt + row.steps.slice(0, shown).map((s) => vocab[s.next - 1]).join('');
	const unseenFrom = (row) => row.steps.findIndex((s) => !s.seen);
	const tauLabel = (tt) => (tt === 0 ? 'τ → 0 (greedy)' : `τ = ${fmt(tt, tt < 1 ? 2 : tt < 10 ? 2 : 0)}`);
</script>

<div class="fx fx-tree" use:measure={(w) => (width = w)}>
	<svg bind:this={svgEl} viewBox="0 0 {W} {H}" width={W} height={H} role={interactive ? 'group' : 'img'} aria-label="The specimen continues ‘{prompt}’ at {tauLabel(rows[0].tau)}: ‘{textOf(rows[0])}’">
		{#if interactive}
			<!-- temperature -->
			<text class="lbl lbl--head" x={L - (narrow ? 26 : 0)} y="30">temperature</text>
			<line class="axis" x1={tx0} x2={tx1} y1="30" y2="30" />
			<line class="axis" x1={tx0 - 34} x2={tx0 - 34} y1="26" y2="34" />
			<text class="lbl lbl--micro" x={tx0 - 34} y="48" text-anchor="middle">→ 0</text>
			{#each [0.1, 1, 10] as v (v)}
				<line class="axis" x1={xT(v)} x2={xT(v)} y1="26" y2="34" />
				<text class="lbl lbl--micro" x={xT(v)} y="48" text-anchor="middle">{v}</text>
			{/each}
			<g transform="translate({xT(tau)} 30)">
				<circle class="handle" class:handle--active={dragging} r="6.5" />
				<text class="lbl fx-tree__tl" y="-12" text-anchor="middle">{tauLabel(tau)}</text>
			</g>
			<rect class="handle-hit" x={tx0 - 46} y="14" width={tx1 - tx0 + 56} height="32" role="slider" tabindex="0" aria-label="sampling temperature τ" aria-valuenow={tau} aria-valuetext={tauLabel(tau)} onpointerdown={tDown} onpointermove={tMove} onpointerup={() => (dragging = false)} onkeydown={tKey} ondblclick={() => (tau = 1)} />
			<text class="lbl fx-tree__seed" x={W - R} y="30" text-anchor="end" role="button" tabindex="0" onclick={() => (seed += 1)} onkeydown={(e) => (e.key === 'Enter' || e.key === ' ') && (seed += 1)}>↻ draw again <tspan class="fx-tree__dim">(seed {seed})</tspan></text>
		{/if}

		{#each rows as row, r (r)}
			{@const y0 = top + r * rowH}
			{@const yText = y0 + colH + 30}
			{@const uf = unseenFrom(row)}
			<g>
				{#if !interactive}<text class="lbl lbl--head" x={0} y={y0 + 10}>{tauLabel(row.tau)}</text>{/if}
				<!-- the prompt -->
				{#each Array.from(prompt) as ch, k (k)}
					<text class="tok fx-tree__prompt" x={cx(k)} y={yText} text-anchor="middle">{showChar(ch)}</text>
				{/each}
				{#if !narrow}<text class="lbl lbl--micro" x={cx(0)} y={yText + 18} text-anchor="middle">prompt</text>{/if}
				<!-- the path through the chosen segments -->
				<path class="fx-tree__path" d={'M' + row.steps.slice(0, shown).map((s, k) => {
					const a = s.next - 1, mid = ((s.F[a] - s.p[a] / 2) * colH);
					return `${cx(prompt.length + k)} ${y0 + mid}`;
				}).join('L')} />
				{#each row.steps.slice(0, shown) as s, k (k)}
					{@const x = cx(prompt.length + k)}
					{@const a0 = s.next - 1}
					<g class="fx-tree__col" class:is-on={interactive && k === hover} class:is-unseen={!s.seen}>
						{#each s.p as pa, a (a)}
							{@const yA = y0 + (s.F[a] - pa) * colH}
							{#if pa * colH >= 0.4}
								<rect class={a === a0 ? 'bar--value' : 'fx-tree__alt'} x={x - bw / 2} y={yA} width={bw} height={pa * colH} />
								{#if pa * colH >= 13}
									<text class="tok fx-tree__seg" class:is-chosen={a === a0} x={x} y={yA + (pa * colH) / 2 + 4.5} text-anchor="middle">{showChar(vocab[a])}</text>
								{/if}
							{/if}
						{/each}
						{#if s.u != null}
							<line class="fx-tree__u" x1={x - bw / 2 - 4} x2={x + bw / 2 + 4} y1={y0 + s.u * colH} y2={y0 + s.u * colH} />
						{/if}
						<text class="tok fx-tree__ch" x={x} y={yText} text-anchor="middle">{showChar(vocab[a0])}</text>
						{#if interactive}
							<rect class="hit" x={x - cw / 2} y={y0 - 4} width={cw} height={colH + 44} role="button" tabindex="0" aria-label="step {k + 1}" onpointerenter={() => (hover = k)} onfocus={() => (hover = k)} onpointerleave={() => (hover = -1)} onblur={() => (hover = -1)} onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); (hover = k); } }} />
						{/if}
					</g>
				{/each}
				<!-- where the contexts leave the training set -->
				{#if uf >= 0 && uf < shown}
					<path class="fx-tree__unseen" d="M{cx(prompt.length + uf) - cw / 2 + 2} {yText + 10} v5 H{cx(prompt.length + shown - 1) + cw / 2 - 2} v-5" />
					<text class="lbl lbl--micro" x={cx(prompt.length + uf) - cw / 2 + 2} y={yText + 30}>{narrow ? 'contexts never seen in training' : 'from here on, contexts the specimen never saw in training'}</text>
				{/if}
				{#if !narrow}
					<text class="lbl lbl--micro" x={L - 10} y={y0 + 4} text-anchor="end">0</text>
					<text class="lbl lbl--micro" x={L - 10} y={y0 + colH + 4} text-anchor="end">1</text>
					<line class="axis" x1={L - 6} x2={L - 6} y1={y0} y2={y0 + colH} />
				{/if}
			</g>
		{/each}
	</svg>
	{#if interactive}
		<Scrubber bind:t phases={[{ at: 0, label: 'prompt' }, { at: 1, label: `${count} characters` }]} label="writing, one character at a time" duration={count * 260} />
		<p class="readout" aria-live="polite">
			{#if hv}
				{@const order = hv.p.map((v, a) => ({ v, a })).sort((u, w) => w.v - u.v).filter((o) => o.v >= 0.005)}
				<span class="readout__who">Step {hover + 1}, after ‘{Array.from(hv.ctx).map(showChar).join('')}’{hv.seen ? '' : ' (a context never seen in training)'}.</span>
				<span class="readout__part">{order.map((o) => `‘${showChar(vocab[o.a])}’ ${fmtProb(o.v, 2, true)}`).join(', ')}{order.length < V ? ', the rest below 0.01' : ''}; entropy {fmt(hv.H, 2)} nats.{#if hv.u != null}{' '}The draw u = {fmt(hv.u, 3)} falls in ‘{showChar(vocab[hv.next - 1])}’.{:else}{' '}At τ → 0 there is no draw: the likeliest character is taken.{/if}</span>
			{:else}
				<span class="readout__who">{tauLabel(rows[0].tau)}:</span>
				<span class="readout__part fx-tree__out">‘{Array.from(textOf(rows[0])).map(showChar).join('')}’</span>
				<span class="readout__hint">Drag the temperature; point at a column for its numbers; “draw again” uses new random numbers u. At the far left, τ → 0: always the likeliest character.</span>
			{/if}
		</p>
	{/if}
</div>

<style>
	.fx-tree__prompt { font-size: 18px; fill: var(--ink); font-weight: 650; }
	.fx-tree__ch { font-size: 18px; fill: var(--ink); }
	.fx-tree__col.is-unseen .fx-tree__ch { fill: var(--ink-2); font-style: italic; }
	.fx-tree__alt { fill: none; stroke: var(--ink-3); stroke-width: var(--hair); }
	.fx-tree__seg { font-size: 12px; fill: var(--ink-2); }
	.fx-tree__seg.is-chosen { fill: var(--paper); }
	.fx-tree__u { stroke: var(--ink); stroke-width: 1.5px; }
	.fx-tree__path { fill: none; stroke: var(--ink); stroke-width: var(--hair); }
	.fx-tree__col.is-on .bar--value { fill: var(--value-deep); }
	.fx-tree__unseen { fill: none; stroke: var(--ink-3); stroke-width: var(--hair); }
	.fx-tree__tl { fill: var(--ink); }
	.fx-tree__seed { cursor: pointer; fill: var(--value-deep); }
	.fx-tree__seed:hover { text-decoration: underline; }
	.fx-tree__dim { fill: var(--ink-3); }
	.fx-tree__out { font-family: var(--serif); font-size: 1.05em; color: var(--ink); }
</style>

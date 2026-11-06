<script module>
	export const meta = { size: 'wide', title: 'Do attention heads relay J-lens directions?' };
</script>

<script>
	// Figure: each attention head's OV map W_o^(h) W_v^(h) tested on populations of unit
	// directions. x: gain, the mean ‖OV v‖ over the population divided by the mean over 2000
	// random unit directions (random = 1). y: label preservation, the mean reciprocal rank of
	// cos(OV v_i, v_i) among cos(OV v_i, v_j), minus the same for V random directions (random
	// = 0). Populations: J-lens vectors at the head's input depth (covector-ink bar, labelled
	// block.head); the same vectors under one fixed random rotation, which keeps their lengths
	// and angles and changes only their orientation to the weights (open ring, joined to its
	// head's J mark); the rows of W_out (small open square). Both models, own panels, one
	// scale. Numbers: both exports' structure.attention_ov.
	import { getContext } from 'svelte';
	import { measure } from '#figures/primitives/measure.js';
	import { fmt, layoutWidth } from '#figures/primitives/scales.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const art = $derived(getContext('artifacts')()?.jspace ?? {});
	const interactive = $derived(mode !== 'print');
	let width = $state(868);
	let focus = $state(null); // {p, k}

	const panels = $derived(
		[
			{ key: 'specimen', name: 'the specimen', heads: art.specimen?.structure.attention_ov ?? [] },
			{ key: 'shakespeare', name: 'the Shakespeare model', heads: art.shakespeare?.structure.attention_ov ?? [] }
		].filter((p) => p.heads.length)
	);
	const POPS = ['j_lens', 'rotated_j_lens', 'logit_lens'];
	const all = $derived(panels.flatMap((p) => p.heads.flatMap((h) => POPS.map((k) => h.populations[k]))));
	const gx = $derived([Math.min(0.75, Math.floor(Math.min(...all.map((q) => q.gain)) * 10) / 10), Math.ceil(Math.max(...all.map((q) => q.gain)) * 10) / 10]);
	const gy = $derived([Math.min(-0.1, Math.floor(Math.min(...all.map((q) => q.label_preservation)) * 20) / 20), Math.ceil(Math.max(...all.map((q) => q.label_preservation)) * 20) / 20]);

	const W = $derived(layoutWidth(width, mode, 340));
	const narrow = $derived(W < 620);
	const padL = 46, gap = 54;
	const pw = $derived(narrow ? W - padL - 10 : (W - padL - 10 - gap) / Math.max(1, panels.length));
	const ph = $derived(narrow ? 230 : 300);
	const px = (i) => (narrow ? padL : padL + i * (pw + gap));
	const py = (i) => 40 + (narrow ? i * (ph + 80) : 0);
	const X = (i, v) => px(i) + ((v - gx[0]) / (gx[1] - gx[0])) * pw;
	const Y = (i, v) => py(i) + ph - ((v - gy[0]) / (gy[1] - gy[0])) * ph;
	const H = $derived(py(narrow ? panels.length - 1 : 0) + ph + 56);
	const xt = $derived(Array.from({ length: Math.floor((gx[1] - gx[0]) / 0.25 + 1e-9) + 1 }, (_, k) => Math.ceil(gx[0] * 4) / 4 + 0.25 * k).filter((t) => t <= gx[1] + 1e-9));
	const yt = $derived(Array.from({ length: 20 }, (_, k) => Math.ceil(gy[0] * 10) / 10 + 0.1 * k).filter((t) => t <= gy[1] + 1e-9));

	// head labels, nudged so that they do not overlap
	function labels(i, heads) {
		const placed = [];
		return heads.map((h) => {
			const q = h.populations.j_lens;
			let x = X(i, q.gain) + 9, y = Y(i, q.label_preservation) + 3.5;
			for (let n = 0; n < 12 && placed.some((p) => Math.abs(p.x - x) < 20 && Math.abs(p.y - y) < 10); n++) y += 10;
			placed.push({ x, y });
			return { x, y, text: `${h.block}.${h.head}` };
		});
	}
	const readout = $derived.by(() => {
		if (!focus) return null;
		const h = panels[focus.p].heads[focus.k], P = h.populations;
		return `${panels[focus.p].name}, block ${h.block}, head ${h.head}: J-lens vectors gain ${fmt(P.j_lens.gain, 2)}, label preservation ${fmt(P.j_lens.label_preservation, 3)}; rotated ${fmt(P.rotated_j_lens.gain, 2)}, ${fmt(P.rotated_j_lens.label_preservation, 3)}; rows of W_out ${fmt(P.logit_lens.gain, 2)}, ${fmt(P.logit_lens.label_preservation, 3)}.`;
	});
</script>

<div class="fx fx-heads" use:measure={(w) => (width = w)}>
	{#if !panels.length}
		<p class="fig__pending">The J-lens exports are missing.</p>
	{:else}
		<svg viewBox="0 0 {W} {H}" width={W} height={H} role={interactive ? 'group' : 'img'} aria-label="OV gain and label preservation of every attention head. {panels.map((p) => `${p.name}: ${p.heads.map((h) => `${h.block}.${h.head} J ${fmt(h.populations.j_lens.label_preservation, 2)}, rotated ${fmt(h.populations.rotated_j_lens.label_preservation, 2)}`).join('; ')}`).join('. ')}">
			<!-- legend -->
			<g transform="translate({padL} 12)">
				<rect class="hd__bar" x="0" y="-1.6" width="14" height="3.2" /><text class="lbl lbl--micro" x="20" y="3">J-lens vectors (block.head)</text>
				<circle class="hd__ring" cx="168" cy="0" r="3.4" /><text class="lbl lbl--micro" x="176" y="3">the same, rotated</text>
				<rect class="hd__sq" x={narrow ? 166 : 284} y={narrow ? 9 : -3} width="6" height="6" /><text class="lbl lbl--micro" x={narrow ? 176 : 294} y={narrow ? 15 : 3}>rows of W<tspan dy="2" font-size="8">out</tspan></text>
			</g>
			{#each panels as p, i (p.key)}
				<text class="lbl lbl--head" x={px(i)} y={py(i) - 8}>{p.name}</text>
				{#each xt as t (t)}
					<line class={Math.abs(t - 1) < 1e-9 ? 'axis' : 'rule'} x1={X(i, t)} x2={X(i, t)} y1={py(i)} y2={py(i) + ph} />
					<text class="lbl lbl--micro" x={X(i, t)} y={py(i) + ph + 14} text-anchor="middle">{fmt(t, 2)}</text>
				{/each}
				{#each yt as t (t)}
					<line class={Math.abs(t) < 1e-9 ? 'axis' : 'rule'} x1={px(i)} x2={px(i) + pw} y1={Y(i, t)} y2={Y(i, t)} />
					<text class="lbl lbl--micro" x={px(i) - 6} y={Y(i, t) + 3} text-anchor="end">{fmt(t, 1)}</text>
				{/each}
				<text class="lbl lbl--micro" x={px(i) + pw} y={py(i) + ph + 30} text-anchor="end">gain (random = 1) →</text>
				<text class="lbl lbl--micro" transform="translate({px(i) - 34} {py(i) + ph / 2}) rotate(-90)" text-anchor="middle">label preservation (random = 0) →</text>
				{#each p.heads as h, k (k)}
					{@const J = h.populations.j_lens}
					{@const R = h.populations.rotated_j_lens}
					{@const O = h.populations.logit_lens}
					<g class="hd__head" class:is-dim={focus && (focus.p !== i || focus.k !== k)}>
						<line class="hd__pair" x1={X(i, J.gain)} y1={Y(i, J.label_preservation)} x2={X(i, R.gain)} y2={Y(i, R.label_preservation)} />
						<rect class="hd__sq" x={X(i, O.gain) - 3} y={Y(i, O.label_preservation) - 3} width="6" height="6" />
						<circle class="hd__ring" cx={X(i, R.gain)} cy={Y(i, R.label_preservation)} r="3.4" />
						<rect class="hd__bar" x={X(i, J.gain) - 7} y={Y(i, J.label_preservation) - 1.6} width="14" height="3.2" />
					</g>
				{/each}
				{#each labels(i, p.heads) as l, k (k)}
					<text class="lbl lbl--micro lbl--covector" x={l.x} y={l.y}>{l.text}</text>
				{/each}
				{#if interactive}
					{#each p.heads as h, k (k)}
						<circle class="hit" cx={X(i, h.populations.j_lens.gain)} cy={Y(i, h.populations.j_lens.label_preservation)} r="9" role="button" tabindex="-1" aria-label="block {h.block}, head {h.head}"
							onpointerenter={() => (focus = { p: i, k })} onpointerleave={() => (focus = null)} onkeydown={() => {}} />
					{/each}
				{/if}
			{/each}
		</svg>
		{#if interactive}
			<p class="readout" aria-live="polite">
				{#if readout}<span class="readout__part">{readout}</span>{:else}<span class="readout__hint">Point at a head’s J-lens mark for its numbers.</span>{/if}
			</p>
		{/if}
	{/if}
</div>

<style>
	.hd__bar { fill: var(--covector); }
	.hd__ring { fill: var(--paper); stroke: var(--ink-2); stroke-width: 1px; }
	.hd__sq { fill: none; stroke: var(--ink-3); stroke-width: 0.9px; }
	.hd__pair { stroke: var(--ink-3); stroke-width: var(--hair); }
	.hd__head { transition: opacity var(--t-quick); }
	.hd__head.is-dim { opacity: 0.3; }
</style>

<script module>
	export const meta = { size: 'wide', title: 'The embedding matrix during training' };
</script>

<script>
	// Figure: what does E look like, and how did it get there?
	// Two Hinton diagrams of E ∈ ℝ^{D×V} — at step 0 and at the scrubbed step — and the V
	// columns as spoke glyphs on one shared scale, each over the faint glyph of its start.
	// The clock t ∈ [0, 1] is the training step, t·steps. θ(step) comes from the adapter's
	// replay of the recorded run (specimen.trajectory(), verified against the artifact's
	// loss to 1e-9); steps 0 and the last are θ₀ and θ themselves, so the static state
	// needs no replay. Print: the final step, with step 0 beside it.
	import { getContext } from 'svelte';
	import { measure } from '../primitives/measure.js';
	import MatrixGlyph from '../primitives/MatrixGlyph.svelte';
	import SpokeGlyph from '../primitives/SpokeGlyph.svelte';
	import Scrubber from '../primitives/Scrubber.svelte';
	import { fmt, showChar, maxAbs, spokeScale, layoutWidth } from '../primitives/scales.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const specimen = $derived(getContext('specimen')());
	const V = $derived(specimen.config.V);
	const D = $derived(specimen.config.D);
	const steps = $derived(specimen.provenance?.steps ?? Math.max(0, specimen.loss.length - 1));
	const off = $derived(specimen.layout.find((l) => l.name === 'E').offset);
	const interactive = $derived(mode !== 'print');

	// svelte-ignore state_referenced_locally
	let t = $state(attrs.t != null ? +attrs.t : 1);
	$effect(() => { if (attrs.t != null) t = +attrs.t; });
	let width = $state(868);
	let hover = $state(null);

	const step = $derived(Math.round(t * steps));
	let traj = $state(null); // replayed lazily, only when an intermediate step is asked for
	const needReplay = $derived(step > 0 && step < steps);
	let replaying = false;
	$effect(() => {
		if (!needReplay || traj !== null || replaying) return;
		replaying = true; // let the readout say so, then replay (a few hundred ms, once)
		setTimeout(() => { traj = specimen.trajectory() ?? false; replaying = false; }, 30);
	});
	const thetaAt = (k) => (k <= 0 ? specimen.theta0 : k >= steps ? specimen.theta : traj ? traj.thetas[k] : null);
	const sliceE = (th) => Array.from({ length: D * V }, (_, k) => th[off + k]);
	const E0 = $derived(sliceE(specimen.theta0 ?? specimen.theta));
	const Et = $derived.by(() => {
		const th = thetaAt(step);
		return th ? sliceE(th) : null;
	});
	const Eshow = $derived(Et ?? E0);
	const shownStep = $derived(Et ? step : 0);
	const lossAt = $derived(specimen.loss?.[shownStep]);

	// one scale for every matrix and glyph at every step: the largest |E| over the run's ends
	const vmax = $derived(maxAbs(E0, sliceE(specimen.theta)));
	const col = (E, a) => Array.from({ length: D }, (_, i) => E[i + D * a]);

	// geometry
	const W = $derived(layoutWidth(width, mode, 330));
	const narrow = $derived(W < 620);
	const cell = $derived(narrow ? Math.min(19, (W - 60) / (2 * V + 2)) : 24);
	const mW = $derived(cell * V);
	const mH = $derived(cell * D);
	const gapM = $derived(narrow ? 26 : 74);
	const mx0 = $derived((W - 2 * mW - gapM) / 2);
	const mx1 = $derived(mx0 + mW + gapM);
	const my = 44;
	const stepG = $derived((W - 20) / V);
	const rG = $derived(Math.min(narrow ? 22 : 40, stepG * 0.42));
	const gy = $derived(my + mH + (narrow ? 64 : 84) + rG);
	const gxs = $derived(Array.from({ length: V }, (_, a) => 10 + stepG * (a + 0.5)));
	const sc = $derived(spokeScale(rG, vmax));
	const H = $derived(gy + rG + 40);

	const PHASES = $derived([
		{ at: 0, label: '0' },
		...(specimen.provenance?.hyper?.warmup ? [{ at: specimen.provenance.hyper.warmup / steps, label: 'warmup ends' }] : []),
		{ at: 0.5, label: String(Math.round(steps / 2)) },
		{ at: 1, label: String(steps) }
	]);

	const moved = $derived.by(() => {
		let best = { d: 0, i: 0, a: 0 };
		for (let a = 0; a < V; a++) for (let i = 0; i < D; i++) {
			const d = Math.abs(Eshow[i + D * a] - E0[i + D * a]);
			if (d > best.d) best = { d, i, a };
		}
		const rms = Math.sqrt(Eshow.reduce((s, v, k) => s + (v - E0[k]) ** 2, 0) / Eshow.length);
		const size = Math.sqrt(E0.reduce((s, v) => s + v * v, 0) / E0.length);
		return { ...best, rms, size };
	});
	const summary = $derived(`${Number.isFinite(lossAt) ? `Loss ${fmt(lossAt, 3)} nats. ` : ''}The entries of E have moved by ${fmt(moved.rms, 2)} (root mean square) from their start, against a typical size of ${fmt(moved.size, 2)}; the largest change is in E[${moved.i + 1}, ${moved.a + 1}], by ${fmt(moved.d, 2)}.`);
	const label = $derived(`At step ${shownStep} of ${steps}. ${summary}`);
</script>

<div class="fx fx-emb" use:measure={(w) => (width = w)}>
	<svg viewBox="0 0 {W} {H}" width={W} height={H} role={interactive ? 'group' : 'img'} aria-label="The embedding matrix. {label}">
		{#each [[mx0, E0, 0], [mx1, Eshow, shownStep]] as [mx, E, k], m (m)}
			<text class="lbl lbl--head" x={mx} y={my - 26}>step {k}</text>
			{#if m === 1 && !narrow}<text class="lbl lbl--micro" x={mx + mW} y={my - 26} text-anchor="end">{k === steps ? 'trained' : 'training'}</text>{/if}
			{#if m === 0 && !narrow}<text class="lbl lbl--micro" x={mx + mW} y={my - 26} text-anchor="end">drawn from N(0, 1)</text>{/if}
			{#each specimen.vocab as ch, a (a)}
				<text class="tok fx-emb__char" x={mx + cell * (a + 0.5)} y={my - 7} text-anchor="middle">{showChar(ch)}</text>
			{/each}
			<MatrixGlyph get={(i, j) => E[i + D * j]} m={D} n={V} x={mx} y={my} {cell} {vmax} grid />
			{#if interactive && m === 1}
				{#each Array.from({ length: D * V }) as _, q (q)}
					<rect class="hit" x={mx + cell * Math.floor(q / D)} y={my + cell * (q % D)} width={cell} height={cell} role="button" aria-label="read this entry; details in the live readout" tabindex="0" onpointerenter={() => (hover = { i: q % D, a: Math.floor(q / D) })} onfocus={() => (hover = { i: q % D, a: Math.floor(q / D) })} onpointerleave={() => (hover = null)} onblur={() => (hover = null)} onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); (hover = { i: q % D, a: Math.floor(q / D) }); } }} />
				{/each}
			{/if}
		{/each}
		{#each Array.from({ length: D }) as _, i (i)}
			<text class="lbl lbl--micro" x={mx0 - 7} y={my + cell * (i + 0.5) + 3.5} text-anchor="end">{i + 1}</text>
		{/each}
		<text class="lbl fx-emb__arrow" x={mx0 + mW + gapM / 2} y={my + mH / 2 + 4} text-anchor="middle">→</text>

		<!-- leaders: each column of the trained matrix to its glyph -->
		{#each gxs as gx, a (a)}
			<line class="leader fx-emb__fan" x1={mx1 + cell * (a + 0.5)} y1={my + mH + 6} x2={gx} y2={gy - rG - 14} />
		{/each}
		<!-- the columns as spoke glyphs: faint at step 0, solid at the shown step -->
		{#each gxs as gx, a (a)}
			<SpokeGlyph values={col(E0, a)} x={gx} y={gy} rmax={sc.rmax} scale={sc.scale} muted compass={false} />
			<SpokeGlyph values={col(Eshow, a)} x={gx} y={gy} rmax={sc.rmax} scale={sc.scale} weight={1.25} tip={narrow ? 1.8 : 2.4} />
			<text class="tok fx-emb__gchar" x={gx} y={gy + rG + 24} text-anchor="middle">{showChar(specimen.vocab[a])}</text>
		{/each}
	</svg>
	{#if interactive && steps > 0}
		<Scrubber bind:t phases={PHASES} duration={3600} label="training step" />
		<p class="readout" aria-live="polite">
			<span class="readout__who">Step {shownStep}{needReplay && traj === null ? ' (replaying the run…)' : ''}{traj === false ? ' (the run could not be replayed)' : ''}.</span>
			{#if hover}
				<span class="readout__part">E[{hover.i + 1}, {hover.a + 1}] (‘{showChar(specimen.vocab[hover.a])}’, component {hover.i + 1}) = {fmt(Eshow[hover.i + D * hover.a], 3)}, from {fmt(E0[hover.i + D * hover.a], 3)} at step 0.</span>
			{:else}
				<span class="readout__part">{summary}</span>
			{/if}
			<span class="readout__hint">Drag the playhead, or press play to train. Point at a square of the right-hand matrix.</span>
		</p>
	{/if}
</div>

<style>
	.fx-emb__char { font-size: 15px; }
	.fx-emb__gchar { font-size: 19px; }
	.fx-emb__arrow { font-size: 18px; fill: var(--ink-3); }
	.fx-emb__fan { stroke: var(--rule); }
</style>

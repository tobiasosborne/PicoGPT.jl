<script>
	// How to read a loss curve: a small instance of LossChart drawn from real losses
	// (pass the run's values), with leaders to the curve, the two reference lines and the
	// step axis. The geometry repeats LossChart's compact layout so the anchors land on
	// the marks.
	import HowToRead from './HowToRead.svelte';
	import LossChart from './LossChart.svelte';
	import { fmt } from './scales.js';

	let { values = [], lnV, floor, xMax = 200 } = $props();

	const cw = 132, ch = 150, cx = 4, cy = 18;
	const L = 26, Tp = 12, B = 20, R = 6;
	const pw = cw - L - R, ph = ch - Tp - B;
	const top = $derived(Math.ceil(Math.max(lnV ?? 0, ...values.filter(Number.isFinite)) * 2 + 0.25) / 2);
	const X = (k) => cx + L + (pw * Math.log1p(k)) / Math.log1p(xMax);
	const Y = (v) => cy + Tp + ph * (1 - v / top);
	// a point on the steep part of the curve: the first step below half of ln V
	const kc = $derived(Math.max(1, values.findIndex((v) => v < (lnV ?? 2) / 2)));
	const notes = $derived([
		{ ax: X(kc) + 1, ay: Y(values[kc] ?? 1), side: 'right', len: cx + cw - X(kc) + 4, text: 'the loss after k\nsteps: the mean\nsurprise, in nats' },
		{ ax: X(xMax) - 2, ay: Y(lnV), side: 'right', len: 10, text: `uniform guessing,\nln V = ${fmt(lnV, 2)}` },
		{ ax: X(xMax) - 2, ay: Y(floor), side: 'right', len: 10, text: `entropy floor,\n${fmt(floor, 4)}: the\ncorpus’s ambiguity` },
		{ ax: X(10), ay: cy + Tp + ph + 2, side: 'below', len: 22, text: 'steps, on a log scale:\n0, 1, 10, 100, …', align: 'start', dx: -40 }
	]);
</script>

<HowToRead width={248} height={214} {notes}>
	{#snippet glyph()}
		<g class="fx">
			<LossChart x={cx} y={cy} width={cw} height={ch} compact series={[{ values }]} {xMax} {lnV} {floor} xLabel="" yLabel="nats" refLabels={false} />
		</g>
	{/snippet}
</HowToRead>

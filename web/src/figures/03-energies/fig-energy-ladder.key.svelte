<script>
	// How to read the energy ladder: the figure's own context at τ = 1, with leaders to a
	// rung, a bar, the τ-ruler and the free energy F. Real numbers throughout.
	import { getContext } from 'svelte';
	import HowToRead from '../primitives/HowToRead.svelte';
	import EnergyLadder, { gibbs, freeEnergy } from '../primitives/EnergyLadder.svelte';
	import { fmt, showChar, clamp } from '../primitives/scales.js';

	let { attrs = {} } = $props();
	const specimen = $derived(getContext('specimen')());
	const start = $derived(clamp((+attrs.window || 3) - 1, 0, specimen.windows.length - 1));
	const pos = $derived(clamp((+attrs.position || 1) - 1, 0, specimen.config.T - 1));
	const z = $derived(specimen.windowTrace(start).Z[pos]);
	const p = $derived(gibbs(z, 1));
	const E = $derived(z.map((v) => -v));
	const E0 = $derived(Math.min(...E));
	const F = $derived(freeEnergy(z, 1));
	const Emin = $derived(Math.floor(Math.min(F, E0) - 1));
	const Emax = $derived(Math.ceil(Math.max(...E) + 1));
	const x = 112, y = 16, h = 220, lead = 24, pScale = 60;
	const xBar = x + lead + 14;
	const yOf = (e) => y + ((Emax - e) / (Emax - Emin)) * h;
	const best = $derived(p.indexOf(Math.max(...p)));
	const topRung = $derived(E.indexOf(Math.max(...E)));
	const gapTop = $derived(E[topRung] - E0);
	// e^gap as a plain number, two significant figures, thousands grouped by thin spaces
	const times = (g) => {
		const v = Math.exp(g);
		if (v > 1e12) return `e^${fmt(g, 1)}`;
		const k = Math.max(0, Math.floor(Math.log10(v)) - 1), r = Math.round(v / 10 ** k) * 10 ** k;
		return r.toLocaleString('en').replace(/,/g, '\u2009');
	};
</script>

<HowToRead width={246} height={292} notes={[
	{ ax: x + 6, ay: yOf(E[topRung]), side: 'left', len: 22, text: `‘${showChar(specimen.vocab[topRung])}’: ${fmt(gapTop, 1)} above the\nlowest rung, so about\n${times(gapTop)} times rarer` },
	{ ax: x - 12, ay: (yOf(E0) + yOf(E0 + 1)) / 2 - 1, side: 'left', len: 14, text: 'τ: each τ higher,\ne times rarer' },
	{ ax: x - 4, ay: yOf(F), side: 'left', len: 28, text: 'F, the free energy' },
	{ ax: xBar + p[best] * pScale * 0.6, ay: yOf(E[best]) + 4, side: 'below', len: Math.max(16, yOf(Emin) - yOf(E[best]) + 4), text: 'bar ∝ probability p;\nthe hairline is p = 1', align: 'start', dx: -46 }
]}>
	{#snippet glyph()}
		<g class="fx">
			<EnergyLadder {z} labels={specimen.vocab} tau={1} {x} {y} height={h} {Emin} {Emax} {pScale} {lead} gap={17} small showF pLabels="big" />
		</g>
	{/snippet}
</HowToRead>

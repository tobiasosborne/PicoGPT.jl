<script>
	// How to read a lens grid: a 3 × 3 excerpt of the figure's own J-lens grid, enlarged, with
	// leaders. The excerpt is chosen from the data: three consecutive letters of the text at
	// three consecutive depths whose cells are all letters, with the widest spread of
	// probabilities, so that the weight of type is easy to see. The notes quote its cells.
	import { getContext } from 'svelte';
	import HowToRead from '#figures/primitives/HowToRead.svelte';
	import { fmtProb } from '#figures/primitives/scales.js';
	import { gridChar, weightOf } from '../primitives/LensGrid.svelte';
	import { at, depthLabel } from '../primitives/jdata.js';

	const sh = $derived(getContext('artifacts')()?.jspace?.shakespeare ?? null);
	const pr = $derived(sh?.lenses.prompt ?? null);
	const vocab = $derived(sh?.vocab ?? []);
	const chars = $derived(Array.from(pr?.text ?? ''));
	const J = $derived(sh?.depth.length ?? 0);
	const cell = (t, j) => ({ ch: vocab[at(pr.ids, 0, t, j, 0) - 1], p: at(pr.probabilities, 0, t, j, 0) });
	const letter = (c) => /^[A-Za-z]$/.test(c ?? '');
	const pick = $derived.by(() => {
		let best = null;
		for (let t = 0; t + 2 < chars.length; t++) {
			if (![0, 1, 2].every((k) => letter(chars[t + k]))) continue;
			for (let j = 0; j + 2 < J; j++) {
				const cs = [0, 1, 2].flatMap((k) => [0, 1, 2].map((r) => cell(t + k, j + r)));
				if (!cs.every((c) => letter(c.ch))) continue;
				const ps = cs.map((c) => c.p), spread = Math.max(...ps) - Math.min(...ps);
				if (!best || spread > best.spread) best = { t, j, spread };
			}
		}
		return best ?? { t: 0, j: 0, spread: 0 };
	});
	const ts = $derived([0, 1, 2].map((k) => pick.t + k));
	const js = $derived([0, 1, 2].map((r) => pick.j + r));
	const X0 = 86, DX = 42, Y0 = 96, DY = 40, FS = 28;
	const cells = $derived(ts.flatMap((t, k) => js.map((j, r) => ({ ...cell(t, j), k, r }))));
	const hi = $derived(cells.reduce((m, c) => (c.p > m.p ? c : m), cells[0]));
	const lo = $derived(cells.reduce((m, c) => (c.p < m.p ? c : m), cells[0]));
	const cx = (k) => X0 + DX * k, cy = (r) => Y0 + DY * (r + 1);
</script>

{#if pr}
	<HowToRead width={250} height={300} title="How to read a lens grid" notes={[
		{ ax: cx(2) + 14, ay: Y0 - 8, side: 'right', len: Math.max(6, 192 - cx(2) - 14), text: 'the\ntext' },
		{ ax: X0 - 40, ay: cy(0) - 13, side: 'above', len: 50, text: 'columns: positions;\nrows: depths, forward\nruns down', align: 'start', dx: 6 - (X0 - 40) },
		{ ax: cx(hi.k) + 13, ay: cy(hi.r) - 9, side: 'right', len: Math.max(6, 192 - cx(hi.k) - 13), text: `‘${gridChar(hi.ch)}’ with\n${fmtProb(hi.p)}:\nbold` },
		{ ax: cx(lo.k), ay: cy(lo.r) + 6, side: 'below', len: Math.max(10, 250 - cy(lo.r)), text: `‘${gridChar(lo.ch)}’ with ${fmtProb(lo.p)}: light. Each cell:\nthe lens’s first choice, weight ∝ p`, align: 'middle', dx: 118 - cx(lo.k) }
	]}>
		{#snippet glyph()}
			<g class="fx">
				{#each ts as t, k (t)}
					<text class="tok" x={cx(k)} y={Y0} text-anchor="middle" font-size={FS * 0.78} style="fill: var(--ink-2)">{gridChar(chars[t])}</text>
				{/each}
				{#each cells as c (c.k * 3 + c.r)}
					<text class="tok" x={cx(c.k)} y={cy(c.r)} text-anchor="middle" font-size={FS} font-weight={weightOf(c.p)}>{gridChar(c.ch)}</text>
				{/each}
				{#each js as j, r (j)}
					<text class="lbl lbl--micro" x={X0 - 34} y={cy(r) - 7} text-anchor="end">{depthLabel(sh.depth[j])}</text>
				{/each}
				<line class="rule" x1={X0 - 22} x2={cx(2) + 18} y1={Y0 + 10} y2={Y0 + 10} />
			</g>
		{/snippet}
	</HowToRead>
{/if}

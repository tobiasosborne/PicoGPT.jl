<script module>
	export const meta = { size: 'wide', title: 'The mixture against its mean' };
	export { default as Key } from './fig-attention-meanfield.key.svelte';
</script>

<script>
	// Figure: does attention predict like a mixture of its sources' predictions, or like
	// their geometric mixture? One reader of one head of the trained specimen
	// (attention-audition/artifacts/mixture.json), the other head held at its usual output:
	//   top     the reader's attention row as arcs; under each source j the prediction p_j
	//           with this head's output replaced by v_j (small multiples, one scale)
	//   middle  four columns on one baseline: the arithmetic mixture Σ A_ij p_j as a stack of
	//           the sources' shares; the stack of weighted surprises A_ij(−ln p_j), whose
	//           e^{−height}, renormalised, is the geometric mixture ∏ p_j^{A_ij}/Z; that
	//           mixture; and the model's own prediction p. Brackets: total variation from p.
	//   bottom  every reader with two or more sources, the same two distances, log axis
	// State: the reader, the head, a pointed-at source. Attrs (1-based, as in the prose):
	// window, position, head. Static state: the export's first figure reader. No number is
	// typed: all are read from the export or computed from it (mixture.js).
	import { getContext } from 'svelte';
	import { measure } from '#figures/primitives/measure.js';
	import Arcs, { loopGeom } from '#figures/primitives/Arcs.svelte';
	import { fmt, fmtProb, showChar, clamp, layoutWidth } from '#figures/primitives/scales.js';
	import { slice, dissenter, argmax, swarm } from './mixture.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const mx = $derived(getContext('artifacts')()?.mixture ?? null);
	const interactive = $derived(mode !== 'print');
	const vocab = $derived(mx?.vocab.data ?? []);
	const V = $derived(vocab.length);
	const T = $derived(mx?.config.T ?? 8);
	const H = $derived(mx?.config.H ?? 2);
	const readers = $derived(mx?.readers ?? []);

	// the reader shown: attrs, else the export's first figure reader
	const initial = () => {
		const f = mx?.figure_readers?.[0];
		const b = +attrs.window || f?.b || 1, i = +attrs.position || f?.i || 1;
		const k = (mx?.readers ?? []).findIndex((r) => r.b === b && r.i === i);
		return { k: Math.max(0, k), head: clamp(+attrs.head || f?.head || 1, 1, mx?.config.H ?? 2) };
	};
	// svelte-ignore state_referenced_locally
	let k = $state(initial().k);
	// svelte-ignore state_referenced_locally
	let head = $state(initial().head);
	let focusJ = $state(-1);
	let width = $state(868);

	const s = $derived(readers[k] ? slice(readers[k], head, V) : null);
	const n = $derived(s?.i ?? 0);
	const veto = $derived(s ? dissenter(s) : 0);
	const yMix = $derived(s ? argmax(s.mix) : 0);

	// every reader with more than one source, same head: the strip
	const many = $derived(readers.map((r, idx) => ({ r, idx })).filter(({ r }) => r.i > 1));
	const all = $derived(many.map(({ r, idx }) => ({ idx, ...slice(r, head, V) })));
	const single = $derived(readers.length - many.length);

	// ── geometry ────────────────────────────────────────────────────────────────
	const W = $derived(layoutWidth(width, mode, 340));
	const narrow = $derived(W < 620);
	const padL = $derived(narrow ? 4 : 112);
	const padR = $derived(narrow ? 4 : 10);
	const step = $derived((W - padL - padR) / T);
	const xs = $derived(Array.from({ length: T }, (_, j) => padL + step * (j + 0.5)));
	const fs = $derived(narrow ? 24 : 32);
	const aspect = 0.36;
	const wmax = $derived(narrow ? 8 : 11);
	// room for the arcs: in print the reader's own longest arc; on screen the longest
	// visible arc of any reader of this head, so the plate does not jump under the pointer
	const reach = $derived.by(() => {
		const arcOf = (q) => { let r = 1; for (let j = 0; j < q.i; j++) if (q.attention_rows.data[j + T * (head - 1)] >= 0.02) r = Math.max(r, q.i - 1 - j); return r; };
		return interactive ? Math.max(...readers.map(arcOf)) : arcOf(readers[k]);
	});
	const yHead = 14;
	const yArc = $derived(yHead + 30 + (reach * step) / 2 * aspect);
	const yTok = $derived(yArc + fs * 0.78);

	// small multiples, one per source
	const mw = $derived(Math.min(step * 0.8, 92));
	const mbw = $derived(mw / (V + (V - 1) * 0.32));
	const mgap = $derived(mbw * 0.32);
	const mh = $derived(narrow ? 30 : 44);
	const mChars = $derived(mbw >= 6.5);
	const yM = $derived(yTok + 36 + mh);

	// the four columns: arithmetic stack, surprise stack, geometric mixture, the model
	const rh = $derived(narrow ? 92 : interactive ? 124 : 108);
	const colW = $derived(narrow ? (W - padL - padR - 2 * 34) / 2 : 137);
	const rbw = $derived(colW / (V + (V - 1) * 0.36));
	const rgap = $derived(rbw * 0.36);
	const rw = $derived(V * rbw + (V - 1) * rgap);
	const bx = (x0, y) => x0 + y * (rbw + rgap);
	const rowGap = $derived(rh + 128); // narrow: baseline to baseline
	const yO = $derived(yM + (mChars ? 26 : 16) + 64 + rh); // first row's baseline
	const cols = $derived.by(() => {
		if (narrow) {
			const xL = padL + 30, xR = W / 2 + 30;
			return { A: [xL, yO], S: [xR, yO], G: [xL, yO + rowGap], P: [xR, yO + rowGap] };
		}
		const free = W - padL - padR - 4 * rw - 10;
		const xA = padL + 10, xS = xA + rw + free * 0.38, xG = xS + rw + free * 0.32, xP = xG + rw + free * 0.3;
		return { A: [xA, yO], S: [xS, yO], G: [xG, yO], P: [xP, yO] };
	});
	const Smax = $derived(s ? Math.max(...s.Sbar) : 1);
	const Saxis = $derived(Math.max(5, Math.ceil(Smax / 5) * 5));
	const kS = $derived(rh / Saxis); // px per nat
	const hy = (c) => c[1] - rh - 50; // column heads
	const yLast = $derived(cols.P[1]);
	const yBr = $derived(yLast + (narrow ? 76 : 64));

	// the strip
	const vals = $derived(all.flatMap((q) => [q.tvMix, q.tvGeo]).filter((v) => v > 0));
	const lo = $derived(vals.length ? Math.floor(Math.log10(Math.min(...vals))) : -6);
	const sx0 = $derived(padL + (narrow ? 8 : 14));
	const sx1 = $derived(W - padR - 10);
	const lx = (v) => sx0 + ((Math.log10(Math.max(v, 10 ** lo)) - lo) / (0 - lo)) * (sx1 - sx0);
	const dr = $derived(narrow ? 2 : 2.6);
	const swA = $derived(swarm(all.map((q) => lx(q.tvMix)), dr));
	const swG = $derived(swarm(all.map((q) => lx(q.tvGeo)), dr));
	const half = $derived(Math.max(...swA.map(Math.abs), ...swG.map(Math.abs), 4) + dr + 2);
	const yStrip = $derived(yBr + (narrow ? 34 : 66));
	const yRowA = $derived(yStrip + (narrow ? 48 : 34) + Math.max(half, 14));
	const yRowG = $derived(yRowA + 2 * half + (narrow ? 26 : 14));
	const yAxis = $derived(yRowG + half + 6);
	const Hsvg = $derived(yAxis + (narrow ? 62 : 48));
	const mean = (a) => a.reduce((x, y) => x + y, 0) / Math.max(1, a.length);
	const meanMix = $derived(mean(all.map((q) => q.tvMix)));
	const meanGeo = $derived(mean(all.map((q) => q.tvGeo)));
	const closer = $derived(all.filter((q) => q.tvGeo < q.tvMix).length);
	const decades = $derived(Array.from({ length: -lo + 1 }, (_, d) => lo + d).filter((d) => !narrow || (d - lo) % 2 === 0 || d === 0));
	const sup = (e) => String(e).replace('-', '⁻').replace(/\d/g, (c) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[c]);

	// arcs want the full matrix; only the reader's row is drawn
	const Arows = $derived(Array.from({ length: T }, (_, r) => Array.from({ length: T }, (_, j) => (r === n - 1 && j < n ? s.A[j] : 0))));

	// ── interaction ─────────────────────────────────────────────────────────────
	const order = $derived(all.slice().sort((a, b) => a.tvMix - b.tvMix).map((q) => q.idx));
	function stripKey(e) {
		if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
		e.preventDefault();
		const at = order.indexOf(k);
		k = order[at < 0 ? 0 : clamp(at + (e.key === 'ArrowLeft' ? -1 : 1), 0, order.length - 1)];
		focusJ = -1;
	}
	function setHead(h) { head = h; focusJ = -1; }
	const shown = (j) => focusJ < 0 || focusJ === j;

	// ── words ───────────────────────────────────────────────────────────────────
	const win = (t) => t.replace(/ /g, '␣');
	const who = $derived(s ? `‘${showChar(s.chars[n - 1])}’ at position ${n} of ‘${win(s.window)}’, head ${head}` : '');
	const next = $derived(s ? showChar(vocab[s.target]) : '');
	const readout = $derived.by(() => {
		if (!s) return [];
		const y = s.target;
		return [
			`Next character ‘${next}’: arithmetic mixture ${fmtProb(s.mix[y], 2, true)}, geometric mixture ${fmtProb(s.geo[y], 2, true)}, the model ${fmtProb(s.p[y], 3, true)}.`,
			`Total variation from the model: arithmetic ${fmtProb(s.tvMix, 3, true)}, geometric ${fmtProb(s.tvGeo, 3, true)}.`,
			focusJ >= 0 ? `Source ${focusJ + 1} (‘${showChar(s.chars[focusJ])}’, weight ${fmtProb(s.A[focusJ], 2, true)}) alone predicts ‘${showChar(vocab[argmax(s.P[focusJ])])}’ ${fmtProb(Math.max(...s.P[focusJ]), 2, true)}.` : ''
		].filter(Boolean);
	});
	const aria = $derived(s ? `Reader ${who}, the other head at its mean. Arithmetic mixture favours ‘${showChar(vocab[yMix])}’ ${fmt(s.mix[yMix], 2)}; geometric mixture ‘${showChar(vocab[argmax(s.geo)])}’ ${fmt(Math.max(...s.geo), 2)}; the model ‘${showChar(vocab[argmax(s.p)])}’ ${fmtProb(Math.max(...s.p), 2)}. Total variation from the model ${fmtProb(s.tvMix, 3, true)} and ${fmtProb(s.tvGeo, 3, true)}. Over ${all.length} readers: mean ${fmt(meanMix, 3)} and ${fmt(meanGeo, 4)}; the geometric mixture is the closer at ${closer}.` : 'Mixture data not available');

	const pl = (q) => fmtProb(q, 2).replace(/^0\./, '.');
	// cumulative stack heights: below[j][y] = Σ_{j' < j} part[j'][y]
	const stackOf = (part) => part.map((_, j) => part[0].map((__, y) => part.slice(0, j).reduce((a, row) => a + row[y], 0)));
	const shares = $derived(s ? s.P.map((row, j) => row.map((q) => s.A[j] * q)) : []);
	const belowA = $derived(s ? stackOf(shares) : []);
	const belowS = $derived(s ? stackOf(s.S) : []);
</script>

{#snippet chars(x0, y0, cs = 12)}
	{#each vocab as ch, y (y)}
		<text class="tok mf__ch" class:mf__ch--target={y === s.target} x={bx(x0, y) + rbw / 2} y={y0 + cs + 1} text-anchor="middle" font-size={cs}>{showChar(ch)}</text>
	{/each}
	<line class="mf__target" x1={bx(x0, s.target) - 1} x2={bx(x0, s.target) + rbw + 1} y1={y0 + cs + 4} y2={y0 + cs + 4} />
{/snippet}

{#snippet stack(x0, y0, part, below, kpx, cls)}
	<line class="mf__base" x1={x0 - 2} x2={x0 + rw + 2} y1={y0} y2={y0} />
	{#each part as row, j (j)}
		{#each row as v, y (y)}
			{@const hgt = v * kpx}
			{#if hgt > 0.05}
				{@const gap = hgt > 2.4 ? 1 : 0}
				<rect class="bar--value mf__seg {cls}" class:is-faded={!shown(j)} x={bx(x0, y)} y={y0 - (below[j][y] + v) * kpx + gap / 2} width={rbw} height={Math.max(0.35, hgt - gap)} />
				{#if hgt >= 12 && rbw >= 11}
					<text class="mf__segl" x={bx(x0, y) + rbw / 2} y={y0 - (below[j][y] + v / 2) * kpx + 3.4} text-anchor="middle">{j + 1}</text>
				{/if}
			{/if}
		{/each}
	{/each}
{/snippet}

{#snippet bars(values, x0, y0)}
	<line class="mf__base" x1={x0 - 2} x2={x0 + rw + 2} y1={y0} y2={y0} />
	<g class="marks">
		{#each values as q, y (y)}
			<rect class="bar--value" x={bx(x0, y)} y={y0 - q * rh} width={rbw} height={Math.max(0, q * rh)} />
		{/each}
	</g>
	{#each values as q, y (y)}
		{#if q >= 0.05}
			<text class="lbl lbl--micro mf__p" class:lbl--strong={q >= 0.1} x={bx(x0, y) + rbw / 2} y={y0 - q * rh - 4} text-anchor="middle">{pl(q)}</text>
		{/if}
	{/each}
{/snippet}

{#snippet sub(base, idx)}<tspan class="m">{base}</tspan><tspan class="m mf__sub" dy="0.3em">{idx}</tspan><tspan dy="-0.3em">&#8202;</tspan>{/snippet}

{#snippet head2(x, y, title, l1, l2)}
	<text class="lbl lbl--head" {x} {y}>{title}</text>
	<text class="lbl lbl--micro" {x} y={y + 15}>{@render l1()}</text>
	{#if l2}<text class="lbl lbl--micro" {x} y={y + 28}>{@render l2()}</text>{/if}
{/snippet}

<div class="fx fx-mf" use:measure={(w) => (width = w)}>
	{#if !s}
		<p class="fig__pending">The mixture export (attention-audition/artifacts/mixture.json) is not available.</p>
	{:else}
	<svg viewBox="0 0 {W} {Hsvg}" width={W} height={Hsvg} role={interactive ? 'group' : 'img'} aria-label={aria}>
		<!-- ── the reader and its sources ─────────────────────────────────────── -->
		<text class="lbl" x={padL} y={yHead}>{#if narrow}reader <tspan class="tok mf__inl">‘{showChar(s.chars[n - 1])}’</tspan>, position {n} of ‘<tspan class="tok mf__inl">{win(s.window)}</tspan>’{:else}the reader <tspan class="tok mf__inl">‘{showChar(s.chars[n - 1])}’</tspan> at position {n} of window {s.b}, ‘<tspan class="tok mf__inl">{win(s.window)}</tspan>’, and the character that comes next, <tspan class="tok mf__inl">‘{next}’</tspan>{/if}</text>
		{#if narrow}
			<text class="lbl lbl--micro" x={padL} y={yHead + 15}>head {head} reads; head {3 - head} is held at its usual output</text>
		{/if}

		{#if !narrow}
			{#if interactive}
				{#each Array.from({ length: H }, (_, h) => h + 1) as h (h)}
					<text class="lbl lbl--head mf__toggle" class:is-on={h === head} x={(h - 1) * 50} y={yArc - 22} role="button" tabindex="0" aria-pressed={h === head} onclick={() => setHead(h)} onkeydown={(e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), setHead(h))}>head {h}</text>
				{/each}
			{:else}
				<text class="lbl lbl--head" x={0} y={yArc - 22}>head {head}</text>
			{/if}
			<text class="lbl lbl--micro" x={0} y={yArc - 8}>its weights {@render sub('A', 'ij')}</text>
			<text class="lbl lbl--micro" x={0} y={yArc + 5}>(head {3 - head} held at</text>
			<text class="lbl lbl--micro" x={0} y={yArc + 18}>its usual output)</text>
			<text class="lbl lbl--head" x={0} y={yM - mh + 4}>source <tspan class="m">j</tspan> alone</text>
			<text class="lbl lbl--micro" x={0} y={yM - mh + 19}>predicts {@render sub('p', 'j')}: the head</text>
			<text class="lbl lbl--micro" x={0} y={yM - mh + 32}>output replaced by {@render sub('v', 'j')}</text>
		{:else if interactive}
			{#each Array.from({ length: H }, (_, h) => h + 1) as h (h)}
				<text class="lbl mf__toggle" class:is-on={h === head} x={W - padR - (H - h) * 52} y={yHead + 34} text-anchor="end" role="button" tabindex="0" aria-pressed={h === head} onclick={() => setHead(h)} onkeydown={(e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), setHead(h))}>head {h}</text>
			{/each}
		{/if}

		<Arcs {xs} y={yArc} A={Arows} reader={n - 1} dir={-1} {wmax} {aspect} labels={false} />
		<!-- each weight at its source's end of the arc (apex labels of nested arcs collide) -->
		<g class="arcs-labels">
			{#each s.A as w, j (j)}
				{#if j === n - 1}
					{@const g = loopGeom(xs[j], yArc, w, { wmax, loop: 9, dir: -1 })}
					<text class="arcs__label" class:mf__wsmall={w < 0.1} x={g.lx} y={g.ly} dy="0.36em">{fmtProb(w, 2)}</text>
				{:else}
					<text class="arcs__label" class:mf__wsmall={w < 0.1} x={Math.max(xs[j] - 5, 24)} y={yArc - 4} text-anchor="end">{fmtProb(w, 2)}</text>
				{/if}
			{/each}
		</g>

		{#each s.chars as ch, j (j)}
			<text class="tok tok--big" class:tok--reader={j === n - 1} class:tok--future={j >= n} x={xs[j]} y={yTok} text-anchor="middle" font-size={fs}>{showChar(ch)}</text>
			<text class="lbl lbl--micro" x={xs[j] + fs * 0.34} y={yTok + 3}>{j + 1}</text>
		{/each}
		{#if n < T}
			<text class="lbl lbl--micro" x={xs[n]} y={yTok + 16} text-anchor="middle">next</text>
		{/if}

		{#each s.P as q, j (j)}
			{@const x0 = xs[j] - mw / 2}
			{@const xk = (y) => x0 + y * (mbw + mgap)}
			{@const top = argmax(q)}
			<g class="mf__src" class:is-faded={!shown(j)}>
				<line class="mf__base" x1={x0 - 2} x2={x0 + mw + 2} y1={yM} y2={yM} />
				<g class="marks">
					{#each q as v, y (y)}
						<rect class="bar--value" x={xk(y)} y={yM - v * mh} width={mbw} height={Math.max(0, v * mh)} />
					{/each}
				</g>
				<text class="tok mf__top" x={xk(top) + mbw / 2} y={yM - q[top] * mh - 4} text-anchor="middle" font-size={narrow ? 11 : 13}>{showChar(vocab[top])}</text>
				{#if mChars}
					{#each vocab as ch, y (y)}
						<text class="tok mf__ch" class:mf__ch--target={y === s.target} x={xk(y) + mbw / 2} y={yM + 10.5} text-anchor="middle" font-size="9.5">{showChar(ch)}</text>
					{/each}
				{/if}
				<line class="mf__target" x1={xk(s.target) - 0.5} x2={xk(s.target) + mbw + 0.5} y1={yM + (mChars ? 13.5 : 3)} y2={yM + (mChars ? 13.5 : 3)} />
				{#if j === veto && q[yMix] < 0.01}
					{@const xv = xk(yMix) + mbw / 2}
					<line class="leader" x1={xv} x2={xv} y1={yM - 2} y2={yM - mh - 9} />
					<text class="lbl lbl--micro mf__veto" x={xv} y={yM - mh - 12} text-anchor="middle">‘{showChar(vocab[yMix])}’ {fmtProb(q[yMix], 2, true)}</text>
				{/if}
			</g>
			{#if interactive}
				<rect class="hit" x={xs[j] - step / 2} y={yTok - fs} width={step} height={yM + 16 - (yTok - fs)} role="button" tabindex="0" aria-label="source {j + 1}, ‘{showChar(s.chars[j])}’, weight {fmt(s.A[j], 2)}" onpointerenter={() => (focusJ = j)} onpointerleave={() => (focusJ = -1)} onfocus={() => (focusJ = j)} onblur={() => (focusJ = -1)} />
			{/if}
		{/each}

		<!-- ── four columns on one scale ──────────────────────────────────────── -->
		{#snippet hA1()}pick <tspan class="m">j</tspan> with probability {@render sub('A', 'ij')}{/snippet}
		{#snippet hA2()}{#if !narrow}and predict {@render sub('p', 'j')}:{' '}{/if}probabilities add{/snippet}
		{#snippet hS1()}the surprises −ln {@render sub('p', 'j')} add,{/snippet}
		{#snippet hS2()}weighted by {@render sub('A', 'ij')}{/snippet}
		{#snippet hG1()}e<tspan class="mf__supt" dy="-0.45em">−height</tspan><tspan dy="0.45em">, scaled</tspan>{/snippet}
		{#snippet hG2()}to sum to one{/snippet}
		{#snippet hP1()}average the values,{/snippet}
		{#snippet hP2()}then decode once{/snippet}
		{@render head2(cols.A[0], hy(cols.A), narrow ? 'choose, predict' : 'choose, then predict', hA1, hA2)}
		{@render head2(cols.S[0], hy(cols.S), narrow ? 'weigh' : 'weigh the evidence', hS1, hS2)}
		{@render head2(cols.G[0], hy(cols.G), 'renormalise', hG1, hG2)}
		{@render head2(cols.P[0], hy(cols.P), 'attention', hP1, hP2)}

		<!-- arithmetic: a stack of the sources' shares A_ij p_j(y) -->
		{@render stack(cols.A[0], cols.A[1], shares, belowA, rh, '')}
		{#each s.mix as q, y (y)}
			{#if q >= 0.05}
				<text class="lbl lbl--micro mf__p" class:lbl--strong={q >= 0.1} x={bx(cols.A[0], y) + rbw / 2} y={cols.A[1] - q * rh - 4} text-anchor="middle">{pl(q)}</text>
			{/if}
		{/each}
		{@render chars(cols.A[0], cols.A[1])}
		<line class="axis" x1={cols.A[0] - 8} x2={cols.A[0] - 8} y1={cols.A[1]} y2={cols.A[1] - rh} />
		{#each [0, 1] as t (t)}
			<line class="axis" x1={cols.A[0] - 11} x2={cols.A[0] - 8} y1={cols.A[1] - t * rh} y2={cols.A[1] - t * rh} />
			<text class="lbl lbl--micro" x={cols.A[0] - 14} y={cols.A[1] - t * rh + 3.5} text-anchor="end">{t}</text>
		{/each}

		<!-- surprise: a stack of the weighted surprises A_ij(−ln p_j(y)), in nats -->
		{@render stack(cols.S[0], cols.S[1], s.S, belowS, kS, '')}
		{#each [...new Set([s.target, yMix])] as y (y)}
			<text class="lbl lbl--micro lbl--strong" x={bx(cols.S[0], y) + rbw / 2} y={cols.S[1] - s.Sbar[y] * kS - 4} text-anchor="middle">{fmt(s.Sbar[y], 2)}</text>
		{/each}
		{@render chars(cols.S[0], cols.S[1])}
		<line class="axis" x1={cols.S[0] - 8} x2={cols.S[0] - 8} y1={cols.S[1]} y2={cols.S[1] - Saxis * kS} />
		{#each Array.from({ length: Saxis / 5 + 1 }, (_, t) => t * 5) as e (e)}
			<line class="axis" x1={cols.S[0] - 11} x2={cols.S[0] - 8} y1={cols.S[1] - e * kS} y2={cols.S[1] - e * kS} />
			<text class="lbl lbl--micro" x={cols.S[0] - 14} y={cols.S[1] - e * kS + 3.5} text-anchor="end">{e}</text>
		{/each}
		<text class="lbl lbl--micro" x={cols.S[0] - 8} y={cols.S[1] - rh - 9} text-anchor="middle">nats</text>

		<!-- the geometric mixture and the model -->
		{@render bars(s.geo, cols.G[0], cols.G[1])}
		{@render chars(cols.G[0], cols.G[1])}
		{@render bars(s.p, cols.P[0], cols.P[1])}
		{@render chars(cols.P[0], cols.P[1])}
		{#if narrow}
			<line class="axis" x1={cols.G[0] - 8} x2={cols.G[0] - 8} y1={cols.G[1]} y2={cols.G[1] - rh} />
			{#each [0, 1] as t (t)}
				<line class="axis" x1={cols.G[0] - 11} x2={cols.G[0] - 8} y1={cols.G[1] - t * rh} y2={cols.G[1] - t * rh} />
				<text class="lbl lbl--micro" x={cols.G[0] - 14} y={cols.G[1] - t * rh + 3.5} text-anchor="end">{t}</text>
			{/each}
		{/if}

		<!-- names -->
		{#each [['A', 'arithmetic mixture'], ['S', 'mean surprise'], ['G', 'geometric mixture'], ['P', narrow ? 'the model' : 'the model’s own']] as [c, name] (c)}
			<text class="lbl lbl--strong" x={cols[c][0]} y={cols[c][1] + 34}>{name}</text>
		{/each}
		<text class="lbl" x={cols.A[0]} y={cols.A[1] + 50}>Σ{@render sub('', 'j')} {@render sub('A', 'ij')} {@render sub('p', 'j')}</text>
		<text class="lbl" x={cols.S[0]} y={cols.S[1] + 50}>Σ{@render sub('', 'j')} {@render sub('A', 'ij')} (−ln {@render sub('p', 'j')}<tspan>)</tspan></text>
		<text class="lbl" x={cols.G[0]} y={cols.G[1] + 50}>∏{@render sub('', 'j')} <tspan class="m">p</tspan><tspan class="m mf__sub" dy="0.3em">j</tspan><tspan class="m mf__sub" dy="-0.75em">A</tspan><tspan class="m mf__sub2" dy="0.2em">ij</tspan><tspan dy="0.25em"> / </tspan><tspan class="m">Z</tspan></text>
		<text class="lbl" x={cols.P[0]} y={cols.P[1] + 50}><tspan class="m">p</tspan></text>

		<!-- distances to the model's own prediction -->
		{#if narrow}
			<text class="lbl" x={padL} y={yBr - 6}>total variation from <tspan class="m">p</tspan>: geometric <tspan class="lbl--strong">{fmtProb(s.tvGeo, 3, true)}</tspan>, arithmetic <tspan class="lbl--strong">{fmtProb(s.tvMix, 3, true)}</tspan></text>
		{:else}
			{@const cA = cols.A[0] + rw / 2}
			{@const cG = cols.G[0] + rw / 2}
			{@const cP = cols.P[0] + rw / 2}
			<path class="mf__br" d="M {cG} {yBr - 6} v 5 H {cP} v -5" />
			<text class="lbl" x={(cG + cP) / 2} y={yBr + 12} text-anchor="middle">total variation <tspan class="lbl--strong">{fmtProb(s.tvGeo, 3, true)}</tspan></text>
			<path class="mf__br" d="M {cA} {yBr - 6} v 32 H {cP} v -32" />
			<text class="lbl" x={(cA + cP) / 2} y={yBr + 44} text-anchor="middle">total variation from <tspan class="m">p</tspan>: <tspan class="lbl--strong">{fmtProb(s.tvMix, 3, true)}</tspan></text>
		{/if}

		<!-- ── every reader ───────────────────────────────────────────────────── -->
		{#if narrow}
			<text class="lbl lbl--head" x={padL} y={yStrip}>every reader with two or more sources</text>
			<text class="lbl lbl--micro" x={padL} y={yStrip + 14}>{all.length} readers; head {head} choosing, as above</text>
		{:else}
			<text class="lbl lbl--head" x={0} y={yStrip}>every reader</text>
			<text class="lbl lbl--micro" x={0} y={yStrip + 14}>{all.length} with two or more</text>
			<text class="lbl lbl--micro" x={0} y={yStrip + 27}>sources; head {head}, as above</text>
		{/if}
		<text class="lbl" x={narrow ? padL : 0} y={yRowA + (narrow ? -half - 4 : 4)}>arithmetic</text>
		<text class="lbl" x={narrow ? padL : 0} y={yRowG + (narrow ? -half - 4 : 4)}>geometric</text>
		<g class="marks">
			{#each all as q, m (q.idx)}
				<circle class="dot--value" cx={lx(q.tvMix)} cy={yRowA + swA[m]} r={dr} />
				<circle class="dot--value" cx={lx(q.tvGeo)} cy={yRowG + swG[m]} r={dr} />
			{/each}
		</g>
		{#each all as q, m (q.idx)}
			{#if q.idx === k}
				<circle class="mf__ring" cx={lx(q.tvMix)} cy={yRowA + swA[m]} r={dr + 2.6} />
				<circle class="mf__ring" cx={lx(q.tvGeo)} cy={yRowG + swG[m]} r={dr + 2.6} />
				<text class="lbl lbl--micro mf__this" x={lx(q.tvGeo)} y={yRowG + swG[m] + dr + 13} text-anchor="middle">this reader</text>
			{/if}
		{/each}
		{#each [[meanMix, yRowA, 3], [meanGeo, yRowG, 4]] as [v, yy, d] (yy)}
			<path class="mf__mean" d="M {lx(v)} {yy - half + 1} l -3.5 -6 h 7 z" />
			{@const right = lx(v) > sx1 - 70}
			<text class="lbl lbl--micro mf__meanl" x={lx(v) + (right ? -6 : 6)} y={yy - half - 1} text-anchor={right ? 'end' : 'start'}>mean {fmt(v, d)}</text>
		{/each}
		<line class="axis" x1={sx0} x2={sx1} y1={yAxis} y2={yAxis} />
		{#each decades as d (d)}
			<line class="axis" x1={lx(10 ** d)} x2={lx(10 ** d)} y1={yAxis} y2={yAxis + 4} />
			<text class="lbl lbl--micro" x={lx(10 ** d)} y={yAxis + 15} text-anchor="middle">{d === 0 ? '1' : `10${sup(d)}`}</text>
		{/each}
		<text class="lbl lbl--micro" x={sx1} y={yAxis + 30} text-anchor="end">total variation from <tspan class="m">p</tspan>, log scale</text>
		<text class="lbl lbl--micro" x={sx1} y={yAxis + 43} text-anchor="end">{closer === all.length ? `the geometric mixture is the closer at every one of the ${all.length}` : `the geometric mixture is the closer at ${closer} of ${all.length}`}{narrow ? '' : `; at the ${single} readers in position 1, with one source, the three coincide`}</text>
		{#if interactive}
			<rect class="hit" x={sx0 - 8} y={yRowA - half} width={sx1 - sx0 + 16} height={yRowG + half - (yRowA - half)} role="slider" tabindex="0" aria-label="reader, in order of the arithmetic mixture's distance (arrow keys)" aria-valuenow={order.indexOf(k) + 1} aria-valuemin="1" aria-valuemax={order.length} aria-valuetext={who} onkeydown={stripKey} />
			{#each all as q, m (q.idx)}
				<circle class="hit" cx={lx(q.tvMix)} cy={yRowA + swA[m]} r={dr + 2} role="presentation" onclick={() => { k = q.idx; focusJ = -1; }} />
				<circle class="hit" cx={lx(q.tvGeo)} cy={yRowG + swG[m]} r={dr + 2} role="presentation" onclick={() => { k = q.idx; focusJ = -1; }} />
			{/each}
		{/if}
	</svg>
	{#if interactive}
		<p class="readout" aria-live="polite">
			<span class="readout__who">Reader {who}.</span>
			{#each readout as r, m (m)}<span class="readout__part">{r}</span>{/each}
			<span class="readout__hint">Point at a source to pick out its share of each stack. Click a dot (or use ←/→ on the dots) to choose another reader; click “head 1” or “head 2” to switch the head.</span>
		</p>
	{/if}
	{/if}
</div>

<style>
	.fx-mf :global(.m) { font-family: var(--serif); font-style: italic; fill: var(--ink); font-size: 1.12em; }
	.fx-mf :global(.mf__sub) { font-size: 0.72em; }
	.fx-mf :global(.mf__sub2) { font-size: 0.6em; }
	.fx-mf :global(.mf__supt) { font-size: 0.78em; }
	.fx-mf .mf__inl { font-size: 13.5px; }
	.mf__base { stroke: var(--ink-3); stroke-width: var(--hair); }
	.mf__top { fill: var(--ink); font-weight: 600; }
	.mf__ch { fill: var(--ink-3); }
	.mf__ch--target { fill: var(--ink); font-weight: 600; }
	.mf__target { stroke: var(--ink); stroke-width: 1px; }
	.mf__veto { fill: var(--ink); }
	.mf__seg, .mf__src { transition: opacity var(--t-quick) var(--ease); }
	.mf__seg.is-faded, .mf__src.is-faded { opacity: 0.25; }
	.mf__segl { font-family: var(--sans); font-size: 9.5px; fill: var(--paper); font-variant-numeric: tabular-nums lining-nums; pointer-events: none; }
	.mf__br { fill: none; stroke: var(--ink-3); stroke-width: var(--hair); }
	.mf__mean, .mf__meanl, .mf__this { fill: var(--ink); }
	.mf__ring { fill: none; stroke: var(--ink); stroke-width: 1.2px; }
	.mf__wsmall { fill: var(--ink-3); }
	.mf__toggle { cursor: pointer; fill: var(--ink-3); font-weight: 400; }
	.mf__toggle.is-on { fill: var(--ink); font-weight: 600; }
	.mf__toggle:hover { fill: var(--value-deep); }
</style>

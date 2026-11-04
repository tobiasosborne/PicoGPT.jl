<script module>
	// Timelines of one pass along one stem, as fractions of the pass's clock t ∈ [0, 1].
	// Forward (value ink) runs down; backward (covector ink) runs up the same paths.
	// (label stops sit where something can be seen: mid-branch, at each ⊕, the readout)
	export const FORWARD_PHASES = [
		{ at: 0, label: 'embed' },
		{ at: 0.26, label: 'attention' },
		{ at: 0.43, label: 'x + a' },
		{ at: 0.64, label: 'MLP' },
		{ at: 0.81, label: 'x + y' },
		{ at: 0.96, label: 'readout' }
	];
	export const BACKWARD_PHASES = [
		{ at: 0, label: 'from the loss' },
		{ at: 0.31, label: 'MLP' },
		{ at: 0.5, label: 'adjoints add' },
		{ at: 0.71, label: 'attention' },
		{ at: 0.88, label: 'adjoints add' }
	];
	const lin = (t, a, b) => Math.min(1, Math.max(0, (t - a) / (b - a)));
	/** Piecewise-linear head position from knots [[t, y], …] (t increasing). */
	function along(t, knots) {
		if (t <= knots[0][0]) return knots[0][1];
		for (let k = 1; k < knots.length; k++) {
			const [t1, y1] = knots[k], [t0, y0] = knots[k - 1];
			if (t <= t1) return y0 + (y1 - y0) * lin(t, t0, t1);
		}
		return knots.at(-1)[1];
	}
	export { lin, along };
</script>

<script>
	// The residual stem: one token column of the stream, drawn as a vertical stem.
	// Forward (ink="value") runs DOWN: x₀ → fork → attention branch → ⊕ → x_mid → fork →
	// MLP branch → ⊕ → x_out → (LN_f, readout). Band width ∝ ‖x‖/√D of the stream on that
	// stretch; a branch reads a copy of the stream (hairline) and returns its update with
	// width ∝ ‖update‖/√D. At ⊕ the update is added: x ← x + f(x), an Euler step.
	// Backward (ink="covector") runs UP the same paths: band width ∝ ‖x̄‖/√D, hatched.
	// The ⊕ and the forks trade places: at a forward ⊕ the adjoint is copied both ways
	// (dot); at a forward fork the adjoint of the copy and that of the stream ADD (⊕).
	// Every width is one real norm; every spoke glyph one real vector (shared scale).
	//
	// Geometry: x (stem), bx (branch node), Y = {x0, f1, n1, p1, mid, f2, n2, p2, out, end}.
	// Widths (px): w = {x0, mid, out} for the three stretches of stream (value: x₀, x_mid,
	// x_out; covector: x̄₀, x̄_mid, x̄_out), wb = {a, y} for the returning legs (value: the
	// updates; covector: what each branch hands back, x̄₀ − x̄_mid and x̄_mid − x̄_out).
	// Vectors g = {x0, a, mid, y, out} with the same meanings, for spoke glyphs.
	// t ∈ [0, 1] is this pass's clock (1 = complete); see FORWARD_PHASES / BACKWARD_PHASES.
	import SpokeGlyph from './SpokeGlyph.svelte';

	let {
		x = 0,
		bx = 40,
		Y,
		w,
		wb,
		g = null,
		rG = 12,
		scale = 1,
		ink = 'value',
		t = 1,
		muted = false,
		glyphs = true,
		track = true,
		bars = null, // { values, scale (px per unit), signed, mark, gap }
		tipScale = 1,
		fitTips = false, // covector glyphs: bars fit between neighbouring spokes (keys)
		part = 'all' // 'attention': only x₀ → attention branch → x_mid (for keys)
	} = $props();
	const one = $derived(part === 'attention');
	const bottom = $derived(one ? Y.mid + rG + 8 : Y.end);

	const cov = $derived(ink === 'covector');
	const gap = $derived(glyphs ? rG + 3 : 0);

	// the stream's three stretches; value changes at the ⊕ (p), covector at the forks (f)
	const cuts = $derived(cov ? [Y.f1, Y.f2] : [Y.p1, Y.p2]);
	const stretches = $derived([
		{ y0: Y.x0, y1: cuts[0], w: w.x0 },
		{ y0: cuts[0], y1: cuts[1], w: w.mid },
		{ y0: cuts[1], y1: Y.end, w: w.out }
	].map((q) => ({ ...q, y1: Math.min(q.y1, bottom) })).filter((q) => q.y1 > q.y0));
	const stations = $derived(one ? [Y.x0, Y.mid] : [Y.x0, Y.mid, Y.out]);
	/** band pieces with gaps around the station glyphs */
	const pieces = $derived.by(() => {
		const out = [];
		for (const s of stretches) {
			let segs = [[s.y0, s.y1]];
			for (const st of stations) {
				if (!gap) break;
				segs = segs.flatMap(([a, b]) => {
					const lo = st - gap, hi = st + gap;
					if (hi <= a || lo >= b) return [[a, b]];
					const r = [];
					if (lo > a) r.push([a, lo]);
					if (hi < b) r.push([hi, b]);
					return r;
				});
			}
			for (const [a, b] of segs) if (b - a > 0.5) out.push({ y0: a, y1: b, w: s.w });
		}
		return out;
	});

	// the head of the ink along the main stem
	const head = $derived(
		cov
			? along(t, [[0.08, Y.end], [0.14, Y.out], [0.2, Y.p2], [0.46, Y.f2], [0.54, Y.mid], [0.6, Y.p1], [0.84, Y.f1], [0.92, Y.x0]])
			: along(t, [[0.06, Y.x0], [0.14, Y.f1], [0.38, Y.p1], [0.46, Y.mid], [0.52, Y.f2], [0.76, Y.p2], [0.84, Y.out], [0.9, Y.end]])
	);
	function clip(p) {
		const a = cov ? Math.max(p.y0, head) : p.y0;
		const b = cov ? p.y1 : Math.min(p.y1, head);
		return b - a > 0.2 ? { ...p, y0: a, y1: b } : null;
	}

	// branch legs: the copy (hairline) and the return (width wb); paths run in the
	// direction this pass travels them, so a dash of length u draws them progressively.
	function legOut(f, n) { // fork → node (geometrically the upper leg)
		const dx = bx - x;
		return `M ${x} ${f} C ${x + dx * 0.85} ${f} ${bx} ${f + (n - f) * 0.2} ${bx} ${n - gap}`;
	}
	function legBack(n, p) { // node → merge (the lower leg)
		const dx = bx - x;
		return `M ${bx} ${n + gap} C ${bx} ${p - (p - n) * 0.2} ${x + dx * 0.85} ${p} ${x} ${p}`;
	}
	const rev = (d) => {
		// reverse a single cubic "M a b C c d e f g h"
		const n = d.replace(/[MC]/g, '').trim().split(/\s+/).map(Number);
		return `M ${n[6]} ${n[7]} C ${n[4]} ${n[5]} ${n[2]} ${n[3]} ${n[0]} ${n[1]}`;
	};
	const branches = $derived.by(() => {
		const B = [
			{ f: Y.f1, n: Y.n1, p: Y.p1, wb: wb.a, v: g?.a, T: cov ? [0.6, 0.66, 0.74, 0.82] : [0.14, 0.22, 0.3, 0.38] },
			{ f: Y.f2, n: Y.n2, p: Y.p2, wb: wb.y, v: g?.y, T: cov ? [0.2, 0.26, 0.34, 0.42] : [0.52, 0.6, 0.68, 0.76] }
		];
		return (one ? B.slice(0, 1) : B).map((b) => {
			const [t0, t1, t2, t3] = b.T;
			const first = lin(t, t0, t1), node = lin(t, t1, t2), second = lin(t, t2, t3);
			// value: copy goes out (upper), update comes back (lower).
			// covector: copy goes in from the merge (lower, reversed), the adjoint returns to the fork (upper, reversed).
			return cov
				? { ...b, copy: rev(legBack(b.n, b.p)), ret: rev(legOut(b.f, b.n)), uCopy: first, uNode: node, uRet: second }
				: { ...b, copy: legOut(b.f, b.n), ret: legBack(b.n, b.p), uCopy: first, uNode: node, uRet: second };
		});
	});

	// the joints: value ⊕ at the merges, covector ⊕ at the forks
	const plusAt = $derived((cov ? [Y.f1, Y.f2] : [Y.p1, Y.p2]).slice(0, one ? 1 : 2));
	const dotAt = $derived((cov ? [Y.p1, Y.p2] : [Y.f1, Y.f2]).slice(0, one ? 1 : 2));
	const reached = (y) => (cov ? head <= y + 0.01 : head >= y - 0.01);

	// glyph values at the stations, as the clock runs (the Euler step, literally)
	const sv = $derived.by(() => {
		if (!g) return null;
		const add = (u, v, s) => u.map((x, k) => x + s * v[k]);
		const zero = (v) => v.map(() => 0);
		if (!cov) {
			const s0 = lin(t, 0, 0.06), s1 = lin(t, 0.38, 0.46), s2 = lin(t, 0.76, 0.84);
			return {
				x0: g.x0.map((v) => v * s0),
				mid: t >= 0.38 ? add(g.x0, g.a, s1) : zero(g.x0),
				out: t >= 0.76 ? add(g.mid, g.y, s2) : zero(g.x0),
				showX0: t > 0, showMid: t >= 0.38, showOut: t >= 0.76
			};
		}
		const s0 = lin(t, 0.14, 0.2), s1 = lin(t, 0.46, 0.54), s2 = lin(t, 0.84, 0.92);
		return {
			out: g.out.map((v) => v * s0),
			mid: t >= 0.46 ? add(g.out, g.y, s1) : zero(g.out),
			x0: t >= 0.84 ? add(g.mid, g.a, s2) : zero(g.out),
			showOut0: t > 0.14, showMid: t >= 0.46, showX0: t >= 0.84
		};
	});

	// readout bars at the foot of the stem
	const barGeom = $derived.by(() => {
		if (!bars) return null;
		const n = bars.values.length, bw = bars.width ?? 4, bg = bars.gap ?? 2;
		const total = n * bw + (n - 1) * bg;
		const base = Y.end + (bars.offset ?? 0);
		const u = cov ? lin(t, 0, 0.08) : lin(t, 0.9, 1);
		return bars.values.map((v, k) => {
			const h = Math.abs(v) * bars.scale * u;
			const x0 = x - total / 2 + k * (bw + bg);
			const up = !bars.signed || v >= 0;
			return { k, v, x: x0, y: up ? base - h : base, w: bw, h, mark: k === bars.mark };
		});
	});
	const barLine = $derived(bars ? { x1: x - ((bars.values.length * (bars.width ?? 4) + (bars.values.length - 1) * (bars.gap ?? 2)) / 2) - 2, y: Y.end + (bars.offset ?? 0) } : null);
</script>

<g class="rstem rstem--{ink}" class:rstem--muted={muted}>
	{#if track}
		<!-- the diagram itself: hairline structure, visible before any ink arrives -->
		<g class="rstem__track">
			<line x1={x} x2={x} y1={Y.x0} y2={bottom} />
			{#each branches as b, k (k)}
				<path d={legOut(b.f, b.n)} /><path d={legBack(b.n, b.p)} />
			{/each}
		</g>
	{/if}

	<!-- the stream: bands, inked as far as the head has come -->
	{#each pieces as p, k (k)}
		{@const c = clip(p)}
		{#if c}
			<rect class="rstem__band" x={x - Math.max(0.6, c.w) / 2} y={c.y0} width={Math.max(0.6, c.w)} height={c.y1 - c.y0} />
		{/if}
	{/each}

	<!-- the branches -->
	{#each branches as b, k (k)}
		{#if b.uCopy > 0}
			<path class="rstem__copy" d={b.copy} pathLength="1" stroke-dasharray="{b.uCopy} 1" />
		{/if}
		{#if b.uRet > 0}
			{#if cov && b.wb > 2.2}
				<!-- a hatched ribbon with covector edges: a 1-form's stack of planes -->
				<path class="rstem__ret rstem__ret--edge" d={b.ret} pathLength="1" stroke-dasharray="{b.uRet} 1" stroke-width={b.wb + 1} />
				<path class="rstem__ret rstem__ret--paper" d={b.ret} pathLength="1" stroke-dasharray="{b.uRet} 1" stroke-width={b.wb - 0.2} />
				<path class="rstem__ret" d={b.ret} pathLength="1" stroke-dasharray="{b.uRet} 1" stroke-width={b.wb - 0.2} />
			{:else}
				<path class="rstem__ret" class:rstem__ret--thin={cov} d={b.ret} pathLength="1" stroke-dasharray="{b.uRet} 1" stroke-width={Math.max(0.6, b.wb)} />
			{/if}
		{/if}
		{#if glyphs && b.v && b.uNode > 0.001}
			<SpokeGlyph values={b.v.map((v) => v * b.uNode)} x={bx} y={b.n} rmax={rG} {scale} {ink} muted={muted} tip={1.9 * tipScale} fit={fitTips} />
		{:else if glyphs && b.v && track}
			<circle class="rstem__station" cx={bx} cy={b.n} r="1.2" />
		{:else if !glyphs && b.uNode > 0}
			<circle class="rstem__node" cx={bx} cy={b.n} r="1.8" />
		{/if}
	{/each}

	<!-- joints -->
	{#each dotAt as y, k (k)}
		<circle class="rstem__fork" class:is-on={reached(y)} cx={x} cy={y} r="1.6" />
	{/each}
	{#each plusAt as y, k (k)}
		<g class="rstem__plus" class:is-on={reached(y)} transform="translate({x} {y})">
			<circle r="4.6" /><path d="M -2.6 0 H 2.6 M 0 -2.6 V 2.6" />
		</g>
	{/each}

	<!-- the stream's value (or adjoint) at the three stations -->
	{#if glyphs && sv}
		{#each cov ? [['out', Y.out, sv.showOut0], ['mid', Y.mid, sv.showMid], ['x0', Y.x0, sv.showX0]] : [['x0', Y.x0, sv.showX0], ['mid', Y.mid, sv.showMid], ['out', Y.out, sv.showOut]] as [key, yy, show] (key)}
			{#if !(one && key === 'out')}
				{#if show}
					<SpokeGlyph values={sv[key]} {x} y={yy} rmax={rG} {scale} {ink} {muted} tip={2.1 * tipScale} fit={fitTips} />
				{:else if track}
					<circle class="rstem__station" cx={x} cy={yy} r="1.2" />
				{/if}
			{/if}
		{/each}
	{/if}

	<!-- readout: the distribution (value) or the seed z̄ = (p − e_y)/N (covector) -->
	{#if barGeom}
		<line class="rstem__base" x1={barLine.x1} x2={2 * x - barLine.x1} y1={barLine.y} y2={barLine.y} />
		{#each barGeom as b (b.k)}
			{#if b.h > 0.05}
				{#if cov && !muted && b.w < 5}
					<!-- too narrow to hatch: the barred covector mark, a stem with a bar across its tip -->
					{@const tip = b.y === barLine.y ? b.y + b.h : b.y}
					<line class="rstem__tick" x1={b.x + b.w / 2} x2={b.x + b.w / 2} y1={barLine.y} y2={tip} />
					<line class="rstem__tick rstem__tick--bar" x1={b.x - 0.3} x2={b.x + b.w + 0.3} y1={tip} y2={tip} />
				{:else}
					<rect class="rstem__bar" class:is-mark={b.mark} x={b.x} y={b.y} width={b.w} height={b.h} />
				{/if}
			{/if}
		{/each}
	{/if}
</g>

<style>
	.rstem__track :global(line), .rstem__track :global(path) { stroke: var(--rule); stroke-width: var(--hair); fill: none; }
	.rstem--value .rstem__band { fill: var(--value); }
	.rstem--covector .rstem__band { fill: url(#hatch-covector); stroke: var(--covector); stroke-width: 0.5px; }
	.rstem__copy { fill: none; stroke-width: 0.8px; stroke-linecap: butt; }
	.rstem--value .rstem__copy { stroke: var(--value); }
	.rstem--covector .rstem__copy { stroke: var(--covector); }
	.rstem__ret { fill: none; stroke-linecap: butt; }
	.rstem--value .rstem__ret { stroke: var(--value); }
	.rstem--covector .rstem__ret { stroke: url(#hatch-covector); }
	:global([data-mode='print']) .rstem--covector .rstem__ret { stroke: var(--covector); stroke-dasharray: 1 2.2; }
	.rstem--covector .rstem__ret--edge { stroke: var(--covector); }
	.rstem--covector .rstem__ret--paper { stroke: var(--paper); }
	.rstem--covector .rstem__ret--thin { stroke: var(--covector); }
	.rstem__fork { fill: var(--rule); }
	.rstem--value .rstem__fork.is-on { fill: var(--value); }
	.rstem--covector .rstem__fork.is-on { fill: var(--covector); }
	.rstem__node { fill: var(--ink-3); }
	.rstem__station { fill: var(--rule); }
	.rstem__plus circle { fill: var(--paper); stroke: var(--rule); stroke-width: 0.8px; }
	.rstem__plus path { stroke: var(--rule); stroke-width: 0.9px; }
	.rstem__plus.is-on circle, .rstem__plus.is-on path { stroke: var(--ink); }
	.rstem__base { stroke: var(--rule); stroke-width: var(--hair); }
	.rstem--value .rstem__bar { fill: var(--value); }
	.rstem--covector .rstem__bar { fill: url(#hatch-covector); stroke: var(--covector); stroke-width: 0.5px; }
	.rstem__tick { stroke: var(--covector); stroke-width: 0.9px; stroke-linecap: butt; }
	.rstem__tick--bar { stroke-width: 1.3px; }
	.rstem--muted .rstem__band, .rstem--muted .rstem__bar { fill: var(--ink-3); opacity: 0.2; stroke: none; }
	.rstem--muted .rstem__copy, .rstem--muted .rstem__ret { stroke: var(--ink-3); opacity: 0.2; }
	/* the recorded forward pass under a backward pass: its joints give way to the covector's */
	.rstem--muted .rstem__plus, .rstem--muted .rstem__fork { display: none; }
</style>

<script module>
	/** The self-read as a teardrop loop: it leaves the reader, leans to the right (the
	 *  future side, where no arc of this reader runs) and comes back. Its stroke is the
	 *  weight, like every arc; the lobe grows with the stroke so the hole stays open,
	 *  and the base is lifted by half a stroke so the loop never sits on the character.
	 *  Returns the path, the lobe centre (cx, cy), its outer radius and the label anchor. */
	export function loopGeom(x, y, w, { wmax = 12, loop = 9, dir = -1 } = {}) {
		const sw = Math.max(0.0001, w * wmax);
		const hole = Math.max(loop * 0.85, sw); // the hole stays open: twice as wide as the stroke
		const rho = hole + sw / 2; // centre line of the lobe
		const L = rho + loop * 0.45; // reader → lobe centre: a short stem
		const tilt = (38 * Math.PI) / 180; // leans right, away from the incoming arcs
		const px = x, py = y + dir * sw * 0.5;
		const ux = Math.sin(tilt), uy = dir * Math.cos(tilt);
		const cx = px + ux * L, cy = py + uy * L;
		const a = Math.asin(Math.min(0.95, rho / L)); // half-angle of the stem at the reader
		const t = Math.sqrt(Math.max(0, L * L - rho * rho)); // tangent length
		const rot = (v, ang) => [v[0] * Math.cos(ang) - v[1] * Math.sin(ang), v[0] * Math.sin(ang) + v[1] * Math.cos(ang)];
		const d1 = rot([ux, uy], -a), d2 = rot([ux, uy], a);
		const p1 = [px + d1[0] * t, py + d1[1] * t], p2 = [px + d2[0] * t, py + d2[1] * t];
		// p1 → around the far side of the lobe → p2; sweep chosen so the arc goes the long way
		const sweep = 1; // clockwise on screen, for either side of the line
		const f = (v) => v.toFixed(2);
		const d = `M ${f(px)} ${f(py)} L ${f(p1[0])} ${f(p1[1])} A ${f(rho)} ${f(rho)} 0 1 ${sweep} ${f(p2[0])} ${f(p2[1])} Z`;
		const outer = rho + sw / 2;
		return { d, sw, cx, cy, outer, lx: cx + outer + 3, ly: cy };
	}
</script>

<script>
	// Attention as arcs between characters. For a reader i, one arc to every
	// source j ≤ i; stroke width ∝ A_ij (weight 1 ↦ wmax). dir = -1 draws above the
	// line (head 1 by convention), +1 below (head 2). A reader reading itself is a
	// loop (loopGeom). Every arc is one number; arcs below `labelAt` are drawn but not
	// labelled. Labels sit in their own group, outside the overprinting marks, so their
	// paper halo really knocks the arcs out behind the digits.
	import { fmtProb } from './scales.js';

	let {
		xs = [],
		y = 0,
		A = [],
		reader = 0,
		dir = -1,
		wmax = 12,
		aspect = 0.62,
		hmax = 1e9,
		loop = 9,
		labelAt = 0.1,
		labels = true,
		muted = false,
		ink = 'value',
		digits = 2
	} = $props();

	const arcs = $derived.by(() => {
		const out = [];
		const row = A[reader] ?? [];
		for (let j = 0; j <= reader && j < row.length; j++) {
			const w = row[j];
			const x1 = xs[j], x2 = xs[reader];
			if (j === reader) {
				const g = loopGeom(x2, y, w, { wmax, loop, dir });
				out.push({ j, w, self: true, d: g.d, lx: g.lx, ly: g.ly, anchor: 'start', dy: '0.36em' });
			} else {
				const rx = Math.abs(x2 - x1) / 2;
				const ry = Math.min(hmax, rx * aspect); // constant aspect: arcs nest, never cross
				const mid = (x1 + x2) / 2;
				// elliptical arc from source to reader
				out.push({ j, w, self: false, x: x1, d: `M ${x1} ${y} A ${rx} ${ry} 0 0 ${dir < 0 ? 1 : 0} ${x2} ${y}`, lx: mid, ly: y + dir * (ry + 7), anchor: 'middle', dy: dir < 0 ? 0 : '0.72em' });
			}
		}
		// a short arc's label would sit on the loop: end it just left of the loop instead
		const self = out.find((a) => a.self);
		if (self) {
			const g = loopGeom(xs[reader], y, self.w, { wmax, loop, dir });
			const left = g.cx - g.outer - 2, half = 12; // half the width of a label like “0.33”
			for (const a of out) {
				if (a.self || a.lx + half <= left || Math.abs(a.ly - g.cy) > g.outer + 10) continue;
				a.lx = left; a.anchor = 'end';
			}
		}
		return out;
	});
</script>

<g class="arcs arcs--{ink}" class:arcs--muted={muted}>
	{#each arcs as a (a.j)}
		{#if a.self}
			<path class="arcs__arc arcs__arc--self" d={a.d} stroke-width={Math.max(0.0001, a.w * wmax)} />
		{:else}
			<path class="arcs__arc" d={a.d} stroke-width={Math.max(0.0001, a.w * wmax)} />
			{#if a.w >= 0.02}
				<path class="arcs__arc" d="M {a.x - 3} {y + dir * 6} L {a.x} {y} L {a.x + 3} {y + dir * 6}" stroke-width={Math.max(0.6, a.w * wmax * 0.3)} />
			{/if}
		{/if}
	{/each}
</g>
{#if labels && !muted}
	<g class="arcs-labels arcs-labels--{ink}">
		{#each arcs as a (a.j)}
			{#if a.w >= labelAt}
				<text class="arcs__label" x={a.lx} y={a.ly} dy={a.dy} text-anchor={a.anchor}>{fmtProb(a.w, digits)}</text>
			{/if}
		{/each}
	</g>
{/if}

<style>
	.arcs__arc--self { stroke-linejoin: round; }
</style>

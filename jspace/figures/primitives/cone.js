// cone.js — the geometry of the J-space (chapter 4 of *Poised to speak*): cones of
// nonnegative combinations of J-lens vectors, nearest points on them, and the drawing of a
// two-vector cone in its own plane. Pure functions of the data; no numbers of their own.
// Used by ConeGlyph.svelte and the fig-j-* figures of chapter 4.
//
// Vectors are plain arrays of length D. A "plane" is the 2-dimensional subspace spanned
// by two vectors v_s, v_u, with the orthonormal basis e1 = v_s/‖v_s‖ and e2 the unit
// vector of v_u − ⟨v_u, e1⟩e1; in it v_s = (‖v_s‖, 0) and v_u = (b1, b2) with b2 > 0.

export const dot = (a, b) => {
	let s = 0;
	for (let i = 0; i < a.length; i++) s += a[i] * b[i];
	return s;
};
export const norm = (a) => Math.sqrt(dot(a, a));

/** Σ_i c_i cols_i (cols: k vectors of length D). */
export function combine(cols, c) {
	const y = new Array(cols[0]?.length ?? 0).fill(0);
	cols.forEach((v, i) => { if (c[i]) for (let q = 0; q < y.length; q++) y[q] += c[i] * v[q]; });
	return y;
}

/** Least squares on a few columns (normal equations, Gaussian elimination with pivoting).
 *  Null if the columns are numerically dependent. */
function lsq(cols, x) {
	const k = cols.length;
	const M = cols.map((u) => [...cols.map((w) => dot(u, w)), dot(u, x)]);
	const scale = Math.max(1e-300, ...M.map((r, i) => Math.abs(r[i])));
	for (let i = 0; i < k; i++) {
		let p = i;
		for (let r = i + 1; r < k; r++) if (Math.abs(M[r][i]) > Math.abs(M[p][i])) p = r;
		if (Math.abs(M[p][i]) < 1e-12 * scale) return null;
		[M[i], M[p]] = [M[p], M[i]];
		for (let r = 0; r < k; r++) {
			if (r === i) continue;
			const f = M[r][i] / M[i][i];
			for (let c = i; c <= k; c++) M[r][c] -= f * M[i][c];
		}
	}
	return M.map((r, i) => r[k] / r[i]);
}

/**
 * min over c ≥ 0 of ‖x − Σ c_i cols_i‖², exactly, for a handful of columns: the minimiser
 * is the least-squares solution on some support P whose coefficients are all positive (or
 * c = 0), so try every support and keep the best feasible one. Returns
 * { c, y = Σ c_i cols_i, d2 = ‖x − y‖² }. (The analysis uses Lawson–Hanson, `nnls` in
 * jspace/src/JLens.jl; for k ≤ 7 the two agree to rounding.)
 */
export function nnlsExact(cols, x) {
	const k = cols.length;
	let best = { c: new Array(k).fill(0), y: new Array(x.length).fill(0), d2: dot(x, x) };
	for (let mask = 1; mask < 1 << k; mask++) {
		const idx = [];
		for (let i = 0; i < k; i++) if (mask & (1 << i)) idx.push(i);
		const z = lsq(idx.map((i) => cols[i]), x);
		if (!z || z.some((v) => !(v > 0))) continue;
		const c = new Array(k).fill(0);
		idx.forEach((i, m) => (c[i] = z[m]));
		const y = combine(cols, c);
		const d2 = dot(x, x) - 2 * dot(x, y) + dot(y, y);
		if (d2 < best.d2 - 1e-14) best = { c, y, d2: Math.max(0, d2) };
	}
	return best;
}

/** The plane of v_s and v_u: its basis and the two vectors' coordinates in it. */
export function planeOf(vs, vu) {
	const na = norm(vs);
	const e1 = vs.map((z) => z / na);
	const b1 = dot(vu, e1);
	const perp = vu.map((z, i) => z - b1 * e1[i]);
	const b2 = norm(perp);
	const e2 = perp.map((z) => z / (b2 || 1));
	return { e1, e2, a: [na, 0], b: [b1, b2], angle: Math.atan2(b2, b1) };
}

/** x in the plane: its orthogonal projection p (plane coordinates) and its height h above it. */
export function inPlane(plane, x) {
	const p = [dot(x, plane.e1), dot(x, plane.e2)];
	return { p, h: Math.sqrt(Math.max(0, dot(x, x) - p[0] ** 2 - p[1] ** 2)) };
}

/** A point of the plane (plane coordinates) back in ℝ^D. */
export const fromPlane = (plane, u) => plane.e1.map((z, i) => u[0] * z + u[1] * plane.e2[i]);

/**
 * The nearest point of the cone {c_s v_s + c_u v_u : c ≥ 0} to a point p of its plane
 * (plane coordinates). The plane falls into four regions: the cone itself (the nearest
 * point is p), two strips beyond its edges (the foot of the perpendicular on that edge),
 * and the polar cone (every point there is nearest to the apex 0).
 * Returns { c: [c_s, c_u], q: nearest point, region: 'cone' | 's' | 'u' | 'apex' }.
 */
export function nearestInPlane(plane, p) {
	const [a1] = plane.a, [b1, b2] = plane.b;
	const cu = p[1] / b2, cs = (p[0] - cu * b1) / a1;
	if (cs >= 0 && cu >= 0) return { c: [cs, cu], q: [p[0], p[1]], region: 'cone' };
	const nb = Math.hypot(b1, b2);
	const ts = Math.max(0, p[0]); // ⟨p, v̂_s⟩, v̂_s = (1, 0)
	const tu = Math.max(0, (p[0] * b1 + p[1] * b2) / nb);
	if (ts === 0 && tu === 0) return { c: [0, 0], q: [0, 0], region: 'apex' };
	if (ts >= tu) return { c: [ts / a1, 0], q: [ts, 0], region: 's' };
	return { c: [0, tu / nb], q: [(tu * b1) / nb, (tu * b2) / nb], region: 'u' };
}

/** Everything about one column and one pair: the plane, x's projection and height, the
 *  nearest point of the cone, and the squared remainder (in-plane part + height²). */
export function pairFit(vs, vu, x) {
	const plane = planeOf(vs, vu);
	const { p, h } = inPlane(plane, x);
	const near = nearestInPlane(plane, p);
	const inplane2 = (p[0] - near.q[0]) ** 2 + (p[1] - near.q[1]) ** 2;
	const xx = dot(x, x);
	return { plane, p, h, ...near, d2: h * h + inplane2, inplane2, frac: (h * h + inplane2) / xx, xx };
}

/** Unit outward normals of the cone's two edges (in the plane): the edges of the polar cone
 *  {y : ⟨y, v_s⟩ ≤ 0, ⟨y, v_u⟩ ≤ 0}. Edge s lies at angle 0, edge u at the angle θ of v_u;
 *  their outward normals point to −π/2 and θ + π/2. */
export function polarEdges(plane) {
	const [b1, b2] = plane.b, nb = Math.hypot(b1, b2);
	return { s: [0, -1], u: [-b2 / nb, b1 / nb] };
}

/** Clip the segment p0–p1 to a convex polygon (counter-clockwise or clockwise). Null if outside. */
export function clipToPolygon(p0, p1, poly) {
	let t0 = 0, t1 = 1;
	const d = [p1[0] - p0[0], p1[1] - p0[1]];
	const n = poly.length;
	// orientation of the polygon
	let area = 0;
	for (let i = 0; i < n; i++) { const a = poly[i], b = poly[(i + 1) % n]; area += a[0] * b[1] - b[0] * a[1]; }
	const sgn = area >= 0 ? 1 : -1;
	for (let i = 0; i < n; i++) {
		const a = poly[i], b = poly[(i + 1) % n];
		// inward normal
		const nx = -sgn * (b[1] - a[1]), ny = sgn * (b[0] - a[0]);
		const num = nx * (p0[0] - a[0]) + ny * (p0[1] - a[1]);
		const den = nx * d[0] + ny * d[1];
		if (Math.abs(den) < 1e-15) { if (num < 0) return null; continue; }
		const t = -num / den;
		if (den > 0) t0 = Math.max(t0, t); else t1 = Math.min(t1, t);
		if (t0 > t1) return null;
	}
	return [[p0[0] + t0 * d[0], p0[1] + t0 * d[1]], [p0[0] + t1 * d[0], p0[1] + t1 * d[1]]];
}

/** The sector between directions θ0 and θ1 (θ0 < θ1 < θ0 + π) of radius R, as a polygon. */
export function sector(theta0, theta1, R, steps = 24) {
	const pts = [[0, 0]];
	for (let i = 0; i <= steps; i++) {
		const t = theta0 + ((theta1 - theta0) * i) / steps;
		pts.push([R * Math.cos(t), R * Math.sin(t)]);
	}
	return pts;
}

/** Hatching of the cone (plane coordinates): rules perpendicular to its bisector, spaced
 *  `gap` apart, clipped to the sector of radius R. A drawing of the set, not data. */
export function hatchCone(plane, R, gap) {
	const th = plane.angle;
	const poly = sector(0, th, R);
	const m = [Math.cos(th / 2), Math.sin(th / 2)], t = [-m[1], m[0]];
	const out = [];
	for (let r = gap; r < R; r += gap) {
		const c = [m[0] * r, m[1] * r];
		const seg = clipToPolygon([c[0] - t[0] * 2 * R, c[1] - t[1] * 2 * R], [c[0] + t[0] * 2 * R, c[1] + t[1] * 2 * R], poly);
		if (seg) out.push(seg);
	}
	return out;
}

/**
 * An orthographic view of the plane from above at elevation β (radians), the plane turned
 * by ψ about its normal: plane direction (cos φ, sin φ) appears at screen (cos(φ+ψ),
 * −sin β·sin(φ+ψ)), height w straight up, foreshortened by cos β. Returns the map
 * (u, v, w) ↦ [X, Y] and its inverse on the plane (w = 0).
 */
export function axonometric({ psi = 0, beta = 0.42, scale = 40, ox = 0, oy = 0 } = {}) {
	const c = Math.cos(psi), s = Math.sin(psi), sb = Math.sin(beta), cb = Math.cos(beta);
	const P = (u, v, w = 0) => [ox + scale * (u * c - v * s), oy - scale * ((u * s + v * c) * sb + w * cb)];
	const inv = (X, Y) => {
		const A = (X - ox) / scale, B = -(Y - oy) / (scale * sb); // A = u c − v s, B = u s + v c
		return [A * c + B * s, -A * s + B * c];
	};
	return { P, inv };
}

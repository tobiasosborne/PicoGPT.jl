// Chapter 6, "The block": LayerNorm drawn as what it is, a projection onto a sphere.
// Every column x that enters one of the specimen's three LayerNorms (before attention,
// before the MLP, before the readout), at forty moments of training, is one radial stroke.
// It starts at the length |x̃| of the centred column and ends at the length |x̂| of its
// normalised image, just inside √D. Its angle is the column's direction projected onto the
// plane of the two leading principal directions of all centred columns. No circle is drawn:
// the sphere appears because every stroke ends on it.
import { run, TRAIN, STEPS } from './specimen.mjs';
import { svg, save, ink, stroke } from './svg.mjs';

const D = 8, eps = 1e-5;
const moments = new Set(Array.from({ length: 40 }, (_, k) => Math.round((STEPS - 1) * (k / 39) ** 2)));
const cols = [];                                           // centred columns x̃ ∈ 1⟂
run({ observe: (m, tr, t) => {
  if (!moments.has(t)) return;
  for (const X of [tr.X0, tr.blocks[0].Xmid, tr.blocks[0].Xout]) for (let n = 0; n < X.shape[1]; n++) {
    const x = Array.from(X.column(n)), mu = x.reduce((a, b) => a + b) / D;
    cols.push({ v: x.map(xi => xi - mu), t });
  }
}});

// Leading two principal directions of the centred columns (power iteration on Σ x̃x̃ᵀ).
const C = Array.from({ length: D }, (_, i) => Array.from({ length: D }, (_, j) => cols.reduce((s, { v }) => s + v[i] * v[j], 0)));
const dirs = [];
for (let k = 0; k < 2; k++) {
  let u = Array.from({ length: D }, (_, i) => Math.sin(1 + i + 3 * k));
  for (let it = 0; it < 300; it++) {
    let w = u.map((_, i) => C[i].reduce((s, c, j) => s + c * u[j], 0));
    for (const e of dirs) { const c = w.reduce((s, x, i) => s + x * e[i], 0); w = w.map((x, i) => x - c * e[i]); }
    const n = Math.hypot(...w); u = w.map(x => x / n);
  }
  dirs.push(u);
}

const norm = v => Math.hypot(...v), rmax = Math.max(...cols.map(({ v }) => norm(v)));
const size = 150, c = size / 2, R = size / 2 - 6, r = s => R * Math.log1p(s) / Math.log1p(rmax);   // logarithmic radius
const f = v => v.toFixed(3);
const paths = cols.map(({ v, t }) => {
  const a = Math.atan2(v.reduce((s, x, i) => s + x * dirs[1][i], 0), v.reduce((s, x, i) => s + x * dirs[0][i], 0));
  const r0 = r(norm(v)), r1 = r(norm(v) / Math.sqrt(norm(v) ** 2 / D + eps));   // |x̂| = |x̃|/σ
  return { d: `M${f(c + r0 * Math.cos(a))} ${f(c - r0 * Math.sin(a))}L${f(c + r1 * Math.cos(a))} ${f(c - r1 * Math.sin(a))}`,
           opacity: 0.05 + 0.2 * t / STEPS, ...(t === STEPS - 1 ? { color: ink.value, opacity: 0.6 } : {}) };
});

save(process.argv[2] ?? '/tmp/block.svg', svg({ width: size, height: size, paths,
  title: 'Every column that entered one of the specimen\'s LayerNorms, drawn as its journey onto the sphere.',
  desc: `${cols.length} centred columns x̃ (80 per LayerNorm × 3 LayerNorms × ${moments.size} moments of training, ${TRAIN}). ` +
        `Each stroke runs radially (radius ∝ log(1 + length)) from |x̃| to |x̂| = |x̃|/σ, which is just inside √D = ${Math.sqrt(D).toFixed(3)}; ` +
        `its angle is the direction of x̃ projected onto the plane of the two leading principal directions of all centred columns. ` +
        `Value ink: the trained model. The circle is not drawn; it is where the strokes end.` }));

// Chapter 2, "Symbols become vectors": how far each of the seven embedding vectors E e_a moves
// from its random start during training, E e_a(t) − E e_a(0), in 64 independent runs. Each
// run is viewed in its own best plane: the span of the two leading singular vectors of its
// final displacement. Every run is a seven-armed burst from one point, each run's burst
// differently turned: what is learned is a geometry, not an orientation.
import { run, TRAIN, STEPS, vocab } from './specimen.mjs';
import { svg, polyline, simplify, save, ink, stroke } from './svg.mjs';

const seeds = 64, D = 8, V = vocab.length;

// Leading two eigenvectors of the symmetric D×D matrix M (power iteration with deflation).
function plane(M) {
  const vecs = [];
  for (let k = 0; k < 2; k++) {
    let v = Array.from({ length: D }, (_, i) => Math.cos(i + k));
    for (let it = 0; it < 500; it++) {
      let w = v.map((_, i) => M[i].reduce((s, m, j) => s + m * v[j], 0));
      for (const u of vecs) { const c = w.reduce((s, x, i) => s + x * u[i], 0); w = w.map((x, i) => x - c * u[i]); }
      const n = Math.hypot(...w); v = w.map(x => x / n);
    }
    vecs.push(v);
  }
  return vecs;
}

const runs = [];
for (let seed = 1; seed <= seeds; seed++) {
  const history = [];                               // history[t][a] = column a of E
  run({ seed, observe: m => {
    const E = m.params.E.data;   // copied below, then turned into displacements
    history.push(Array.from({ length: V }, (_, a) => Array.from(E.subarray(D * a, D * a + D))));
  }});
  const start = history[0];
  for (const cols of history) cols.forEach((e, a) => e.forEach((v, i) => e[i] = v - start[a][i]));   // displacement
  const final = history.at(-1);
  const M = Array.from({ length: D }, (_, i) => Array.from({ length: D }, (_, j) => final.reduce((s, e) => s + e[i] * e[j], 0)));
  const [u, w] = plane(M), dot = (x, y) => x.reduce((s, xi, i) => s + xi * y[i], 0);
  const scale = Math.max(...final.map(e => Math.hypot(dot(e, u), dot(e, w))));
  runs.push(Array.from({ length: V }, (_, a) => history.map(cols => [dot(cols[a], u) / scale, dot(cols[a], w) / scale])));
}

const size = 150, c = size / 2, R = size / 2 - 6;
const paths = runs.flatMap((paths7, r) => paths7.map(p =>
  ({ d: polyline(simplify(p.map(([x, y]) => [c + R * x, c - R * y]))), opacity: 0.35,
     ...(r === 0 ? { color: ink.value, width: stroke.emphasis, opacity: 0.9 } : {}) })));

save(process.argv[2] ?? '/tmp/vectors.svg', svg({ width: size, height: size, paths,
  title: 'The seven embedding vectors, followed through training, in 64 independent runs.',
  desc: `Each path is the displacement E e_a(t) − E e_a(0) ∈ ℝ⁸ of one character's embedding over ${STEPS} steps ` +
        `(${TRAIN}; seeds 1–64), projected onto the plane of the two leading singular vectors of that run's final ` +
        `displacement and scaled so its longest final displacement reaches the rim. In value ink: one run (seed 1).` }));

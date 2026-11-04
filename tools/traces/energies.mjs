// Chapter 3, "Energies and probabilities": the Gibbs distribution p_β ∝ exp(β z) for
// real logit vectors z, followed from β = 0 (the uniform centre) to β → ∞ (a corner).
// The 7-simplex is projected onto the plane by sending vertex a to the a-th corner of a
// heptagon (a linear map, so straight lines stay straight). Every logit column the
// specimen produced at 40 moments during training contributes one path.
import { run } from './specimen.mjs';
import { svg, polyline, simplify, save, ink } from './svg.mjs';

const steps = 700, V = 7;
const opt = { optimizer: 'adamw', lr: 1e-2, clipNorm: 1,
              schedule: { warmupSteps: 20, totalSteps: steps, minLR: 1e-4 } };
const moments = new Set(Array.from({ length: 40 }, (_, k) => Math.round((steps - 1) * (k / 39) ** 2)));
const columns = [];
run({ steps, opt, observe: (m, tr, t) => {
  if (!moments.has(t)) return;
  for (let n = 0; n < tr.Z.shape[1]; n++) columns.push({ z: Float64Array.from(tr.Z.column(n)), late: t > steps / 2 });
}});

const gibbs = (z, β) => {
  const m = Math.max(...z), w = z.map(v => Math.exp(β * (v - m))), s = w.reduce((a, b) => a + b);
  return w.map(v => v / s);
};

export function render({ size = 1000, margin = 80 } = {}) {
  const R = size / 2 - margin, c = size / 2;
  const corner = a => [c + R * Math.sin(2 * Math.PI * a / V), c - R * Math.cos(2 * Math.PI * a / V)];
  const project = p => p.reduce(([x, y], pa, a) => { const [u, v] = corner(a); return [x + pa * u, y + pa * v]; }, [0, 0]);
  const βs = Array.from({ length: 160 }, (_, k) => 10 ** (-2 + 3.4 * k / 159));   // 0.01 … 25
  const paths = columns.map(({ z, late }) => ({
    d: polyline(simplify([[c, c], ...βs.map(β => project(gibbs(z, β)))], 0.1)),
    width: late ? 0.32 : 0.28, opacity: late ? 0.5 : 0.22, stroke: late ? ink.ink : ink.ink,
  }));
  // The simplex itself: seven hairline edges of the heptagon, in accent.
  const ring = Array.from({ length: V + 1 }, (_, a) => corner(a % V));
  paths.unshift({ d: polyline(ring), width: 0.6, opacity: 0.8, stroke: ink.accent });
  return svg({ width: size, height: size, paths });
}

const out = process.argv[2] ?? '/tmp/energies.svg';
save(out, render());
console.log(out, columns.length, 'paths');

// Cover: every parameter of the specimen, followed through one training run.
// Time flows down the page (log scale); a path's horizontal position is its value.
// Accent: the 56 coordinates of the embedding E, where the letters acquire meaning.
import { run } from './specimen.mjs';
import { svg, polyline, simplify, save, ink } from './svg.mjs';

const steps = 700;
const opt = { optimizer: 'adamw', lr: 1e-2, clipNorm: 1,
              schedule: { warmupSteps: 20, totalSteps: steps, minLR: 1e-4 } };
const theta = [];
const model = run({ steps, opt, observe: m => theta.push(m.theta.slice()) });
const p = theta[0].length;
const E = model.layout.find(l => l.name === 'E');
const inE = i => i >= E.offset && i < E.offset + E.size;

let lo = Infinity, hi = -Infinity;
for (const t of theta) for (const v of t) lo = Math.min(lo, v), hi = Math.max(hi, v);

export function render({ width = 900, height = 1260, margin = 70 } = {}) {
  const X = v => margin + (width - 2 * margin) * (v - lo) / (hi - lo);
  const Y = s => margin + (height - 2 * margin) * Math.log1p(s / 8) / Math.log1p((steps - 1) / 8);
  const path = i => polyline(simplify(theta.map((t, s) => [X(t[i]), Y(s)])));
  const paths = [];
  for (let i = 0; i < p; i++) if (!inE(i)) paths.push({ d: path(i), width: 0.28, opacity: 0.42 });
  for (let i = 0; i < p; i++) if (inE(i)) paths.push({ d: path(i), width: 0.5, opacity: 0.9, stroke: ink.accent });
  return svg({ width, height, paths });
}

const out = process.argv[2] ?? '/tmp/cover.svg';
save(out, render());
console.log(out, 'p =', p, 'range', lo.toFixed(3), hi.toFixed(3));

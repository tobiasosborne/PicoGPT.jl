// Chapter 5, "Attention": every arc the specimen drew while it learned.
// The corpus lies along a baseline. For each window, query position t reads key s ≤ t
// with weight A[t,s]; that reading is an arc from s to t, inked in proportion to A.
// An arc's height is span × weight. Head 1 arcs rise above the line, head 2 arcs hang below it. Forty moments of training
// are overlaid, so darkness = weight × persistence.
import { run, corpus } from './specimen.mjs';
import { svg, save, ink } from './svg.mjs';

const steps = 700;
const opt = { optimizer: 'adamw', lr: 1e-2, clipNorm: 1,
              schedule: { warmupSteps: 20, totalSteps: steps, minLR: 1e-4 } };
const moments = new Set(Array.from({ length: 40 }, (_, k) => Math.round((steps - 1) * (k / 39) ** 1.5)));
const readings = [];                       // {h, from, to, w, t}
run({ steps, opt, observe: (m, tr, t) => {
  if (!moments.has(t)) return;
  const A = tr.blocks[0].A, [T, , H, B] = A.shape;
  for (let b = 0; b < B; b++) for (let h = 0; h < H; h++) {
    const M = A.attention(h, b);
    for (let i = 0; i < T; i++) for (let j = 0; j < i; j++)        // j = i is self-reading; skip the loop
      readings.push({ h, from: b + j, to: b + i, w: M.get(i, j), t });
  }
}});

export function render({ width = 1400, height = 760, margin = 70 } = {}) {
  const n = corpus.length, gap = (width - 2 * margin) / (n - 1), base = height / 2;
  const x = k => margin + gap * k;
  const paths = readings.filter(r => r.w > 0.01).map(({ h, from, to, w, t }) => {
    const span = x(to) - x(from), lift = (h === 0 ? -1 : 1) * span * w;   // height ∝ weight
    const mx = (x(from) + x(to)) / 2;
    return { d: `M${x(from).toFixed(2)} ${base}Q${mx.toFixed(2)} ${(base + 2 * lift).toFixed(2)} ${x(to).toFixed(2)} ${base}`,
             width: 0.4, opacity: Math.sqrt(w) * (0.1 + 0.6 * (t / steps) ** 2) };
  });
  // the text itself: one short accent tick per character, a gap for each space
  for (let k = 0; k < n; k++) if (corpus[k] !== ' ')
    paths.push({ d: `M${x(k)} ${base - 5}L${x(k)} ${base + 5}`, width: 1.2, stroke: ink.accent });
  return svg({ width, height, paths });
}

const out = process.argv[2] ?? '/tmp/attention.svg';
save(out, render());
console.log(out, readings.length, 'readings');

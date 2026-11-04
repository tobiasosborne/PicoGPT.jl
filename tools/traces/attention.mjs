// Chapter 5, "Attention": every reading the specimen made while it learned.
// The corpus lies along a baseline. In each window, reader position i reads source j < i
// with weight A[i,j]; that reading is an arc from j to i whose height is span × weight.
// Head 1 arcs rise above the line, head 2 arcs hang below it. Forty moments of training
// are overlaid; later moments are inked more strongly, so the learned pattern emerges
// from the faint record of how it got there.
import { run, TRAIN, corpus, STEPS } from './specimen.mjs';
import { svg, save, ink, stroke } from './svg.mjs';

const moments = new Set(Array.from({ length: 40 }, (_, k) => Math.round((STEPS - 1) * (k / 39) ** 1.5)));
const readings = [];
run({ observe: (m, tr, t) => {
  if (!moments.has(t)) return;
  const A = tr.blocks[0].A, [T, , H, B] = A.shape;
  for (let b = 0; b < B; b++) for (let h = 0; h < H; h++) {
    const M = A.attention(h, b);
    for (let i = 0; i < T; i++) for (let j = 0; j < i; j++)
      readings.push({ h, from: b + j, to: b + i, w: M.get(i, j), t });
  }
}});

const width = 150, height = 84, margin = 5, base = 38;
const gap = (width - 2 * margin) / (corpus.length - 1), x = k => margin + gap * k;
const f = v => v.toFixed(3);

const paths = [];
for (const { h, from, to, w, t } of readings) {
  const opacity = Math.sqrt(w) * (0.1 + 0.6 * (t / STEPS) ** 2);
  if (opacity < 0.04) continue;                       // below the visible threshold: not drawn
  const lift = (h === 0 ? -1 : 1) * (x(to) - x(from)) * w, mid = (x(from) + x(to)) / 2;
  paths.push({ d: `M${f(x(from))} ${base}Q${f(mid)} ${f(base + 2 * lift)} ${f(x(to))} ${base}`, opacity });
}
for (let k = 0; k < corpus.length; k++) if (corpus[k] !== ' ')
  paths.push({ d: `M${f(x(k))} ${base - 1}L${f(x(k))} ${base + 1}`, color: ink.value, width: stroke.emphasis * 1.6 });

save(process.argv[2] ?? '/tmp/attention.svg', svg({ width, height, paths,
  title: 'Every reading the specimen\'s two attention heads made while it learned.',
  desc: `The corpus "${corpus}" lies along the baseline (value ticks mark letters, gaps mark spaces). ` +
        `Reader i reading source j < i with weight A[i,j] is an arc from j to i of height span × weight; ` +
        `head 1 above the line, head 2 below. ${moments.size} moments of training (${TRAIN}), later moments inked more strongly. ` +
        `Head 1 learns short, local readings; head 2 learns to reach far back.` }));

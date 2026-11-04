// Chapter 1, "The task": the chain rule P(x₂…x₈ | x₁) = Πₖ P(xₖ₊₁ | x₁…xₖ) drawn as a tree.
// Every continuation of "t" is a path from the left edge; each string owns an interval of the
// vertical axis whose length is its probability, nested inside its prefix's interval (the
// picture behind arithmetic coding). Edges are inked by probability. Forty moments of
// training are overlaid, so the broad fan of the untrained model collapses, step by step,
// onto the one text that is true.
import { SPECIMEN_CONFIG, createModel, createBatch, forward } from '../../web/src/specimen/index.mjs';
import { run, TRAIN, STEPS, id, vocab } from './specimen.mjs';
import { svg, save, ink, stroke } from './svg.mjs';

const V = vocab.length, depth = SPECIMEN_CONFIG.T, cut = 2e-4;
const moments = new Set(Array.from({ length: 40 }, (_, k) => Math.round((STEPS - 1) * (k / 39) ** 2)));

// All continuations of "t" with probability > cut, level by level: {seq, lo, hi, p}.
function tree(model) {
  let level = [{ seq: [id('t')], lo: 0, hi: 1 }];
  const edges = [];
  for (let k = 1; k < depth && level.length; k++) {
    const batch = createBatch(level.map(n => n.seq));
    const { probs } = forward(model, batch);
    const next = [];
    level.forEach((n, b) => {
      let lo = n.lo;
      for (let a = 0; a < V; a++) {
        const hi = lo + (n.hi - n.lo) * probs.data[V * (k - 1 + k * b) + a];
        if (hi - lo > cut) { const child = { seq: [...n.seq, a + 1], lo, hi }; next.push(child); edges.push([k, n, child]); }
        lo = hi;
      }
    });
    level = next;
  }
  return edges;
}

const snapshots = [];
run({ observe: (m, tr, t) => moments.has(t) && snapshots.push({ t, edges: tree(createModel(SPECIMEN_CONFIG, m.theta)) }) });

const width = 150, height = 100, margin = 5;
const X = k => margin + (width - 2 * margin) * (k - 1) / (depth - 1);
const Y = n => margin + (height - 2 * margin) * (n.lo + n.hi) / 2;
const f = v => v.toFixed(3);
const paths = [];
for (const { t, edges } of snapshots) for (const [k, a, b] of edges) {
  const late = t === STEPS - 1, p = b.hi - b.lo;
  const opacity = late ? Math.min(1, 2 * p) : 1.2 * p ** 0.6;
  if (opacity < 0.04) continue;                       // too faint to see: not drawn
  const x0 = X(k), x1 = X(k + 1), y0 = Y(a), y1 = Y(b), xm = (x0 + x1) / 2;
  paths.push({ d: `M${f(x0)} ${f(y0)}C${f(xm)} ${f(y0)} ${f(xm)} ${f(y1)} ${f(x1)} ${f(y1)}`, opacity,
               ...(late ? { color: ink.value, width: stroke.emphasis } : {}) });
}

save(process.argv[2] ?? '/tmp/task.svg', svg({ width, height, paths,
  title: 'Every continuation of "t" the specimen considers, from untrained to trained.',
  desc: `Columns are positions 1 to ${depth}. Each string's probability P(x₂…xₖ | "t") = Π P(next | prefix) is the length of ` +
        `its own interval of the vertical axis, nested inside its prefix's interval; edges join each string to its ` +
        `one-character extensions, inked in proportion to probability^0.6 (edges too faint to see are omitted). ${moments.size} moments ` +
        `of training (${TRAIN}) are overlaid. Value ink: the final model, which has learned "to be or".` }));

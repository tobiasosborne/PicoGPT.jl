// Chapter 4, "Measuring surprise": the surprise −log p(y) of every prediction the specimen
// makes, followed through training. Most fall towards zero; predictions the text leaves
// genuinely open settle on plateaux at their conditional entropies. Their mean is the
// floor no model can beat.
import { run, TRAIN, STEPS } from './specimen.mjs';
import { svg, polyline, simplify, save, ink, stroke } from './svg.mjs';

const curves = [];                                   // curves[n][t] = −log p(y_n) at step t
run({ observe: (m, { probs }, t, { y }) => {
  const [V, N] = probs.shape;
  for (let n = 0; n < N; n++) (curves[n] ??= []).push(-Math.log(probs.data[V * n + y[n]]));
}});

const width = 150, height = 100, margin = 5, top = Math.max(...curves.map(c => Math.max(...c)));
const X = t => margin + (width - 2 * margin) * Math.log1p(t) / Math.log1p(STEPS - 1);
const Y = s => height - margin - (height - 2 * margin) * Math.sqrt(s / top);   // √-scale keeps the plateaux apart
const paths = curves.map(c => ({ d: polyline(simplify(c.map((s, t) => [X(t), Y(s)]))), opacity: 0.55 }));
const mean = curves[0].map((_, t) => curves.reduce((a, c) => a + c[t], 0) / curves.length);
paths.push({ d: polyline(simplify(mean.map((s, t) => [X(t), Y(s)]))), color: ink.value, width: stroke.emphasis * 1.5 });

save(process.argv[2] ?? '/tmp/surprise.svg', svg({ width, height, paths,
  title: 'The surprise of each of the specimen\'s 80 predictions, as it learns.',
  desc: `Each hairline is −log p(y) for one (context, next character) pair, over ${STEPS} steps of training ` +
        `(${TRAIN}); time runs left to right on a log(1 + step) scale, surprise upward on a square-root scale ` +
        `from 0 to ${top.toFixed(2)} nats. The value-ink line is their mean, the loss, which ends at ` +
        `${mean.at(-1).toFixed(4)} nats: the plateaux are contexts the text itself leaves ambiguous.` }));

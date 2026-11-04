// Chapter 3, "Energies and probabilities": the Gibbs distribution p_β ∝ exp(β z) for
// real logit vectors z, followed from β = 0 (the uniform centre) towards β → ∞ (a corner).
// The simplex is projected onto the plane by sending vertex a to the a-th corner of a
// heptagon — a linear map, so the simplex's straight edges stay straight.
import { run, TRAIN, STEPS, vocab } from './specimen.mjs';
import { svg, polyline, simplify, save, ink, stroke } from './svg.mjs';

const V = vocab.length, moments = 40;
const at = new Set(Array.from({ length: moments }, (_, k) => Math.round((STEPS - 1) * (k / (moments - 1)) ** 2)));
const columns = [];
run({ observe: (m, tr, t) => {
  if (!at.has(t)) return;
  for (let n = 0; n < tr.Z.shape[1]; n++) columns.push({ z: Float64Array.from(tr.Z.column(n)), late: t > STEPS / 2 });
}});

const gibbs = (z, β) => {
  const m = Math.max(...z), w = z.map(v => Math.exp(β * (v - m))), s = w.reduce((a, b) => a + b);
  return w.map(v => v / s);
};

const size = 150, R = size / 2 - 6, c = size / 2;
const corner = a => [c + R * Math.sin(2 * Math.PI * a / V), c - R * Math.cos(2 * Math.PI * a / V)];
const project = p => p.reduce(([x, y], pa, a) => { const [u, v] = corner(a); return [x + pa * u, y + pa * v]; }, [0, 0]);
const βs = Array.from({ length: 160 }, (_, k) => 10 ** (-2 + 3.4 * k / 159));        // 0.01 … 25

const paths = [{ d: polyline(Array.from({ length: V + 1 }, (_, a) => corner(a % V))), color: ink.value, width: stroke.emphasis, opacity: 0.7 }];
for (const { z, late } of columns)
  paths.push({ d: polyline(simplify([[c, c], ...βs.map(β => project(gibbs(z, β)))])), opacity: late ? 0.5 : 0.22 });

save(process.argv[2] ?? '/tmp/energies.svg', svg({ width: size, height: size, paths,
  title: 'The Gibbs distribution of every logit vector the specimen produced, as the temperature falls.',
  desc: `${columns.length} logit columns z ∈ ℝ⁷ (80 per moment, ${moments} moments of training: ${TRAIN}). ` +
        `Each path is p_β = softmax(βz) for β from 0.01 to 25, projected linearly onto the plane with ` +
        `the seven characters ${vocab.map(c => c === ' ' ? '␣' : c).join(' ')} at the corners of the heptagon (value ink, clockwise from the top). ` +
        `Every path starts at the uniform centre; it ends at a corner when one character wins, ` +
        `and on an edge or face when the text is genuinely ambiguous.` }));

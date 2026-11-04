// Cover: every parameter of the specimen, followed through one training run.
// Time flows down the plate (logarithmically); a path's horizontal position is its value.
// Value ink: the 56 coordinates of the embedding E, where the letters acquire meaning.
import { run, TRAIN } from './specimen.mjs';
import { svg, polyline, simplify, save, ink, stroke } from './svg.mjs';

const theta = [];
const model = run({ observe: m => theta.push(m.theta.slice()) });
const steps = theta.length, p = theta[0].length;
const E = model.layout.find(l => l.name === 'E');
const inE = i => i >= E.offset && i < E.offset + E.size;

let lo = Infinity, hi = -Infinity;
for (const t of theta) for (const v of t) lo = Math.min(lo, v), hi = Math.max(hi, v);

const width = 150, height = 200, margin = 6;
const X = v => margin + (width - 2 * margin) * (v - lo) / (hi - lo);
const Y = s => margin + (height - 2 * margin) * Math.log1p(s / 8) / Math.log1p((steps - 1) / 8);
const path = i => polyline(simplify(theta.map((t, s) => [X(t[i]), Y(s)])));

const paths = [];
for (let i = 0; i < p; i++) if (!inE(i)) paths.push({ d: path(i), opacity: 0.42 });
for (let i = 0; i < p; i++) if (inE(i)) paths.push({ d: path(i), color: ink.value, width: stroke.emphasis * 0.8, opacity: 0.85 });

save(process.argv[2] ?? '/tmp/cover.svg', svg({ width, height, paths,
  title: `Every one of the specimen's ${p} parameters, followed through training.`,
  desc: `Time runs down the plate on a logarithmic scale (${steps} AdamW steps, ${TRAIN}); each path's ` +
        `horizontal position is the value of one parameter θᵢ, from ${lo.toFixed(2)} to ${hi.toFixed(2)}. ` +
        `The left family starts near zero; the right family is the 24 LayerNorm gains, which start at exactly 1. ` +
        `In value ink: the 56 coordinates of the embedding E. Eight paths do not visibly move: the key bias b_k, whose gradient vanishes identically; floating-point rounding moves them by about 10⁻¹¹.` }));

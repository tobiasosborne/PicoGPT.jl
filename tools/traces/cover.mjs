// Cover: every parameter of the specimen, followed through one training run.
// Time flows down the plate on a log(1 + step) scale; a path's horizontal position is how far that
// parameter has moved from where it started, θᵢ(t) − θᵢ(0). All paths leave one point:
// the picture is exactly what learning changed.
// Value ink: the 56 coordinates of the embedding E, where the letters acquire meaning.
import { run, TRAIN, OPT } from './specimen.mjs';
import { svg, polyline, simplify, save, ink, stroke } from './svg.mjs';

const theta = [];
const model = run({ observe: m => theta.push(m.theta.slice()) });
const θ0 = theta[0];
for (const t of theta) for (let i = 0; i < t.length; i++) t[i] -= θ0[i];        // displacement
const steps = theta.length, p = theta[0].length;
const E = model.layout.find(l => l.name === 'E');
const inE = i => i >= E.offset && i < E.offset + E.size;

// Horizontal: asinh(Δ/δ), linear for small displacements and logarithmic for large ones,
// with δ the median final displacement, so the bulk and the few long travellers both show.
const finals = theta.at(-1).map(Math.abs).sort((a, b) => a - b), δ = finals[finals.length >> 1];
const sx = v => Math.asinh(v / δ);
let r = 0;
for (const t of theta) for (const v of t) r = Math.max(r, Math.abs(sx(v)));

const width = 150, height = 200, margin = 6;
const X = v => width / 2 + (width / 2 - margin) * sx(v) / r;
const Y = s => margin + (height - 2 * margin) * Math.log1p(s) / Math.log1p(steps - 1);
const path = i => polyline(simplify(theta.map((t, s) => [X(t[i]), Y(s)])));

const paths = [];
for (let i = 0; i < p; i++) if (!inE(i)) paths.push({ d: path(i), opacity: 0.42 });
for (let i = 0; i < p; i++) if (inE(i)) paths.push({ d: path(i), color: ink.value, width: stroke.emphasis * 0.8, opacity: 0.85 });

save(process.argv[2] ?? '/tmp/cover.svg', svg({ width, height, paths,
  title: `Every one of the specimen's ${p} parameters, followed through training.`,
  desc: `Time runs down the plate on a log(1 + step) scale: the book's specimen, from its stored θ₀, over ${steps} steps (${TRAIN}). Each path is one ` +
        `parameter's displacement Δ = θᵢ(t) − θᵢ(0) from its random starting value, placed horizontally at ` +
        `asinh(Δ/δ) with δ = ${δ.toFixed(4)} the median final displacement (linear near 0, logarithmic far out); ` +
        `every path starts on the centre line. The straight roof at the top is Adam's first update, which is sign descent: ` +
        `nearly every parameter moves by the same distance η₁ = ${(OPT.eta / OPT.warmup).toFixed(3)} (weight decay and ε cause the small differences). In value ink: the 56 coordinates ` +
        `of the embedding E. Eight paths do not visibly leave the centre line: the key bias b_k, whose gradient is identically zero.` }));

// Chapter 7, "Pulling back": the cover's twin. The same run, the same plate geometry, but each
// path is a component of the gradient ∇θ rather than of θ. Values spread out and settle;
// covectors start wild and die away to nothing, because at a minimum there is nothing left
// to pull back. Horizontal position is asinh(∇θᵢ / g₀), which is linear near zero and
// logarithmic far from it.
import { run, TRAIN } from './specimen.mjs';
import { svg, polyline, simplify, save, ink, stroke } from './svg.mjs';

// Each component as the optimiser sees it: Adam's first moment m ← β₁m + (1−β₁)∇θ, β₁ = 0.9,
// bias-corrected. (The raw gradient flips sign from step to step and smears into a solid mass.)
const grads = [], β1 = 0.9;
let mom = null;
run({ observe: (m, tr, t) => {
  mom ??= new Float64Array(m.grad.length);
  for (let i = 0; i < mom.length; i++) mom[i] = β1 * mom[i] + (1 - β1) * m.grad[i];
  grads.push(mom.map(v => v / (1 - β1 ** (t + 1))));
}});
const steps = grads.length, p = grads[0].length;

let rms = 0; for (const v of grads[0]) rms += v * v; const g0 = Math.sqrt(rms / p);
const s = v => Math.asinh(v / g0);
let m = 0; for (const g of grads) for (const v of g) m = Math.max(m, Math.abs(s(v)));

const width = 150, height = 200, margin = 6;
const X = v => width / 2 + (width / 2 - margin) * s(v) / m;
const Y = t => margin + (height - 2 * margin) * Math.log1p(t / 8) / Math.log1p((steps - 1) / 8);
const paths = Array.from({ length: p }, (_, i) =>
  ({ d: polyline(simplify(grads.map((g, t) => [X(g[i]), Y(t)]), 0.08)), color: ink.covector, opacity: 0.16 }));

save(process.argv[2] ?? '/tmp/pulling-back.svg', svg({ width, height, paths,
  title: `Every component of the gradient ∇θ ∈ ℝ^${p}, followed through the same training run as the cover.`,
  desc: `Time runs down the plate on a logarithmic scale (${steps} steps, ${TRAIN}). Each path is one component ` +
        `of ∂ℓ/∂θᵢ (before clipping), averaged as Adam averages it (first moment, β₁ = 0.9, bias-corrected), placed horizontally at asinh(∂ℓ/∂θᵢ / g₀) with g₀ = ${g0.toExponential(2)} ` +
        `(the initial root-mean-square component): linear near zero, logarithmic beyond. ` +
        `As the loss reaches its floor every component falls back to zero.` }));

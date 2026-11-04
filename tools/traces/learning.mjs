// Chapter 8, "Learning": the loss of the specimen trained from 120 different seeds with
// plain Adam at a constant learning rate, against the same runs with a cosine schedule.
// Every run reaches the floor; plain Adam does not stay there. Near the minimum its second-
// moment estimate shrinks, the normalised step stays large, and the run is flung off again:
// the slingshot. The schedule shrinks the step: while η is still large the scheduled runs fly
// too, and as η falls the flights stop.
import { run, STEPS, OPT } from './specimen.mjs';
import { svg, polyline, simplify, save, ink, stroke } from './svg.mjs';

const seeds = 48, steps = 2000;
// As in the chapter's figure (fig-slingshot): Adam itself, without decay or clipping.
const base = { optimizer: 'adamw', eta: OPT.eta, beta1: 0.9, beta2: 0.95, eps: 1e-8, weight_decay: 0, clip: Infinity };
const plain = { ...base, schedule: null };                                             // constant η
const sched = { ...base, schedule: 'warmup_cosine', warmup: 20, steps, eta_min: OPT.eta / 10 };
const record = opt => Array.from({ length: seeds }, (_, seed) => {
  const L = []; run({ seed: seed + 1, steps, opt, observe: (m, tr) => L.push(tr.loss) }); return L; });
const runs = { plain: record(plain), sched: record(sched) };

const width = 150, height = 100, margin = 5;
const lo = Math.log(0.05), hi = Math.log(2.2);
const X = t => margin + (width - 2 * margin) * t / (steps - 1);
const Y = l => height - margin - (height - 2 * margin) * (Math.log(l) - lo) / (hi - lo);
const draw = (L, o) => ({ d: polyline(simplify(L.map((l, t) => [X(t), Y(l)]))), ...o });
const paths = [
  ...runs.plain.map(L => draw(L, { opacity: 0.2 })),
  ...runs.sched.map(L => draw(L, { color: ink.value, opacity: 0.3 })),
];
// A jump: the loss more than doubles in one step, after the run has first reached the floor.
const jumps = L => L.map((l, t) => t > 300 && l > 2 * L[t - 1] ? t : -1).filter(t => t >= 0);
const count = rs => rs.reduce((n, L) => n + jumps(L).length, 0);
const last = rs => Math.max(...rs.flatMap(jumps));

save(process.argv[2] ?? '/tmp/learning.svg', svg({ width, height, paths,
  title: `The specimen trained from ${seeds} seeds: Adam at a constant rate keeps leaving the floor; as a cosine schedule shrinks the rate, the flights stop.`,
  desc: `Loss (logarithmic, 0.05 to 2.2 nats) against step (linear, 0 to ${steps}). Ink: Adam (β₂ 0.95, no weight decay, no clipping) at constant η = ${OPT.eta}. ` +
        `Value ink: the same with 20 warmup steps and cosine decay to ${OPT.eta / 10}. Fresh draws of θ₀ from the book's initial distributions, JS seeds 1–${seeds}. ` +
        `After step 300 the constant-rate runs make ${count(runs.plain)} jumps in which the loss more than doubles in one step, ` +
        `the last at step ${last(runs.plain)}; the scheduled runs make ${count(runs.sched)}, the last at step ${last(runs.sched)}, ` +
        `when η has fallen to ${(OPT.eta / 10 + (OPT.eta - OPT.eta / 10) * (1 + Math.cos(Math.PI * (last(runs.sched) - 20) / (steps - 20))) / 2).toExponential(1)}.` }));

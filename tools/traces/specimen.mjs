// The specimen, its corpus, and a recorder for training runs.
import { SPECIMEN_CONFIG, createModel, createBatch, init, step, forward, backward }
  from '../../web/src/specimen/index.mjs';

export const corpus = 'to be or not to be';
export const vocab = [...new Set(corpus)].sort();            // ' ' b e n o r t
export const id = c => vocab.indexOf(c) + 1;                  // 1-based, as in Julia

// Every length-T window with a complete next-character target.
export function windows(T = SPECIMEN_CONFIG.T) {
  const x = [], y = [];
  for (let s = 0; s + T < corpus.length; s++) {
    x.push([...corpus.slice(s, s + T)].map(id));
    y.push([...corpus.slice(s + 1, s + T + 1)].map(id));
  }
  return createBatch(x, y);
}

// The run every trace records: AdamW, warmup + cosine, clipped. Seed 0.
export const STEPS = 700;
export const OPT = { optimizer: 'adamw', lr: 1e-2, clipNorm: 1,
                     schedule: { warmupSteps: 20, totalSteps: STEPS, minLR: 1e-4 } };
export const TRAIN = 'lr 1e-2, 20 warmup steps, cosine to 1e-4, clip 1, seed 0';

// Train from seed, calling observe(model, trace, t, batch) before every update.
export function run({ seed = 0, steps = STEPS, opt = OPT, observe = () => {} } = {}) {
  const model = createModel(SPECIMEN_CONFIG, init(SPECIMEN_CONFIG, seed));
  const batch = windows();
  for (let t = 0; t < steps; t++) {
    const trace = forward(model, batch);
    backward(model, trace);
    observe(model, trace, t, batch);
    step(model, opt);                 // applies the gradient just computed
  }
  return model;
}

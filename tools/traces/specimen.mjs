// The specimen, its corpus, and a recorder for training runs.
import { SPECIMEN_CONFIG, createModel, createBatch, init, trainStep, forward, backward }
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

export const ADAM = { optimizer: 'adamw', lr: 1e-2, clipNorm: 1 };

// Train from seed, calling observe(model, trace, t) before every update.
export function run({ seed = 0, steps = 600, opt = ADAM, observe = () => {} } = {}) {
  const model = createModel(SPECIMEN_CONFIG, init(SPECIMEN_CONFIG, seed));
  const batch = windows();
  for (let t = 0; t < steps; t++) {
    const trace = forward(model, batch);
    backward(model, trace);
    observe(model, trace, t);
    trainStep(model, batch, opt);
  }
  return model;
}

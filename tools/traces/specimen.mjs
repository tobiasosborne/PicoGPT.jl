// The specimen, its corpus, and a recorder for training runs.
import { readFileSync } from 'node:fs';
import { SPECIMEN_CONFIG, SPECIMEN_RECIPE, createModel, createBatch, init, step, forward, backward }
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

// The run every trace records: the book's canonical specimen recipe (BRIEF §13).
export const OPT = SPECIMEN_RECIPE;
export const STEPS = OPT.steps;
export const TRAIN = `AdamW, η ${OPT.eta} → ${OPT.eta_min} (warmup ${OPT.warmup}, cosine), β₂ ${OPT.beta2}, ` +
                     `weight decay ${OPT.weight_decay} on matrices, clip ${OPT.clip}`;

// The book's specimen starts from the θ₀ that Julia drew and stored in artifacts/specimen.json;
// run({ seed: BOOK }) replays exactly the run the book describes. A numeric seed instead draws a
// fresh θ₀ from the same distributions with the JS engine's own generator.
export const BOOK = 'book';
const bookθ0 = Float64Array.from(JSON.parse(readFileSync(new URL('../../artifacts/specimen.json', import.meta.url), 'utf8')).theta0);

// Train, calling observe(model, trace, t, batch) before every update.
export function run({ seed = BOOK, steps = STEPS, opt = OPT, observe = () => {} } = {}) {
  const model = createModel(SPECIMEN_CONFIG, seed === BOOK ? bookθ0 : init(SPECIMEN_CONFIG, seed));
  const batch = windows();
  for (let t = 0; t < steps; t++) {
    const trace = forward(model, batch);
    backward(model, trace);
    observe(model, trace, t, batch);
    step(model, opt);                 // applies the gradient just computed
  }
  return model;
}

import { createBatch } from './index.mjs';

export const corpus = 'to be or not to be';
export const vocab = [' ', 'b', 'e', 'n', 'o', 'r', 't'];

/** Every complete next-token pair. Optionally repeat windows to measure a chosen B. */
export function specimenBatch(batchSize = undefined) {
  const ids = Array.from(corpus, c => vocab.indexOf(c) + 1), T = 8;
  const windows = ids.length - T, B = batchSize ?? windows;
  const x = [], y = [];
  for (let b = 0; b < B; b++) {
    const start = b % windows;
    x.push(ids.slice(start, start + T)); y.push(ids.slice(start + 1, start + T + 1));
  }
  return createBatch(x, y);
}

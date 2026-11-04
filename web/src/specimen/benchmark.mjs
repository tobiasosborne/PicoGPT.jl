import { performance } from 'node:perf_hooks';
import { createModel, SPECIMEN_CONFIG, trainStep } from './index.mjs';
import { specimenBatch } from './fixtures.mjs';

const model = createModel(SPECIMEN_CONFIG), batch = specimenBatch(11);
const opt = { optimizer: 'adamw', lr: 1e-3, weightDecay: 0.01, clipNorm: 1 };
const steps = Number(process.argv[2] ?? 3000), rounds = 5;
if (!Number.isSafeInteger(steps) || steps < 1) throw new RangeError('Steps must be a positive integer');
for (let i = 0; i < 1500; i++) trainStep(model, batch, opt);
const times = [];
for (let r = 0; r < rounds; r++) {
  const start = performance.now();
  for (let i = 0; i < steps; i++) trainStep(model, batch, opt);
  times.push((performance.now() - start) * 1000 / steps);
}
times.sort((a, b) => a - b);
console.log(`Float64 specimen: ${model.theta.length} parameters, B=${batch.B}, T=${batch.T}, N=${batch.N}`);
console.log('11 benchmark sequences: 10 complete next-token windows plus a repeated first window.');
console.log(`Full forward + backward + clipped AdamW: median ${times[2].toFixed(1)} µs/step; min ${times[0].toFixed(1)}, max ${times[4].toFixed(1)} (${steps} steps × ${rounds} rounds, Node ${process.version})`);

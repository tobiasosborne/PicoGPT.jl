import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { createModel, createBatch, forward, backward, trainStep, initial, init, drawTheta } from './index.mjs';
import { makeSpecimen } from '../lib/model/specimen.js';

const artifact = new URL('../../../artifacts/specimen.json', import.meta.url);

function scalar(actual, expected, tolerance, path) {
  assert.ok(Number.isFinite(actual) && Math.abs(actual - expected) <= tolerance,
    `${path}: JS=${actual}, Julia=${expected}, absolute error=${Math.abs(actual - expected)}, limit=${tolerance}`);
}
function vector(actual, expected, tolerance, path) {
  assert.equal(actual.length, expected.length, `${path}: length`);
  for (let i = 0; i < expected.length; i++) scalar(actual[i], expected[i], tolerance, `${path}[${i}]`);
}
function trace(actual, expected, path = 'forward') {
  if (typeof expected === 'number') return scalar(actual, expected, 1e-12, path);
  if (expected.shape && expected.data) {
    assert.deepEqual(Array.from(actual.shape), expected.shape, `${path}: shape`);
    return vector(actual.data, expected.data, 1e-12, path);
  }
  for (const key of Object.keys(expected)) {
    assert.ok(actual[key] !== undefined, `Missing ${path}.${key}`);
    trace(actual[key], expected[key], `${path}.${key}`);
  }
}

test('Julia specimen: layout, P, EVERY forward array, ∇θ, and full training trajectory', {
  skip: !existsSync(artifact) && 'artifacts/specimen.json is absent; run the Julia specimen exporter to enable conformance.',
}, t => {
  const ref = JSON.parse(readFileSync(artifact, 'utf8'));
  const model = createModel(ref.config, ref.theta0), batch = createBatch(ref.batch.x, ref.batch.y);
  assert.equal(model.layout.length, ref.layout.length);
  for (let i = 0; i < ref.layout.length; i++) {
    assert.equal(model.layout[i].offset, ref.layout[i].offset, `layout[${i}]: offset`);
    assert.deepEqual(model.layout[i].shape, ref.layout[i].shape, `layout[${i}]: shape`);
    assert.deepEqual(initial(model.layout[i].name, ref.config),
      { mean: ref.init.params[i].mean, std: ref.init.params[i].std }, `init[${i}]`);
  }
  vector(init(ref.config, 42), drawTheta(model.layout, ref.init.params, 42), 0, 'init distributions/draws');
  trace(model.P, ref.P, 'P');
  trace(forward(model, batch), ref.forward);
  vector(backward(model).theta, ref.grad, 1e-12, 'grad');
  const hyper = ref.train.hyper;
  const opt = { ...hyper, steps: ref.train.steps, optimizer: ref.train.optimizer };
  if (hyper.betas) [opt.beta1, opt.beta2] = hyper.betas;
  assert.equal(ref.train.loss.length, ref.train.steps + 1, 'loss at theta0 and after every update');
  let maxLossError = 0;
  for (let k = 0; k <= ref.train.steps; k++) {
    const loss = k < ref.train.steps ? trainStep(model, batch, opt).loss : forward(model, batch).loss;
    scalar(loss, ref.train.loss[k], 1e-9, `train.loss[${k}] (after ${k} updates)`);
    maxLossError = Math.max(maxLossError, Math.abs(loss - ref.train.loss[k]));
  }
  vector(model.theta, ref.train.theta_final, 1e-9, 'train.theta_final');
  const maxThetaError = Math.max(...model.theta.map((x, i) => Math.abs(x - ref.train.theta_final[i])));

  // The figures must replay the very same engine run, preserving their public API.
  const specimen = makeSpecimen({ config: ref.config, theta0: ref.theta0, theta: ref.train.theta_final,
    init: ref.init, vocab: ref.vocab, corpus: ref.corpus, loss: ref.train.loss,
    provenance: { hyper, steps: ref.train.steps } });
  vector(specimen.lab.initTheta(42), init(ref.config, 42), 0, 'lab.initTheta');
  const replay = specimen.trajectory();
  assert.ok(replay, 'figure trajectory conforms');
  assert.equal(replay.loss.length, ref.train.steps + 1);
  vector(replay.thetas.at(-1), model.theta, 0, 'figure/engine theta_final');
  t.diagnostic(`Conformed: ${model.theta.length} θ entries, all forward tensors and ∇θ ≤ 1e-12; ${ref.train.steps + 1} losses, max error ${maxLossError}; θ_final max error ${maxThetaError} (limits 1e-9).`);
});

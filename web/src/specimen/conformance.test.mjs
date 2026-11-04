import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { createModel, createBatch, forward, backward, trainStep } from './index.mjs';

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
  }
  trace(model.P, ref.P, 'P');
  trace(forward(model, batch), ref.forward);
  vector(backward(model).theta, ref.grad, 1e-12, 'grad');
  const hyper = ref.train.hyper;
  const opt = { ...hyper, optimizer: ref.train.optimizer };
  if (hyper.betas) [opt.beta1, opt.beta2] = hyper.betas;
  assert.equal(ref.train.loss.length, ref.train.steps, 'one pre-update loss per training step');
  for (let i = 0; i < ref.train.steps; i++) scalar(trainStep(model, batch, opt).loss, ref.train.loss[i], 1e-9, `train.loss[${i}]`);
  vector(model.theta, ref.train.theta_final, 1e-9, 'train.theta_final');
  t.diagnostic(`Conformed: ${model.theta.length} θ entries, all forward tensors, ∇θ ≤ 1e-12; ${ref.train.steps} training steps ≤ 1e-9.`);
});

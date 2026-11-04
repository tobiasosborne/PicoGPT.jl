import test from 'node:test';
import assert from 'node:assert/strict';
import { Tensor, createModel, createBatch, SPECIMEN_CONFIG, parameterLayout, initial, init, seededRandom,
  positionalEncoding, forward, backward, step, trainStep, clipGlobalNorm, globalNorm,
  learningRate, sample, generate, resetOptimizer } from './index.mjs';
import { specimenBatch } from './fixtures.mjs';
import { Run, scheduleRate } from '../lib/model/learning.js';

function close(a, b, atol = 1e-12, message = '') {
  assert.ok(Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= atol,
    `${message}: ${a} vs ${b}, error ${Math.abs(a - b)} > ${atol}`);
}
function arrays(a, b, tol = 1e-12) {
  assert.equal(a.length, b.length);
  for (let i = 0; i < a.length; i++) close(a[i], b[i], tol, `index ${i}`);
}
function nodes(trace) {
  const out = [['X0', trace.X0]];
  trace.blocks.forEach((block, l) => Object.entries(block).forEach(([key, tensor]) => out.push([`blocks.${l}.${key}`, tensor])));
  return out.concat(['Xf', 'Z', 'probs'].map(key => [key, trace[key]]));
}

test('θ order, column-major named subarray views, and positional convention', () => {
  const m = createModel();
  const sizes = m.layout.map(v => v.size);
  assert.equal(m.theta.length, sizes.reduce((a, b) => a + b, 0));
  assert.deepEqual(m.layout.map(v => v.name), ['E', ...[
    'gamma1', 'beta1', 'Wq', 'bq', 'Wk', 'bk', 'Wv', 'bv', 'Wo', 'bo',
    'gamma2', 'beta2', 'W1', 'b1', 'W2', 'b2'].map(k => `blocks.0.${k}`), 'gammaf', 'betaf', 'Wout', 'bout']);
  let offset = 0;
  for (const entry of parameterLayout(SPECIMEN_CONFIG)) { assert.equal(entry.offset, offset); offset += entry.size; }
  assert.equal(m.params.E.data.buffer, m.theta.buffer);
  m.params.E.data[9] = 3.25; assert.equal(m.theta[9], 3.25); assert.equal(m.params.E.get(1, 1), 3.25);
  const P = positionalEncoding(5, 4);
  for (let t = 0; t < 4; t++) for (let r = 0; r < 5; r++) {
    const phase = t / 10000 ** (2 * Math.floor(r / 2) / 5);
    assert.equal(P.get(r, t), r % 2 ? Math.cos(phase) : Math.sin(phase));
  }
  assert.throws(() => createModel({ ...SPECIMEN_CONFIG, H: 3 }));
  assert.throws(() => createModel(SPECIMEN_CONFIG, [1]));
  assert.throws(() => createBatch([[1.5]], [[1]]));
  assert.throws(() => forward(m, [[0]], [[1]]));
});

test('seeded initialization, parameter groups and residual matrix scaling', () => {
  arrays(init(SPECIMEN_CONFIG, 41), init(SPECIMEN_CONFIG, 41), 0);
  assert.notDeepEqual(init(SPECIMEN_CONFIG, 41), init(SPECIMEN_CONFIG, 42));
  const scaled = createModel(SPECIMEN_CONFIG, init(SPECIMEN_CONFIG, 41));
  const unscaled = createModel(SPECIMEN_CONFIG, init(SPECIMEN_CONFIG, 41, { residualScaling: false }));
  for (const entry of scaled.layout) {
    const name = entry.name.split('.').at(-1);
    for (let j = 0; j < entry.size; j++) {
      const i = entry.offset + j;
      if (name === 'Wo' || name === 'W2') close(scaled.theta[i], unscaled.theta[i] / Math.sqrt(2), 4e-16);
      else close(scaled.theta[i], unscaled.theta[i], 0);
      if (name.startsWith('gamma')) assert.equal(scaled.theta[i], 1);
      if (name.startsWith('beta') || name.startsWith('b')) assert.equal(scaled.theta[i], 0);
    }
  }
  const c = { D: 4, F: 7, L: 2 };
  for (const [name, mean, std] of [['E', 0, 1], ['Wq', 0, 0.5], ['Wk', 0, 0.5],
    ['Wv', 0, 0.5], ['W1', 0, 0.5], ['Wo', 0, 0.25], ['W2', 0, 1 / Math.sqrt(28)],
    ['Wout', 0, 0.25], ['gamma1', 1, 0], ['gamma2', 1, 0], ['gammaf', 1, 0],
    ['beta1', 0, 0], ['beta2', 0, 0], ['betaf', 0, 0], ['b1', 0, 0], ['bout', 0, 0]]) {
    assert.deepEqual(initial(name, c), { mean, std }, name);
  }
});

for (const [name, config, batch] of [
  ['specimen, full batch', SPECIMEN_CONFIG, specimenBatch(11)],
  ['two blocks, nonstandard F', { V: 4, D: 4, H: 2, F: 7, L: 2, T: 3 }, createBatch([[1, 2, 3], [4, 3, 1]], [[2, 3, 4], [3, 1, 2]])],
]) {
  test(`finite differences: every θ entry (${name})`, () => {
    const m = createModel(config, init(config, 42));
    forward(m, batch); const analytical = backward(m).theta.slice();
    const h = 1e-6;
    for (let i = 0; i < m.theta.length; i++) {
      const value = m.theta[i];
      m.theta[i] = value + h; const plus = forward(m, batch).loss;
      m.theta[i] = value - h; const minus = forward(m, batch).loss;
      m.theta[i] = value;
      close(analytical[i], (plus - minus) / (2 * h), 1e-8, `θ[${i}]`);
    }
  });
}

test('finite differences: EVERY entry of EVERY intermediate adjoint, including pre-mask S and auxiliary probs', () => {
  const config = { V: 3, D: 4, H: 2, F: 7, L: 2, T: 3 };
  const batch = createBatch([[1, 2, 3], [2, 3, 1]], [[2, 3, 1], [3, 1, 2]]);
  const m = createModel(config, init(config, 19));
  const f = forward(m, batch), bars = nodes(backward(m)).map(([name, tensor]) => [name, tensor.data.slice()]);
  let path = '', index = 0, delta = 0;
  const options = { after(name, tensor) { if (name === path) tensor.data[index] += delta; } };
  for (const [name, values] of bars) {
    path = name;
    for (index = 0; index < values.length; index++) {
      delta = 1e-6; const plus = forward(m, batch, null, options).loss;
      delta = -1e-6; const minus = forward(m, batch, null, options).loss;
      close(values[index], (plus - minus) / 2e-6, 2e-8, `${path}[${index}]`);
    }
  }
  assert.equal(f, forward(m, batch), 'trace object is reused');
});

test('arbitrary covectors: probs softmax pullback and a score seed above the causal mask', () => {
  const c = { V: 3, D: 4, H: 2, F: 5, L: 1, T: 3 };
  const m = createModel(c, init(c, 7)), batch = createBatch([[1, 2, 3]]);
  const pseed = new Tensor([3, 3], Float64Array.from({ length: 9 }, (_, i) => Math.sin(i)));
  const sseed = new Tensor([3, 3, 2, 1]); sseed.data[3] = 0.7;
  const objective = f => {
    let value = f.blocks[0].S.data[3] * 0.7;
    for (let i = 0; i < 9; i++) value += f.probs.data[i] * pseed.data[i];
    return value;
  };
  forward(m, batch);
  const grad = backward(m, m.trace, { probs: pseed, blocks: [{ S: sseed }] }).theta.slice();
  for (let i = 0; i < m.theta.length; i++) {
    const v = m.theta[i];
    m.theta[i] = v + 1e-6; const plus = objective(forward(m, batch));
    m.theta[i] = v - 1e-6; const minus = objective(forward(m, batch));
    m.theta[i] = v;
    close(grad[i], (plus - minus) / 2e-6, 1e-8, `VJP θ[${i}]`);
  }
});

test('causality and independence of sequences; attention is row-stochastic and lower-triangular', () => {
  const m = createModel({ ...SPECIMEN_CONFIG, L: 2 });
  const x = [[1, 2, 3, 4, 5, 6, 7, 1], [7, 6, 5, 4, 3, 2, 1, 7]], y = x;
  const f = forward(m, x, y), baseline = f.Z.data.slice();
  for (const block of f.blocks) for (let b = 0; b < 2; b++) for (let h = 0; h < 2; h++) {
    const A = block.A.attention(h, b);
    assert.equal(A, block.A.attention(h, b));
    for (let i = 0; i < 8; i++) {
      let sum = 0;
      for (let j = 0; j < 8; j++) { sum += A.get(i, j); if (j > i) assert.equal(A.get(i, j), 0); }
      close(sum, 1, 5e-16);
    }
  }
  const changed = x.map(row => row.slice());
  changed[0][5] = 1; changed[0][6] = 1; changed[0][7] = 2;
  changed[1].fill(3);
  forward(m, changed, y);
  arrays(f.Z.data.subarray(0, 5 * 7), baseline.subarray(0, 5 * 7), 0);
  assert.notEqual(f.Z.data[7 * 7], baseline[7 * 7]);
  const bars = backward(m);
  for (const block of bars.blocks) for (let b = 0; b < 2; b++) for (let h = 0; h < 2; h++) {
    const dS = block.S.attention(h, b);
    for (let i = 0; i < 8; i++) for (let j = i + 1; j < 8; j++) assert.equal(dS.get(i, j), 0);
  }
});

test('stable log-sum-exp loss and its exact p − e_y gradient', () => {
  const m = createModel({ V: 3, D: 2, H: 1, F: 4, L: 0, T: 1 });
  m.params.bout.data.set([1000, -1000, 500]);
  const f = forward(m, [[1]], [[2]]);
  assert.ok(f.loss > 1999 && f.loss < 2001);
  const g = backward(m);
  arrays(g.Z.data, [1, -1, Math.exp(-500)], 1e-200);
  assert.ok(g.probs.data.every(v => v === 0));
  assert.throws(() => backward(m, forward(m, [[1]])), /target/);
});

test('global clipping before moments, GD, AdamW bias correction, and matrix-only decay', () => {
  arrays([globalNorm(Float64Array.of(3, 4)), globalNorm(Float64Array.of(1e300, 1e300))], [5, Math.sqrt(2) * 1e300], 0);
  const v = Float64Array.of(3, 4); assert.equal(clipGlobalNorm(v, 1), 5); arrays(v, [0.6, 0.8]);
  clipGlobalNorm(v, 0); arrays(v, [0, 0]);
  const m = createModel(); m.theta.fill(2); m.grad[0] = 3; m.grad[1] = 4;
  step(m, { optimizer: 'gd', lr: 0.2, weightDecay: 0.1, clipNorm: 1 });
  close(m.theta[0], 1.84); close(m.theta[1], 1.8); close(m.theta[2], 1.96);
  for (const entry of m.layout) if (entry.shape.length === 1) {
    assert.ok(m.theta.subarray(entry.offset, entry.offset + entry.size).every(x => x === 2), `${entry.name} must not decay`);
  }
  resetOptimizer(m); m.theta.fill(2); m.grad.fill(0);
  const options = { lr: 0.02, beta1: 0.8, beta2: 0.9, eps: 1e-6, weightDecay: 0.1 };
  let theta = 2, mean = 0, variance = 0;
  for (let t = 1; t <= 3; t++) {
    const g = t % 2 === 0 ? -0.4 : 0.7;
    m.grad.fill(0); m.grad[0] = g;
    mean = 0.8 * mean + 0.2 * g; variance = 0.9 * variance + 0.1 * g * g;
    theta -= 0.02 * (mean / (1 - 0.8 ** t) / (Math.sqrt(variance / (1 - 0.9 ** t)) + 1e-6) + 0.1 * theta);
    step(m, options); close(m.theta[0], theta);
  }
  close(m.theta[1], 2 * 0.998 ** 3);
  for (const entry of m.layout) if (entry.shape.length === 1) {
    assert.ok(m.theta.subarray(entry.offset, entry.offset + entry.size).every(x => x === 2), `${entry.name} must not decay`);
  }
  resetOptimizer(m); m.grad.fill(0); m.grad[0] = 3; m.grad[1] = 4;
  const result = step(m, { eta: 0.02, clip: 1, weight_decay: 0.1 });
  close(result.gradNorm, 5); close(result.clipScale, 0.2);
  arrays(m.grad.subarray(0, 2), [0.6, 0.8]);
  arrays(m.optimizer.m.subarray(0, 2), [0.06, 0.08]);
  arrays(m.optimizer.v.subarray(0, 2), [0.05 * 0.6 ** 2, 0.05 * 0.8 ** 2]); // default β2 = 0.95
  assert.throws(() => step(m, { beta1: 1 }));

  // Figure API: retain raw ∇θ, share optimizer buffers and record the actual update.
  const batch = specimenBatch(), theta0 = init(SPECIMEN_CONFIG, 19);
  for (const optimizer of ['adamw', 'gd']) {
    const opts = { optimizer, eta: 0.02, clip: 0.01, weightDecay: 0.01,
      schedule: { warmup: 2, total: 3, etaMin: 0.002 } };
    const run = new Run(SPECIMEN_CONFIG, theta0, batch, opts), engine = createModel(SPECIMEN_CONFIG, theta0);
    const moments = run.m, variance = run.v, delta = run.delta;
    for (let k = 0; k < 3; k++) {
      const loss = run.gradient(), raw = run.grad.slice();
      const r = trainStep(engine, batch, { optimizer, eta: 0.02, clip: 0.01, weight_decay: 0.01,
        warmup: 2, steps: 3, eta_min: 0.002 });
      close(run.step(), loss, 0);
      arrays(run.theta, engine.theta, 0); arrays(run.grad, raw, 0);
      arrays(run.m, engine.optimizer.m, 0); arrays(run.v, engine.optimizer.v, 0);
      arrays(run.delta, engine.optimizer.delta, 0);
      close(run.gradNorm[k], globalNorm(raw), 0); close(run.updateNorm[k], r.updateNorm, 0);
      close(run.rate(k + 1), r.lr, 0); assert.equal(run.lr[k], r.lr);
      assert.equal(run.t, k + 1); assert.equal(run.clipped[k], true);
    }
    assert.equal(run.m, moments); assert.equal(run.v, variance); assert.equal(run.delta, delta);
    assert.equal(run.m, run.model.optimizer.m); assert.equal(run.mask, run.model.optimizer.mask);
    assert.equal(run.steps(0), NaN);
    const raw = run.grad.slice(); run.currentLoss(); arrays(run.grad, raw, 0);
  }
});

test('warmup/cosine schedule and allocation-free result/trace reuse', () => {
  const s = { lr: 0.1, warmupSteps: 2, totalSteps: 6, minLR: 0.01 };
  close(learningRate(1, s), 0.05); close(learningRate(2, s), 0.1);
  close(learningRate(4, s), 0.055); close(learningRate(6, s), 0.01); close(learningRate(20, s), 0.01);
  const canonical = { eta: 0.1, warmup: 2, steps: 6, eta_min: 0.01 };
  for (const t of [1, 2, 3, 4, 6, 20]) {
    close(learningRate(t, canonical), learningRate(t, s), 0);
    close(scheduleRate(t, { eta: 0.1, warmup: 2, total: 6, etaMin: 0.01 }), learningRate(t, canonical), 0);
  }
  close(learningRate(1, { ...canonical, schedule: null }), 0.1);
  const m = createModel(), b = specimenBatch();
  const r = trainStep(m, b, { lr: 0.1, schedule: s }), f = m.trace, grad = backward(m), theta = m.theta;
  const { m: moments, v: variance, mask, delta } = m.optimizer;
  assert.equal(trainStep(m, b, { lr: 0.1, schedule: s }), r);
  assert.equal(r.lr, 0.1); assert.equal(m.trace, f); assert.equal(backward(m), grad); assert.equal(m.theta, theta);
  assert.equal(m.optimizer.m, moments); assert.equal(m.optimizer.v, variance);
  assert.equal(m.optimizer.mask, mask); assert.equal(m.optimizer.delta, delta);
});

test('the full specimen learns, and sampling is seeded and temperature-dependent', () => {
  const m = createModel(), b = specimenBatch(), before = forward(m, b).loss;
  for (let i = 0; i < 200; i++) trainStep(m, b, { lr: 0.01, clipNorm: 1 });
  assert.ok(forward(m, b).loss < before / 2);
  const r1 = seededRandom(7), r2 = seededRandom(7), logits = [-1, 0, 1];
  for (let i = 0; i < 40; i++) assert.equal(sample(logits, { rng: r1 }), sample([99, 100, 101], { rng: r2 }));
  assert.equal(sample(logits, { temperature: 0 }), 3);
  assert.equal(sample([1, 1], { temperature: 0, tokenBase: 0 }), 0);
  assert.equal(sample([-1, 0, 1], { temperature: Number.MIN_VALUE, rng: () => 0 }), 3);
  arrays(generate(m, [1, 2], 10, { rng: seededRandom(123) }), generate(m, [1, 2], 10, { rng: seededRandom(123) }), 0);
});

// specimen-data.js — where the book's specimen numbers come from (server side).
//
// 1. artifacts/specimen.json (written by Julia, BRIEF §5) — canonical. We use its
//    theta0, train.theta_final and train.loss verbatim.
// 2. Until that file exists: the JS engine (web/src/specimen, conformance-tested
//    against Julia) trains the specimen with the provisional recipe below. Pages
//    say so (provenance.kind === 'provisional'), so no number masquerades as final.

import fs from 'node:fs';
import path from 'node:path';
import { createModel, createBatch, init, trainStep, SPECIMEN_CONFIG } from '#specimen';
import { ROOT } from './book.js';

export const CORPUS = 'to be or not to be';
export const VOCAB = [' ', 'b', 'e', 'n', 'o', 'r', 't'];

const PROVISIONAL = { seed: 0, optimizer: 'adamw', lr: 3e-3, weightDecay: 0, clipNorm: 1, steps: 2000 };

let cached = null;
let cachedKey = '';

function artifactPath() { return path.join(ROOT, 'artifacts', 'specimen.json'); }

/** Plain-JSON specimen description shared by every page (≈ 30 kB). */
export function specimenData() {
	const p = artifactPath();
	const key = fs.existsSync(p) ? `artifact:${fs.statSync(p).mtimeMs}` : 'provisional';
	if (cached && cachedKey === key) return cached;
	cachedKey = key;
	cached = key === 'provisional' ? provisional() : fromArtifact(p);
	return cached;
}

function fromArtifact(p) {
	const a = JSON.parse(fs.readFileSync(p, 'utf8'));
	const train = a.train ?? {};
	return {
		config: { ...SPECIMEN_CONFIG, ...a.config },
		vocab: a.vocab ?? VOCAB,
		corpus: a.corpus ?? CORPUS,
		theta0: a.theta0,
		theta: train.theta_final ?? a.theta0,
		loss: train.loss ?? [],
		init: a.init ?? null, // the initial distributions (chapter 8 redraws θ₀ for other seeds)
		provenance: {
			kind: 'artifact',
			source: 'artifacts/specimen.json',
			optimizer: train.optimizer ?? null,
			hyper: train.hyper ?? null,
			steps: train.steps ?? (train.loss?.length ?? 0),
			floor: train.entropy_floor ?? null,
			greedy: train.greedy_from_to_be_or ?? null
		}
	};
}

function provisional() {
	const config = SPECIMEN_CONFIG;
	const ids = Array.from(CORPUS, (c) => VOCAB.indexOf(c) + 1);
	const x = [], y = [];
	for (let s = 0; s + config.T < ids.length; s++) { x.push(ids.slice(s, s + config.T)); y.push(ids.slice(s + 1, s + config.T + 1)); }
	const batch = createBatch(x, y);
	const theta0 = init(config, PROVISIONAL.seed);
	const model = createModel(config, theta0);
	const opt = { optimizer: PROVISIONAL.optimizer, lr: PROVISIONAL.lr, weightDecay: PROVISIONAL.weightDecay, clipNorm: PROVISIONAL.clipNorm };
	const loss = [];
	for (let i = 0; i < PROVISIONAL.steps; i++) loss.push(trainStep(model, batch, opt).loss);
	return {
		config: { ...config },
		vocab: VOCAB,
		corpus: CORPUS,
		theta0: Array.from(theta0),
		theta: Array.from(model.theta),
		loss,
		provenance: { kind: 'provisional', source: 'JS engine, provisional recipe', hyper: PROVISIONAL, steps: PROVISIONAL.steps }
	};
}

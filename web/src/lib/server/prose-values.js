// The handful of prose readings that depend on model data, expanded before Pandoc.
// Shapes and worked mathematical examples remain explicit in the chapter source.
import fs from 'node:fs';
import path from 'node:path';
import { parameterLayout } from '#specimen';

export function proseValues(root) {
	const read = (file) => JSON.parse(fs.readFileSync(path.join(root, file), 'utf8'));
	const a = read('artifacts/specimen.json');
	const sh = read('artifacts/shakespeare/loss.json');
	const large = read('artifacts/shakespeare-large/loss.json');
	const memorised = read('artifacts/shakespeare-large/memorised.json');
	const layout = parameterLayout(a.config);
	const count = layout.reduce((sum, p) => sum + p.size, 0);
	if (count !== a.theta0.length || count !== a.train.theta_final.length) throw new Error('Specimen layout and artifact parameter counts disagree');
	const shCount = sh.model.param_count;
	if (!Number.isInteger(shCount)) throw new Error('Shakespeare model parameter count is missing');
	const size = (name) => layout.find((p) => p.name === name).size;
	return {
		largeParamsMillions: (large.model.param_count / 1e6).toFixed(2),
		smallParamsMillions: (shCount / 1e6).toFixed(2),
		largeSteps: large.final.step.toLocaleString('en-GB'),
		largePasses: String(Math.round(large.tokens.at(-1) / memorised.corpus.characters / 10) * 10),
		largeTrainingPasses: String(Math.round(large.tokens.at(-1) / memorised.corpus.trainCharacters)),
		largeBestStep: large.best.step.toLocaleString('en-GB'),
		largeBestLoss: large.best.val_loss.toFixed(3),
		largeFinalTrain: large.final.train_loss.toFixed(3),
		largeFinalVal: large.final.val_loss.toFixed(3),
		smallBestLoss: sh.best.val_loss.toFixed(3),
		trainingPercent: String(100 * memorised.corpus.trainFraction),
		validationPercent: String(Math.round(100 * (1 - memorised.corpus.trainFraction))),
		memorisedMinimum: String(memorised.minimum),
		specimenParams: String(count),
		embeddingParams: String(size('E')),
		attentionParams: String(4 * (a.config.D ** 2 + a.config.D)),
		differencePasses: String(2 * count),
		shakespeareParams: shCount.toLocaleString('en-GB'),
		shakespeareDifferenceMillions: (2 * shCount / 1e6).toFixed(1),
		predictions: String((a.corpus.length - a.config.T) * a.config.T),
		expOverflow: Math.log(Number.MAX_VALUE).toFixed(1)
	};
}

export function expandProse(md, root) {
	if (!md.includes('{{value:')) return md;
	const values = proseValues(root);
	return md.replace(/\{\{value:([A-Za-z]+)\}\}/g, (_, name) => {
		if (!(name in values)) throw new Error(`Unknown computed prose value: ${name}`);
		return values[name];
	});
}

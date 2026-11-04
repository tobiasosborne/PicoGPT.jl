// The handful of prose readings that depend on model data, expanded before Pandoc.
// Shapes and worked mathematical examples remain explicit in the chapter source.
import fs from 'node:fs';
import path from 'node:path';
import { parameterLayout } from '#specimen';

export function proseValues(root) {
	const read = (file) => JSON.parse(fs.readFileSync(path.join(root, file), 'utf8'));
	const a = read('artifacts/specimen.json');
	const sh = read('artifacts/shakespeare/loss.json');
	const layout = parameterLayout(a.config);
	const count = layout.reduce((sum, p) => sum + p.size, 0);
	if (count !== a.theta0.length || count !== a.train.theta_final.length) throw new Error('Specimen layout and artifact parameter counts disagree');
	const shCount = sh.model.param_count;
	if (!Number.isInteger(shCount)) throw new Error('Shakespeare model parameter count is missing');
	const size = (name) => layout.find((p) => p.name === name).size;
	return {
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

// Reproduce the chapter 9 exhibit: node scripts/export_memorised.mjs [--check].
// No dependency on the web build; the corpus is only needed to regenerate this file.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';

const root = fileURLToPath(new URL('../', import.meta.url));
const corpusPath = 'artifacts/data/tinyshakespeare.txt';
const samplePath = 'artifacts/shakespeare-large/samples.json';
const extraPath = 'artifacts/shakespeare-large/samples-extra.json';
const outputPath = 'artifacts/shakespeare-large/memorised.json';
const corpus = fs.readFileSync(path.join(root, corpusPath), 'utf8');
const samples = JSON.parse(fs.readFileSync(path.join(root, samplePath), 'utf8'));
const extra = JSON.parse(fs.readFileSync(path.join(root, extraPath), 'utf8'));
// Tiny Shakespeare is ASCII: JS string indices, Julia character indices and bytes agree.
if (/[^\x00-\x7f]/.test(corpus)) throw new Error('Expected the ASCII Tiny Shakespeare corpus');
const trainFraction = 0.9; // scripts/common.jl: shakespeare_split (contiguous prefix).
const split = Math.round(trainFraction * corpus.length);
const train = corpus.slice(0, split);
const minimum = 20;

function matches(text) {
	const spans = [];
	let furthest = 0;
	for (let start = 0; start + minimum <= text.length; start++) {
		const prefix = text.slice(start, start + minimum);
		let end = start, sourceStart = -1;
		for (let at = train.indexOf(prefix); at >= 0; at = train.indexOf(prefix, at + 1)) {
			let length = minimum;
			while (start + length < text.length && at + length < train.length && text[start + length] === train[at + length]) length++;
			if (start + length > end) { end = start + length; sourceStart = at; }
		}
		// The longest right extension at every start; discard those contained in an
		// earlier match. What remains cannot extend left or right anywhere in train.
		if (end > furthest && end - start >= minimum) {
			spans.push({ start, end, sourceStart });
			furthest = end;
		}
	}
	return spans;
}

const output = {
	definition: 'All inclusion-maximal substrings of the sample, at least minimum characters long, occurring exactly in the training prefix. Case, punctuation, spaces and newlines are significant. Overlapping spans are retained; underlining shows their union.',
	indices: 'Zero-based, half-open character ranges; sourceStart is one occurrence in the training prefix.',
	minimum,
	corpus: { path: corpusPath, sha256: createHash('sha256').update(corpus).digest('hex'), characters: corpus.length, trainFraction, trainCharacters: split, validationCharacters: corpus.length - split },
	sources: [samplePath, extraPath].map((file) => ({ path: file, sha256: createHash('sha256').update(fs.readFileSync(path.join(root, file))).digest('hex') })),
	samples: [
		...samples.samples.map((sample) => ({ ...sample, source: samplePath })),
		...extra.samples.map((sample) => ({ step: extra.step, temperature: extra.temperature, ...sample, source: extraPath }))
	].map((sample) => ({ ...sample, spans: matches(sample.text) }))
};
const json = JSON.stringify(output, null, 2) + '\n';
if (process.argv.includes('--check')) {
	if (fs.readFileSync(path.join(root, outputPath), 'utf8') !== json) throw new Error('memorised.json is stale; regenerate it');
	console.log(`${outputPath}: reproduced exactly`);
} else {
	fs.writeFileSync(path.join(root, outputPath), json);
	console.log(outputPath);
}

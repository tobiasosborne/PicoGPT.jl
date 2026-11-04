// artifacts.js — data from artifacts/ beyond the specimen, for the figures that use it.
//
// A page's server load calls artifactsFor(the ids of the figures on the page). Only the
// sources those figures need are read, condensed and sent with the page; figures read
// them through getContext('artifacts')(). A source whose files are missing is null, and
// its figures say so. Every number is read or computed from the files; none is typed.

import fs from 'node:fs';
import path from 'node:path';
import { ROOT } from './book.js';

const SOURCES = [
	{ key: 'shakespeare', match: /^fig-shakespeare-/, load: shakespeare },
	// (extension, epilogue) only the Shakespeare model's shape and parameter count, for fig-scale
	{ key: 'shakespeareModel', match: /^fig-scale$/, load: () => readJSON(path.join(ROOT, 'artifacts', 'shakespeare', 'loss.json'))?.model ?? null }
];

/** {key: data} for the sources the given figures need. */
export function artifactsFor(ids = []) {
	const out = {};
	for (const s of SOURCES) if (ids.some((id) => s.match.test(id))) out[s.key] = s.load();
	return out;
}

const cache = new Map();
function cached(key, files, make) {
	const stamp = files.map((f) => (fs.existsSync(f) ? fs.statSync(f).mtimeMs : '-')).join('|');
	const hit = cache.get(key);
	if (hit?.stamp === stamp) return hit.value;
	const value = make();
	cache.set(key, { stamp, value });
	return value;
}
const readJSON = (f) => (fs.existsSync(f) ? JSON.parse(fs.readFileSync(f, 'utf8')) : null);

// ── Tiny Shakespeare (scripts/export_book_data.jl) ─────────────────────────────

const ATTN_MIN = 0.01; // attention weights below this are not sent (and could not be seen)

function shakespeare() {
	const dir = path.join(ROOT, 'artifacts', 'shakespeare');
	const text = path.join(ROOT, 'artifacts', 'data', 'tinyshakespeare.txt');
	const files = ['loss.json', 'samples.json', 'attention.json'].map((f) => path.join(dir, f)).concat(text);
	return cached('shakespeare', files, () => {
		const [lossF, samplesF, attnF] = files;
		const loss = readJSON(lossF), samples = readJSON(samplesF), attn = readJSON(attnF);
		if (!loss && !samples && !attn) return null;
		const words = trainingWords(text);
		return {
			source: 'artifacts/shakespeare/',
			loss: loss && {
				model: loss.model, settings: loss.settings, uniform: loss.uniform, description: loss.description,
				step: loss.step, tokens: loss.tokens, seconds: loss.seconds,
				train: loss.train_loss, val: loss.val_loss, batch: loss.batch_loss, best: loss.best, final: loss.final
			},
			samples: samples && {
				prompt: samples.prompt, length: samples.length, temperatures: samples.temperatures, steps: samples.steps,
				samples: samples.samples.map((s) => ({ ...s, words: words ? wordStats(s.text, words) : null }))
			},
			attention: attn && condenseAttention(attn)
		};
	});
}

/** The words (maximal runs of letters, lower-cased) of the training text: its first 90 %
 *  (scripts/common.jl, shakespeare_split). Null if the text is absent. */
function trainingWords(file) {
	if (!fs.existsSync(file)) return null;
	const all = fs.readFileSync(file, 'utf8');
	const train = all.slice(0, Math.round(0.9 * all.length));
	return new Set(train.toLowerCase().match(/[a-z]+/g));
}

/** How many of a sample's words (runs of two or more letters) occur in the training text,
 *  and where in the text the others are ([start, end) character ranges). */
function wordStats(text, words) {
	let total = 0, known = 0;
	const unknown = [];
	for (const m of text.matchAll(/[A-Za-z]+/g)) {
		if (m[0].length < 2) continue;
		total++;
		if (words.has(m[0].toLowerCase())) known++;
		else unknown.push([m.index, m.index + m[0].length]);
	}
	return { total, known, unknown };
}

/** Attention as sparse rows per head: rows[i] = [j, w, j, w, …] for w ≥ ATTN_MIN (3 decimals),
 *  plus, per position, the model's probability of the character that actually comes next
 *  and its own favourite. */
function condenseAttention(a) {
	const { T, H, L } = a;
	const A = a.A.data;
	const heads = [];
	for (let l = 0; l < L; l++)
		for (let h = 0; h < H; h++) {
			const rows = [];
			for (let i = 0; i < T; i++) {
				const r = [];
				for (let j = 0; j <= i; j++) {
					const w = A[i + T * (j + T * (h + H * l))];
					if (w >= ATTN_MIN) r.push(j, Math.round(w * 1000) / 1000);
				}
				rows.push(r);
			}
			heads.push({ layer: l + 1, head: h + 1, rows });
		}
	const V = a.vocab.length, P = a.probs.data;
	const next = [];
	for (let i = 0; i < T; i++) {
		const col = P.slice(V * i, V * (i + 1));
		let best = 0;
		for (let k = 1; k < V; k++) if (col[k] > col[best]) best = k;
		const y = i + 1 < T ? a.vocab.indexOf(a.chars[i + 1]) : -1;
		next.push({ p: y >= 0 ? col[y] : null, best: a.vocab[best], pBest: col[best] });
	}
	return { step: a.step, prompt: a.prompt, chars: a.chars, T, H, L, V: a.vocab.length, min: ATTN_MIN, heads, next, description: a.description };
}

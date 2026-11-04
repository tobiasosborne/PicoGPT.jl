// Epilogue: the same kind of score as chapter 9, at the scale of the real model. Each stave
// is one moment of the Tiny-Shakespeare model's training (top: untrained; bottom: trained);
// its strands are the model's own samples at four temperatures. A stave has a line for each
// of the 65 characters, ordered by how often they occur in Shakespeare, so the most common
// character (the space, in value ink) is at the top. The untrained model writes noise across
// the whole stave; the trained one has learned that text lives near the top and returns to
// the space every few characters.
import { readFileSync, existsSync } from 'node:fs';
import { svg, save, ink } from './svg.mjs';

const root = new URL('../../', import.meta.url).pathname;
const S = JSON.parse(readFileSync(root + 'artifacts/shakespeare/samples.json', 'utf8'));
const source = existsSync(root + 'artifacts/data/tinyshakespeare.txt')
  ? readFileSync(root + 'artifacts/data/tinyshakespeare.txt', 'utf8')
  : S.samples.at(-1).text;                                            // fallback: the trained model's own text
const count = new Map();
for (const c of source) count.set(c, (count.get(c) ?? 0) + 1);
for (const { text } of S.samples) for (const c of text) if (!count.has(c)) count.set(c, 0);
const rank = new Map([...count].sort((a, b) => b[1] - a[1]).map(([c], r) => [c, r]));
const lines = rank.size, chars = 96;

const width = 150, height = 200, margin = 6, gap = 5, n = S.steps.length;
const staff = (height - 2 * margin - (n - 1) * gap) / n;
const x = i => margin + (width - 2 * margin) * i / (chars - 1);
const y = (k, c) => margin + k * (staff + gap) + staff * rank.get(c) / (lines - 1);
const f = v => v.toFixed(3);
const paths = [];
S.steps.forEach((step, k) => {
  paths.push({ d: `M${f(x(0))} ${f(y(k, ' '))}L${f(x(chars - 1))} ${f(y(k, ' '))}`, color: ink.value, opacity: 0.6 });
  for (const s of S.samples.filter(s => s.step === step)) {
    const t = [...s.text].slice(0, chars);
    let d = `M${f(x(0))} ${f(y(k, t[0]))}`;
    for (let i = 1; i < t.length; i++) {
      const xm = (x(i - 1) + x(i)) / 2;
      d += `C${f(xm)} ${f(y(k, t[i - 1]))} ${f(xm)} ${f(y(k, t[i]))} ${f(x(i))} ${f(y(k, t[i]))}`;
    }
    paths.push({ d, opacity: 0.55 - 0.1 * S.temperatures.indexOf(s.temperature) });
  }
});

save(process.argv[2] ?? '/tmp/epilogue.svg', svg({ width, height, paths,
  title: 'The Tiny-Shakespeare model learning to write, as a score.',
  desc: `${n} staves, top to bottom, at training steps ${S.steps.join(', ')}. Each stave has ${lines} lines, one per character, ` +
        `ordered by frequency in Tiny Shakespeare (the space, in value ink, at the top). Each strand is the first ${chars} ` +
        `characters of one sample from the model at that step, at temperatures ${S.temperatures.join(', ')} ` +
        `(darker = colder), prompted with a newline.` }));

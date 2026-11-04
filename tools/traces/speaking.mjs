// Chapter 9, "Speaking": the trained specimen, asked to continue "t", 240 times at each of three
// temperatures. The page is a score of three staves. The seven lines of a stave are the seven
// characters, and each sampled continuation is one strand passing through the line of each
// character it writes, left to right. Cold, every strand plays the corpus. As the temperature
// rises, the strands fray.
import { generate, seededRandom } from '../../web/src/specimen/index.mjs';
import { run, TRAIN, vocab, id } from './specimen.mjs';
import { svg, save, ink, stroke } from './svg.mjs';

const model = run(), samples = 240, length = 40, temperatures = [0.5, 1, 2];
const staves = temperatures.map((τ, k) => {
  const rng = seededRandom(1000 + k);
  return Array.from({ length: samples }, () => generate(model, [id('t')], length - 1, { temperature: τ, rng }));
});

const width = 150, height = 200, margin = 6, gap = 10, V = vocab.length;
const staff = (height - 2 * margin - 2 * gap) / 3, x = i => margin + (width - 2 * margin) * i / (length - 1);
const y = (k, c) => margin + k * (staff + gap) + staff * (c - 1) / (V - 1);        // c is a 1-based id
const f = v => v.toFixed(3);
const paths = [];
staves.forEach((strands, k) => {
  for (let c = 1; c <= V; c++)                                                     // the staff lines
    paths.push({ d: `M${f(x(0))} ${f(y(k, c))}L${f(x(length - 1))} ${f(y(k, c))}`, color: ink.value, opacity: 0.35 });
  for (const s of strands) {
    let d = `M${f(x(0))} ${f(y(k, s[0]))}`;
    for (let i = 1; i < length; i++) {
      const xm = (x(i - 1) + x(i)) / 2;
      d += `C${f(xm)} ${f(y(k, s[i - 1]))} ${f(xm)} ${f(y(k, s[i]))} ${f(x(i))} ${f(y(k, s[i]))}`;
    }
    paths.push({ d, opacity: 0.12 });
  }
});

const show = c => (c === ' ' ? '␣' : c);
save(process.argv[2] ?? '/tmp/speaking.svg', svg({ width, height, paths,
  title: 'The trained specimen continuing "t", 240 times at each of three temperatures.',
  desc: `Three staves, top to bottom, at temperatures ${temperatures.join(', ')}. The seven lines of each stave ` +
        `(value ink) are the characters ${vocab.map(show).join(' ')}, top to bottom; each strand is one sampled ` +
        `continuation of ${length} characters, passing through the line of each character it writes. ` +
        `The model is the book's specimen (${TRAIN}, from its stored θ₀); the samples use seeds 1000–1002.` }));

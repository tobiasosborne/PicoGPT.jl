// Prologue: everything the specimen ever reads. The corpus is written on a stave of seven
// lines, one per character, as in chapter 9's score. Each of the ten training windows (eight
// characters of input and the eight characters it must predict, one place later) is a strand
// on its own slightly lower copy of the stave, so the ten windows lie shingled like roof tiles
// along the text. The strand's value-ink tail is the last target, the eighth prediction.
import { corpus, vocab, id } from './specimen.mjs';
import { SPECIMEN_CONFIG } from '../../web/src/specimen/index.mjs';
import { svg, save, ink, stroke } from './svg.mjs';

const T = SPECIMEN_CONFIG.T, V = vocab.length, n = corpus.length, B = n - T;
const width = 150, height = 110, margin = 6, drop = 3.2, staff = height - 2 * margin - (B - 1) * drop;
const x = i => margin + (width - 2 * margin) * i / (n - 1);
const y = (b, c) => margin + b * drop + staff * (id(c) - 1) / (V - 1);
const f = v => v.toFixed(3);
const strand = (b, from, to) => {
  let d = `M${f(x(from))} ${f(y(b, corpus[from]))}`;
  for (let i = from + 1; i <= to; i++) {
    const xm = (x(i - 1) + x(i)) / 2;
    d += `C${f(xm)} ${f(y(b, corpus[i - 1]))} ${f(xm)} ${f(y(b, corpus[i]))} ${f(x(i))} ${f(y(b, corpus[i]))}`;
  }
  return d;
};
const paths = [];
for (let b = 0; b < B; b++) {
  paths.push({ d: strand(b, b, b + T - 1), opacity: 0.75 });                         // the input window
  paths.push({ d: strand(b, b + T - 1, b + T), color: ink.value, width: stroke.emphasis });   // its last target
}

save(process.argv[2] ?? '/tmp/prologue.svg', svg({ width, height, paths,
  title: `Everything the specimen reads: the ${B} training windows of "${corpus}".`,
  desc: `A stave of ${V} lines, one per character (${vocab.map(c => c === ' ' ? '␣' : c).join(' ')}, top to bottom); the corpus runs left to right. ` +
        `Each of the ${B} windows of ${T} input characters is a strand on its own copy of the stave, each copy ${drop} mm lower than the last; ` +
        `the value-ink tail is the step to the window's last target.` }));

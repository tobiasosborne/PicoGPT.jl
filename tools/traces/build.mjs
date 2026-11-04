// Regenerate every frontispiece into web/public/traces/.  Usage: node tools/traces/build.mjs [slug…]
import { execFileSync } from 'node:child_process';
import { readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const out = join(here, '../../web/public/traces');
const slugs = process.argv.slice(2).length ? process.argv.slice(2)
  : readdirSync(here).filter(f => f.endsWith('.mjs') && !['build.mjs', 'specimen.mjs', 'svg.mjs'].includes(f))
                     .map(f => f.slice(0, -4));
for (const slug of slugs)
  execFileSync('node', [join(here, `${slug}.mjs`), join(out, `${slug}.svg`)], { stdio: 'inherit' });

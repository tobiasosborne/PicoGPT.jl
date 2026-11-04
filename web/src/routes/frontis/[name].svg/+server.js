// /frontis/<slug>.svg — the trace from public/traces/<slug>.svg as the page uses it:
// any full-bleed background rectangle is removed so the strokes sit on the book's
// own paper (and invert cleanly at night). Prerendered for every trace present.
import fs from 'node:fs';
import path from 'node:path';

const DIR = path.join(process.cwd(), 'public', 'traces');
export const prerender = true;
export const trailingSlash = 'never';

export function entries() {
	return fs.existsSync(DIR) ? fs.readdirSync(DIR).filter((f) => f.endsWith('.svg')).map((f) => ({ name: f.replace(/\.svg$/, '') })) : [];
}

export function GET({ params }) {
	const file = path.join(DIR, `${params.name}.svg`);
	if (!/^[\w-]+$/.test(params.name) || !fs.existsSync(file)) return new Response('not found', { status: 404 });
	let svg = fs.readFileSync(file, 'utf8');
	svg = svg.replace(/<rect\b[^>]*\bwidth="100%"[^>]*\bheight="100%"[^>]*\/>/, '');
	return new Response(svg, { headers: { 'content-type': 'image/svg+xml', 'cache-control': 'public, max-age=3600' } });
}

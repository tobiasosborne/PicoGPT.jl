// book.js — build-time conversion of book/chapters/NN-slug.md into page data.
//
// pandoc (+ book.lua) → HTML fragment → maths via Temml (MathML Core, rendered by
// the browser in XCharter Math) → code excerpts from the Julia source → segments:
// a chapter is a list of {t:'html'} and {t:'fig'} parts so the page can mount each
// figure component in place (SSR'd in its static state, hydrated for interaction).
//
// Runs in Node only (dev server, prerender). Never imported by client code.

import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import temml from 'temml';
import { extract, highlight } from './julia.js';
import { MACROS } from './macros.js';
import { OUTLINE, BOOK } from '../outline.js';

const WEB = process.cwd(); // vite runs in web/
export const ROOT = process.env.BOOK_ROOT ?? path.resolve(WEB, '..');
const CHAPTERS = path.join(ROOT, 'book', 'chapters');
const FILTER = path.join(WEB, 'src', 'lib', 'server', 'book.lua');
const STRICT = process.env.BOOK_STRICT === '1';

function pandocBin() {
	if (process.env.PANDOC) return process.env.PANDOC;
	const local = path.join(os.homedir(), '.local', 'bin', 'pandoc');
	return fs.existsSync(local) ? local : 'pandoc';
}

const warn = (msg) => {
	if (STRICT) throw new Error(msg);
	console.warn(`\x1b[33m[book]\x1b[0m ${msg}`);
};

// ── chapter list and the cross-reference index ───────────────────────────────

/** Files present in book/chapters, merged with the planned outline. */
export function listChapters() {
	const files = fs.existsSync(CHAPTERS) ? fs.readdirSync(CHAPTERS).filter((f) => /^\d\d-[a-z0-9-]+\.md$/.test(f)) : [];
	const present = new Map();
	for (const f of files) {
		const [, nn, slug] = f.match(/^(\d\d)-([a-z0-9-]+)\.md$/);
		const src = fs.readFileSync(path.join(CHAPTERS, f), 'utf8');
		const h1 = src.match(/^#\s+(.+?)\s*(\{[^}]*\})?\s*$/m);
		const id = h1?.[2]?.match(/#([\w-]+)/)?.[1] ?? `ch-${slug}`;
		present.set(+nn, { num: +nn, slug, file: f, path: path.join(CHAPTERS, f), title: h1 ? h1[1].trim() : slug, id, exists: true });
	}
	const out = OUTLINE.map((o) => ({ ...o, ...(present.get(o.num) ?? { exists: false, id: `ch-${o.slug}` }) }));
	for (const [num, c] of present) if (!OUTLINE.some((o) => o.num === num)) out.push({ gist: '', ...c });
	return out.sort((a, b) => a.num - b.num);
}

function scanIds(src, chapterNum) {
	const figures = [], sections = [], exercises = [];
	let secCount = 0, exCount = 0, figCount = 0;
	for (const line of src.split('\n')) {
		const div = line.match(/^:{3,}\s*\{([^}]*)\}/);
		const bare = line.match(/^:{3,}\s*([a-z]+)\s*$/);
		if (div) {
			const attrs = div[1];
			const id = attrs.match(/#([\w-]+)/)?.[1];
			if (/(^|\s)\.figure(\s|$)/.test(attrs)) { figCount++; if (id) figures.push([id, `${chapterNum}.${figCount}`]); }
			if (/(^|\s)\.exercise(\s|$)/.test(attrs)) { exCount++; if (id) exercises.push([id, `${chapterNum}.${exCount}`]); }
		} else if (bare?.[1] === 'exercise') exCount++;
		const h2 = line.match(/^##\s+(.*?)\s*(\{[^}]*\})?\s*$/);
		if (h2 && !line.startsWith('###') && !/\.unnumbered|(^|\s)-(\s|$)/.test(h2[2] ?? '')) {
			secCount++;
			const id = h2[2]?.match(/#([\w-]+)/)?.[1];
			if (id) sections.push([id, `${chapterNum}.${secCount}`]);
		}
	}
	return { figures, sections, exercises };
}

let indexCache = { key: '', path: '', data: null };
/** Global numbering for @fig-/@sec-/@ex-/@ch- references across chapters. */
export function bookIndex() {
	const chapters = listChapters().filter((c) => c.exists);
	const key = chapters.map((c) => `${c.file}:${fs.statSync(c.path).mtimeMs}`).join('|');
	if (indexCache.key === key) return indexCache;
	const data = { figures: {}, sections: {}, exercises: {}, chapters: {}, where: {}, bynum: {} };
	// planned chapters resolve to "Chapter n" (unlinked) until their file exists
	for (const c of listChapters()) if (!c.exists) data.chapters[c.id] = { num: c.unnumbered ? '' : String(c.num), slug: c.slug, planned: true, title: c.title };
	for (const c of chapters) {
		const label = c.unnumbered ? '' : String(c.num);
		const prefix = c.unnumbered ? 'E' : String(c.num); // figure/section numbering prefix
		data.chapters[c.id] = { num: label, slug: c.slug };
		data.bynum[String(c.num)] = c.slug;
		// (extension, epilogue) chapter files, so the `::: dictionary` filter can read every names table
		(data.files ??= []).push({ num: label, slug: c.slug, id: c.id, title: c.title, path: c.path });
		const ids = scanIds(fs.readFileSync(c.path, 'utf8'), prefix);
		for (const [k, v] of ids.figures) { data.figures[k] = v; data.where[k] = String(c.num); }
		for (const [k, v] of ids.sections) { data.sections[k] = v; data.where[k] = String(c.num); }
		for (const [k, v] of ids.exercises) { data.exercises[k] = v; data.where[k] = String(c.num); }
	}
	const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'picogpt-book-'));
	const file = path.join(dir, 'index.json');
	fs.writeFileSync(file, JSON.stringify(data));
	indexCache = { key, path: file, data };
	return indexCache;
}

// ── maths ────────────────────────────────────────────────────────────────────

const unescape = (s) => s.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&amp;/g, '&');

export function tex(src, display = false, where = '') {
	try {
		return temml.renderToString(src, { displayMode: display, macros: { ...MACROS }, throwOnError: true, annotate: true, wrap: display ? 'none' : 'tex' });
	} catch (e) {
		warn(`maths error${where ? ` in ${where}` : ''}: ${e.message}\n    ${src}`);
		return `<code class="tex-error" title="${e.message.replace(/"/g, '&quot;')}">${src.replace(/</g, '&lt;')}</code>`;
	}
}

function renderMath(html, where) {
	// display maths sits in a scrolling span: MathML itself does not scroll on narrow screens
	html = html.replace(/<tex-([md])>([\s\S]*?)<\/tex-\1>/g, (_, kind, body) =>
		kind === 'd' ? `<span class="eq-scroll">${tex(unescape(body), true, where)}</span>` : tex(unescape(body), false, where));
	html = html.replace(/(<math\b(?![^>]*display="block")[^>]*>(?:(?!<\/math>)[\s\S])*<\/math>)([.,;:!?]+)/g, '<span class="math-punct">$1$2</span>');
	// a paragraph holding only display maths becomes an equation block
	return html.replace(/<p>\s*(<span class="eq-scroll">[\s\S]*?<\/span>)\s*<\/p>/g, '<div class="eq">$1</div>');
}

// ── code excerpts ────────────────────────────────────────────────────────────

function renderCode(spec, where) {
	const file = spec.file;
	const rel = file ?? '(no file)';
	const sel = spec.fn ? `fn=${spec.fn}` : spec.struct ? `struct=${spec.struct}` : spec.region ? `region=${spec.region}` : spec.lines ? `lines=${spec.lines}` : '';
	const abs = file ? path.join(ROOT, file) : null;
	let result;
	if (!abs || !fs.existsSync(abs)) result = { error: `source not found: ${rel}` };
	else result = extract(fs.readFileSync(abs, 'utf8'), spec);
	if (result.error) {
		warn(`code excerpt in ${where}: ${result.error} (${rel} ${sel})`);
		return `<figure class="code code--missing" data-file="${rel}" data-sel="${sel}"><figcaption class="code__head"><span class="code__file">${rel}</span> <span class="code__sel">${sel}</span></figcaption><p class="code__missing">Excerpt not yet available: ${result.error}. It is pulled from the Julia source at build time and will appear when that source exists.</p></figure>`;
	}
	const href = (c) => `${BOOK.repo}/blob/master/${rel}#L${c.start}-L${c.end}`;
	const range = result.chunks.map((c) => (c.start === c.end ? `${c.start}` : `${c.start}–${c.end}`)).join(', ');
	const body = result.chunks
		.map((c) => {
			const lines = highlight(c.code).split('\n');
			const nums = lines.map((_, k) => `<span>${c.start + k}</span>`).join('');
			return `<div class="code__chunk"><a class="code__nums" href="${href(c)}" aria-hidden="true">${nums}</a><pre><code>${lines.join('\n')}</code></pre></div>`;
		})
		.join('<div class="code__gap" aria-hidden="true">⋮</div>');
	const caption = spec.caption ? `<div class="code__caption">${spec.caption}</div>` : '';
	return `<figure class="code" data-file="${rel}" data-sel="${sel}"><figcaption class="code__head"><a class="code__file" href="${href(result.chunks[0])}">${rel}</a><span class="code__where">${result.label} · ${result.chunks.length > 1 ? 'lines' : 'line' + (result.chunks[0].start === result.chunks[0].end ? '' : 's')} ${range}</span></figcaption>${body}${caption}</figure>`;
}

// ── chapters ─────────────────────────────────────────────────────────────────

const cache = new Map();

function sourcesKey(md) {
	// code excerpts depend on Julia files: include their mtimes in the cache key
	const files = [...md.matchAll(/file="([^"]+)"/g)].map((m) => m[1]);
	return files.map((f) => { const p = path.join(ROOT, f); return fs.existsSync(p) ? `${f}:${fs.statSync(p).mtimeMs}` : `${f}:-`; }).join('|');
}

/**
 * Render one chapter.
 * @returns {{meta, segments: Array<{t:'html', html:string}|{t:'fig', id, num, attrs, classes, caption, notes}>, figures:string[]}}
 */
export function renderChapter(slug) {
	const chapters = listChapters();
	const c = chapters.find((x) => x.slug === slug && x.exists);
	if (!c) return null;
	const md = fs.readFileSync(c.path, 'utf8');
	const index = bookIndex();
	const key = [fs.statSync(c.path).mtimeMs, fs.statSync(FILTER).mtimeMs, index.key, sourcesKey(md)].join('#');
	const hit = cache.get(slug);
	if (hit?.key === key) return hit.value;

	const label = c.unnumbered ? '' : String(c.num);
	const prefix = c.unnumbered ? 'E' : label;
	let html = execFileSync(pandocBin(), ['--from', 'markdown+smart', '--to', 'html5', '--wrap=none', '--lua-filter', FILTER, '-M', `chapter=${prefix}`, '-M', `index=${index.path}`, c.path], { encoding: 'utf8', maxBuffer: 64 << 20, stdio: ['ignore', 'pipe', 'pipe'] });

	const metaMatch = html.match(/<!--BOOKMETA (.*?)-->/);
	const meta = metaMatch ? JSON.parse(metaMatch[1]) : {};
	html = html.replace(/<!--BOOKMETA .*?-->\n?/, '');
	const where = c.file;
	html = renderMath(html, where);
	html = html.replace(/<!--CODE (.*?)-->/g, (_, json) => renderCode(JSON.parse(json), where));
	if (meta.claim) meta.claim = renderMath(meta.claim, where);
	if (meta.titleHtml) meta.titleHtml = renderMath(meta.titleHtml, where);

	const segments = [];
	const figures = [];
	const parts = html.split(/<!--FIG (.*?)-->([\s\S]*?)<!--\/FIG-->/);
	for (let k = 0; k < parts.length; k += 3) {
		const h = parts[k].trim();
		if (h) segments.push({ t: 'html', html: h });
		if (k + 1 < parts.length) {
			const spec = JSON.parse(parts[k + 1]);
			const inner = parts[k + 2];
			const notes = [...inner.matchAll(/<aside class="note[^"]*">[\s\S]*?<\/aside>/g)].map((m) => m[0]);
			const caption = inner.replace(/<aside class="note[^"]*">[\s\S]*?<\/aside>/g, '').trim();
			segments.push({ t: 'fig', id: spec.id, num: spec.num, attrs: spec.attrs ?? {}, classes: Array.isArray(spec.classes) ? spec.classes : [], caption, notes });
			figures.push(spec.id);
		}
	}
	const value = {
		meta: { ...meta, num: label, slug: c.slug, file: c.file, title: meta.title ?? c.title },
		segments,
		figures
	};
	cache.set(slug, { key, value });
	return value;
}

/** Neighbours for chapter navigation (only chapters that exist). */
export function neighbours(slug) {
	const list = listChapters().filter((c) => c.exists);
	const k = list.findIndex((c) => c.slug === slug);
	const pick = (c) => (c ? { slug: c.slug, num: c.unnumbered ? '' : String(c.num), title: c.title } : null);
	return { prev: pick(list[k - 1]), next: pick(list[k + 1]) };
}

/** Does a frontispiece trace exist? (public/traces/<slug>.svg, cover.svg) */
export function traceFor(slug) {
	const p = path.join(WEB, 'public', 'traces', `${slug}.svg`);
	if (!fs.existsSync(p)) return null;
	const svg = fs.readFileSync(p, 'utf8');
	const vb = svg.match(/viewBox="([^"]+)"/)?.[1]?.split(/[\s,]+/).map(Number);
	const title = svg.match(/<title>([\s\S]*?)<\/title>/)?.[1]?.trim() ?? null;
	const desc = svg.match(/<desc>([\s\S]*?)<\/desc>/)?.[1]?.trim() ?? null;
	return { src: `frontis/${slug}.svg`, aspect: vb && vb[2] && vb[3] ? vb[2] / vb[3] : null, bytes: Buffer.byteLength(svg), title, desc };
}

/** Every figure slot in the book, in order: {id, num, attrs, caption, notes, chapter}. */
export function allFigures() {
	const out = [];
	for (const c of listChapters().filter((x) => x.exists)) {
		const ch = renderChapter(c.slug);
		for (const seg of ch.segments) if (seg.t === 'fig') out.push({ ...seg, chapter: { slug: c.slug, num: ch.meta.num, title: ch.meta.title } });
	}
	return out;
}

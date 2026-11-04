#!/usr/bin/env node
// print/build.mjs — the print edition of PicoGPT, built from the web book's HTML.
//
//   npm run pdf            stage web/, build it, paginate /print/, write print/PicoGPT.pdf
//   npm run proof          … and render page PNGs + contact sheets into print/proof/
//   npm run pdf:reuse      skip the site build (reuse .work/site/build) — for CSS iteration
//
// Flags: --skip-build  --proof  --strict (warnings fail the build)  --tolerant (leave out figures
//        that do not compile, with a warning)  --out=FILE  --keep-open
//
// Pipeline (one renderer, DESIGN.md §8):
//   1. Stage web/ into print/.work/site (so a concurrent `npm run dev`/`build` in web/ is
//      never disturbed) and prerender it with Vite/SvelteKit, BOOK_ROOT pointing at the repo.
//   2. Serve the static build from a throwaway localhost server.
//   3. Headless Chromium opens /print/, waits for <html data-print-ready="1">, then
//      paginates the page's own HTML and CSS with Paged.js (Previewer API) plus the
//      handlers in paginate.js (figure floats, page-side classes, diagnostics).
//   4. Checks. Hard errors always fail; warnings fail under --strict.
//   5. Chromium prints the paged DOM; pdf-lib adds the outline (bookmarks) and metadata.
//   6. --proof: pdftoppm + ImageMagick montage → page PNGs, colour and greyscale contact sheets.

import { spawnSync, execFileSync } from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { PDFDocument, PDFName, PDFString, PDFNumber, PDFArray, PDFNull, PDFHexString } from 'pdf-lib';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const WEB = path.join(ROOT, 'web');
const WORK = path.join(HERE, '.work');
const STAGE = path.join(WORK, 'site');
const PROOF = path.join(HERE, 'proof');

const args = process.argv.slice(2);
const flag = (f) => args.includes(`--${f}`);
const opt = (k, d) => args.find((a) => a.startsWith(`--${k}=`))?.split('=').slice(1).join('=') ?? d;
const OUT = path.resolve(opt('out', path.join(HERE, 'PicoGPT.pdf')));
const STRICT = flag('strict');

const t0 = Date.now();
const log = (...m) => console.log(`\x1b[2m[print ${((Date.now() - t0) / 1000).toFixed(1).padStart(5)}s]\x1b[0m`, ...m);
const die = (msg) => { console.error(`\n\x1b[31m✗ print build failed:\x1b[0m ${msg}\n`); process.exit(1); };
const warnings = [];
const warn = (m) => { warnings.push(m); console.warn(`\x1b[33m  ! ${m}\x1b[0m`); };

// ── 1. stage and build the site ─────────────────────────────────────────────────
function stage() {
	fs.rmSync(STAGE, { recursive: true, force: true });
	fs.mkdirSync(STAGE, { recursive: true });
	const skip = new Set(['node_modules', 'build', '.svelte-kit', '.vite']);
	for (const e of fs.readdirSync(WEB)) {
		if (skip.has(e)) continue;
		fs.cpSync(path.join(WEB, e), path.join(STAGE, e), { recursive: true, dereference: false });
	}
	fs.symlinkSync(path.join(WEB, 'node_modules'), path.join(STAGE, 'node_modules'), 'dir');
}

function buildSite() {
	if (!fs.existsSync(path.join(WEB, 'node_modules'))) die('web/node_modules is missing: run `npm install` in web/ first.');
	log('staging web/ →', path.relative(ROOT, STAGE));
	stage();
	log('building the site (vite build, prerender)…');
	const vite = path.join(WEB, 'node_modules', 'vite', 'bin', 'vite.js');
	let out = '';
	for (let attempt = 0; ; attempt++) {
		const r = spawnSync(process.execPath, [vite, 'build', '--logLevel', 'warn'], {
			cwd: STAGE,
			env: { ...process.env, BOOK_ROOT: ROOT, BASE_PATH: '', NODE_ENV: 'production' },
			encoding: 'utf8',
			maxBuffer: 256 << 20
		});
		out = `${r.stdout ?? ''}${r.stderr ?? ''}`;
		fs.writeFileSync(path.join(WORK, 'site-build.log'), out);
		if (r.status === 0) break;
		// --tolerant (proofs while chapters are being written): a figure component that does not
		// compile is removed from the staged copy, so the book shows it as "in preparation"
		const failure = out.replace(/\x1b\[[0-9;]*m/g, '').split(/Build failed with|error during build/).slice(1).join('\n');
		const bad = failure.match(/(src\/figures\/[\w-]+\/(fig-[\w-]+?)(?:\.key)?\.svelte)/);
		if (flag('tolerant') && bad && attempt < 8) {
			warn(`${bad[1]} does not compile; left out of this proof (${failure.match(/(Unexpected token|Could not resolve '[^']+'|[A-Z_]+_ERROR[^\n]*)/)?.[1] ?? 'see the build log'})`);
			for (const f of [`${bad[2]}.svelte`, `${bad[2]}.key.svelte`]) fs.rmSync(path.join(STAGE, path.dirname(bad[1]), f), { force: true });
			continue;
		}
		console.error(out.split('\n').slice(-40).join('\n'));
		die(`vite build exited with ${r.status} (log: ${path.relative(ROOT, path.join(WORK, 'site-build.log'))})${bad ? '\n  (a figure does not compile; `--tolerant` leaves such figures out of a proof)' : ''}`);
	}
	// surface the book pipeline's own warnings (missing excerpts, maths errors, unresolved refs)
	const bookWarn = [...new Set(out.split('\n').filter((l) => l.includes('[book]')).map((l) => l.replace(/\x1b\[[0-9;]*m/g, '').trim()))];
	for (const w of bookWarn) warn(`site: ${w}`);
	if (!fs.existsSync(path.join(STAGE, 'build', 'print', 'index.html')) && !fs.existsSync(path.join(STAGE, 'build', 'print.html'))) die('the build has no /print/ page.');
}

// ── 2. serve ────────────────────────────────────────────────────────────────────
const MIME = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png', '.jpg': 'image/jpeg', '.webp': 'image/webp', '.woff2': 'font/woff2', '.woff': 'font/woff', '.ttf': 'font/ttf', '.otf': 'font/otf', '.txt': 'text/plain', '.webm': 'video/webm', '.mp4': 'video/mp4' };
function serve(dir) {
	const server = http.createServer((req, res) => {
		const u = decodeURIComponent(new URL(req.url, 'http://x').pathname);
		let f = path.join(dir, path.normalize(u).replace(/^(\.\.[/\\])+/, ''));
		if (!f.startsWith(dir)) { res.writeHead(403).end(); return; }
		if (fs.existsSync(f) && fs.statSync(f).isDirectory()) f = path.join(f, 'index.html');
		else if (!fs.existsSync(f) && fs.existsSync(`${f}.html`)) f = `${f}.html`;
		if (!fs.existsSync(f)) { res.writeHead(404).end('not found'); return; }
		res.writeHead(200, { 'content-type': MIME[path.extname(f)] ?? 'application/octet-stream', 'cache-control': 'no-store' });
		fs.createReadStream(f).pipe(res);
	});
	return new Promise((resolve) => server.listen(0, '127.0.0.1', () => resolve(server)));
}

// ── 3–5. paginate and print ─────────────────────────────────────────────────────
async function paginate(base) {
	const browser = await chromium.launch({ args: ['--font-render-hinting=none', '--disable-lcd-text'] });
	// 1400 px: wider than every web breakpoint, so the print grid is the only one in force
	const context = await browser.newContext({ viewport: { width: 1400, height: 1000 }, deviceScaleFactor: 1, colorScheme: 'light', reducedMotion: 'reduce', locale: 'en-GB' });
	const page = await context.newPage();
	const errors = [];
	const consoleLines = [];
	page.on('pageerror', (e) => errors.push(`page error: ${e.message}`));
	page.on('console', (m) => {
		const t = m.text();
		if (m.type() === 'error') errors.push(`console error: ${t}`);
		else if (m.type() === 'warning' || /Unable to layout/.test(t)) consoleLines.push(`${m.type()}: ${t}`);
	});
	page.on('requestfailed', (r) => errors.push(`request failed: ${r.url()} (${r.failure()?.errorText})`));
	page.on('response', (r) => { if (r.status() >= 400) errors.push(`HTTP ${r.status()}: ${r.url()}`); });

	// One pass: load /print/, wait until it is ready, paginate. Cross-reference page numbers
	// change the layout they describe, so passes repeat until they stop changing.
	const pass = async (xpages, k) => {
		errors.length = 0;
		consoleLines.length = 0;
		await page.goto(`${base}/print/`, { waitUntil: 'networkidle', timeout: 120000 });
		try {
			await page.waitForFunction(() => document.documentElement.dataset.printReady === '1', null, { timeout: 120000 });
		} catch {
			die(`/print/ never set data-print-ready="1" (hydration failed?)\n${errors.join('\n')}`);
		}
		if (errors.length) die(`the print page reported errors before pagination:\n  ${errors.join('\n  ')}`);
		await page.addScriptTag({ path: path.join(HERE, 'node_modules', 'pagedjs', 'dist', 'paged.js') });
		await page.addScriptTag({ path: path.join(HERE, 'paginate.js') });
		const r = await page.evaluate((xp) => window.__picoPaginate({ xpages: xp }), xpages);
		if (r.error) die(`Paged.js: ${r.error}`);
		if (!r.pages) die('Paged.js produced no pages.');
		if (errors.length) die(`errors during pagination:\n  ${errors.join('\n  ')}`);
		log(`pass ${k}: ${r.pages} pages in ${(r.ms / 1000).toFixed(1)} s`);
		return r;
	};
	// a reference earns a page number when its target is not on the same spread (page 1 is a recto)
	const spread = (p) => Math.floor(p / 2);
	const decide = (x) => Object.fromEntries(Object.entries(x).filter(([, v]) => spread(v.page) !== spread(v.target)).map(([k, v]) => [k, String(v.target)]));
	const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
	log('paginating /print/ with Paged.js in headless Chromium…');
	let xpages = {};
	let result = await pass(xpages, 1);
	for (let k = 2; k <= 5; k++) {
		const next = decide(result.xrefs);
		if (same(next, xpages)) break;
		if (k === 5) { warn('cross-reference page numbers did not settle after 4 passes'); break; }
		xpages = next;
		result = await pass(xpages, k);
	}
	log(`${Object.keys(xpages).length} of ${Object.keys(result.xrefs).length} cross-references carry a page number`);
	fs.writeFileSync(path.join(WORK, 'xpages.json'), JSON.stringify(xpages));
	// "Unable to layout item": Paged.js could not fit a block on a page and moved on; whether
	// anything was lost is checked by the audit (lost content is an error), so these are notes
	const notes = consoleLines.filter((l) => /Unable to layout item/.test(l));
	for (const l of consoleLines) if (!notes.includes(l)) warn(`paged.js ${l}`);
	result.diagnostics.pagedNotes = notes.length;

	const d = result.diagnostics;
	for (const e of d.errors) errors.push(e);
	for (const w of d.warnings) warn(w);
	if (errors.length) die(`layout errors:\n  ${errors.join('\n  ')}`);

	log('printing to PDF…');
	const raw = await page.pdf({ preferCSSPageSize: true, printBackground: true, tagged: true, outline: false, displayHeaderFooter: false });
	if (flag('keep-open')) { log('--keep-open: browser left running; Ctrl-C to quit'); await new Promise(() => {}); }
	await browser.close();
	return { raw, result };
}

// ── pdf-lib: outline (bookmarks) and metadata ─────────────────────────────────────
async function finish(raw, result) {
	const pdf = await PDFDocument.load(raw, { updateMetadata: false });
	const pages = pdf.getPages();
	if (pages.length !== result.pages) die(`Chromium printed ${pages.length} pages but Paged.js laid out ${result.pages} (a page box is overflowing the sheet).`);
	const { width: W, height: H } = pages[0].getSize();
	const mm = (pt) => (pt / 72 * 25.4).toFixed(1);
	if (Math.abs(W / 72 * 25.4 - result.trim.w) > 0.5 || Math.abs(H / 72 * 25.4 - result.trim.h) > 0.5) die(`PDF page size ${mm(W)}×${mm(H)} mm differs from the trim ${result.trim.w}×${result.trim.h} mm.`);

	// Outline: front matter, then each chapter with its sections.
	const ctx = pdf.context;
	const outlinesRef = ctx.nextRef();
	const tree = result.outline; // [{title, page, y, children:[…]}]
	const make = (items, parentRef) => {
		const refs = items.map(() => ctx.nextRef());
		items.forEach((it, k) => {
			const pageRef = pages[Math.min(it.page, pages.length - 1)].ref;
			const top = H - it.y * 0.75; // CSS px from the page top → PDF points from the bottom
			const dict = ctx.obj({
				Title: PDFHexString.fromText(it.title),
				Parent: parentRef,
				Dest: [pageRef, PDFName.of('XYZ'), PDFNull, PDFNumber.of(Math.max(0, Math.min(H, top + 6))), PDFNull]
			});
			if (k > 0) dict.set(PDFName.of('Prev'), refs[k - 1]);
			if (k < items.length - 1) dict.set(PDFName.of('Next'), refs[k + 1]);
			if (it.children?.length) {
				const kids = make(it.children, refs[k]);
				dict.set(PDFName.of('First'), kids[0]);
				dict.set(PDFName.of('Last'), kids[kids.length - 1]);
				dict.set(PDFName.of('Count'), PDFNumber.of(-it.children.length)); // closed by default
			}
			ctx.assign(refs[k], dict);
		});
		return refs;
	};
	if (tree.length) {
		const top = make(tree, outlinesRef);
		ctx.assign(outlinesRef, ctx.obj({ Type: 'Outlines', First: top[0], Last: top[top.length - 1], Count: PDFNumber.of(top.length) }));
		pdf.catalog.set(PDFName.of('Outlines'), outlinesRef);
		pdf.catalog.set(PDFName.of('PageMode'), PDFName.of('UseOutlines'));
	}
	// facing pages in viewers that honour it; page labels: roman front matter, arabic body
	pdf.catalog.set(PDFName.of('PageLayout'), PDFName.of('TwoPageRight'));
	const vp = ctx.obj({ DisplayDocTitle: true });
	pdf.catalog.set(PDFName.of('ViewerPreferences'), vp);

	const when = process.env.SOURCE_DATE_EPOCH ? new Date(+process.env.SOURCE_DATE_EPOCH * 1000) : new Date(result.built + 'T12:00:00Z');
	pdf.setTitle(`${result.meta.title}: ${result.meta.subtitle}`, { showInWindowTitleBar: true });
	pdf.setAuthor(result.meta.author);
	pdf.setSubject('A transformer built from scratch in Julia, every number shown. Print edition.');
	pdf.setKeywords(['transformer', 'GPT', 'attention', 'backpropagation', 'Julia', 'machine learning']);
	pdf.setCreator('PicoGPT print/build.mjs — SvelteKit HTML, Paged.js 0.4.3, headless Chromium');
	pdf.setProducer('Chromium (Skia/PDF) + pdf-lib');
	pdf.setLanguage('en-GB');
	pdf.setCreationDate(when);
	pdf.setModificationDate(when);
	const bytes = await pdf.save({ useObjectStreams: false });
	fs.mkdirSync(path.dirname(OUT), { recursive: true });
	fs.writeFileSync(OUT, bytes);
}

// ── checks on the written PDF ───────────────────────────────────────────────────
function checkFonts() {
	let out;
	try { out = execFileSync('pdffonts', [OUT], { encoding: 'utf8' }); } catch { warn('pdffonts not available: font embedding not checked'); return []; }
	const rows = out.split('\n').slice(2).filter(Boolean);
	const fonts = rows.map((r) => {
		const name = r.slice(0, 37).trim();
		const cols = r.slice(37).trim().split(/\s+/);
		// type may be two words ("CID TrueType"); the emb/sub/uni columns are the 3 yes/no before obj
		const yn = cols.filter((c) => c === 'yes' || c === 'no');
		return { name, emb: yn[0] === 'yes' };
	});
	const notEmbedded = fonts.filter((f) => !f.emb);
	if (notEmbedded.length) die(`fonts not embedded: ${notEmbedded.map((f) => f.name).join(', ')}`);
	const families = [...new Set(fonts.map((f) => f.name.replace(/^[A-Z]{6}\+/, '').replace(/[-,].*$/, '')))].sort();
	const expected = /^(Literata|Archivo|JuliaMono|XCharter)/i;
	const strays = families.filter((f) => !expected.test(f));
	if (strays.length) warn(`fallback fonts in the PDF (a glyph missing from the book's faces?): ${strays.join(', ')}`);
	return families;
}

function proof(nPages) {
	const have = (b) => spawnSync('sh', ['-c', `command -v ${b}`]).status === 0;
	if (!have('pdftoppm') || !have('montage')) { warn('pdftoppm/montage missing: no proof images'); return; }
	fs.rmSync(PROOF, { recursive: true, force: true });
	fs.mkdirSync(path.join(PROOF, 'pages'), { recursive: true });
	log('rendering proof pages…');
	execFileSync('pdftoppm', ['-r', '110', '-png', OUT, path.join(PROOF, 'pages', 'p')]);
	execFileSync('pdftoppm', ['-r', '36', '-png', OUT, path.join(PROOF, 'thumb')]);
	execFileSync('pdftoppm', ['-r', '36', '-gray', '-png', OUT, path.join(PROOF, 'gthumb')]);
	const thumbs = (pre) => fs.readdirSync(PROOF).filter((f) => f.startsWith(pre) && f.endsWith('.png')).sort();
	// spreads: page 1 is a recto, so pad one blank before it; 8 pages (4 spreads) per row
	const blank = path.join(PROOF, 'blank.png');
	const first = path.join(PROOF, thumbs('thumb-')[0]);
	execFileSync('convert', [first, '-fill', '#6b6e75', '-colorize', '100', blank]);
	const sheet = (pre, out) => {
		const files = [blank, ...thumbs(pre).map((f) => path.join(PROOF, f))];
		execFileSync('montage', [...files, '-tile', '8x', '-geometry', '+1+10', '-background', '#6b6e75', path.join(PROOF, out)]);
	};
	sheet('thumb-', 'contact-sheet.png');
	sheet('gthumb-', 'contact-sheet-grey.png');
	for (const f of [...thumbs('thumb-'), ...thumbs('gthumb-')]) fs.rmSync(path.join(PROOF, f));
	fs.rmSync(blank);
	log(`proof: ${path.relative(ROOT, PROOF)}/contact-sheet.png, contact-sheet-grey.png, pages/p-NNN.png (${nPages} pages)`);
}

// ── main ────────────────────────────────────────────────────────────────────────
fs.mkdirSync(WORK, { recursive: true });
if (!flag('skip-build')) buildSite();
else if (!fs.existsSync(path.join(STAGE, 'build'))) die('--skip-build, but there is no previous build in print/.work/site/build');
const server = await serve(path.join(STAGE, 'build'));
const base = `http://127.0.0.1:${server.address().port}`;
let res;
try {
	res = await paginate(base);
} finally {
	server.close();
}
await finish(res.raw, res.result);
const families = checkFonts();
// every mark in the book is vector; a raster image means Chromium flattened something (a pattern,
// a blend over translucency) and it would print soft
{
	const rasters = (fs.readFileSync(OUT).toString('latin1').match(/\/Subtype\s*\/Image/g) ?? []).length;
	if (rasters) warn(`${rasters} raster image object(s) in the PDF: something was flattened (find it with \`mutool trace print/PicoGPT.pdf | grep fill_image\`)`);
}
const kb = Math.round(fs.statSync(OUT).size / 1024);
fs.writeFileSync(path.join(WORK, 'report.json'), JSON.stringify({ pages: res.result.pages, warnings, layout: res.result.diagnostics, fonts: families }, null, 1));
log(`wrote ${path.relative(process.cwd(), OUT) || OUT}: ${res.result.pages} pages, ${kb} kB; fonts: ${families.join(', ')}`);
if (flag('proof')) proof(res.result.pages);
if (warnings.length) {
	console.warn(`\n\x1b[33m${warnings.length} warning(s)\x1b[0m (full list: print/.work/report.json)`);
	if (STRICT) die('--strict: warnings are errors.');
}
console.log(`\x1b[32m✓ ${OUT}\x1b[0m`);

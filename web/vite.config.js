import adapter from '@sveltejs/adapter-static';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const WEB = process.cwd();
const ROOT = process.env.BOOK_ROOT ?? path.resolve(WEB, '..');

// The edition: the main book (default) or a folder at the repository root holding an
// edition.js, book/chapters, figures and traces (see src/lib/editions/book.js).
const EDITION = process.env.BOOK_EDITION ?? 'book';
const EDITION_FILE = EDITION === 'book' ? path.join(WEB, 'src', 'lib', 'editions', 'book.js') : path.join(ROOT, EDITION, 'edition.js');
const edition = await import(pathToFileURL(EDITION_FILE).href);
const FIGURES = path.join(ROOT, edition.PATHS.figures);

/** Reload the page when book prose, Julia sources or exported artifacts change. */
function bookWatch() {
	const watched = [path.join(ROOT, 'book'), path.join(ROOT, 'src'), path.join(ROOT, 'artifacts'), path.join(WEB, 'public', 'traces'), ...(EDITION === 'book' ? [] : [path.join(ROOT, EDITION)])];
	return {
		name: 'picogpt-book-watch',
		configureServer(server) {
			for (const p of watched) server.watcher.add(p);
			server.watcher.on('all', (_event, file) => {
				if (watched.some((p) => file.startsWith(p))) server.ws.send({ type: 'full-reload', path: '*' });
			});
		}
	};
}

export default defineConfig({
	plugins: [
		sveltekit({
			compilerOptions: {
				runes: ({ filename }) => (filename.split(/[/\\]/).includes('node_modules') ? undefined : true)
			},
			adapter: adapter({ pages: 'build', assets: 'build', fallback: undefined, strict: true }),
			files: { assets: 'public' },
			paths: { base: process.env.BASE_PATH ?? '', relative: true },
			prerender: {
				entries: ['*'],
				handleMissingId: 'warn',
				// an edition without traces yet has no /frontis/<name>.svg to prerender; every other route must be seen
				handleUnseenRoutes: ({ routes, message }) => { if (routes.some((r) => !r.startsWith('/frontis/'))) throw new Error(message); }
			},
			version: { name: 'picogpt' }
		}),
		bookWatch()
	],
	resolve: {
		alias: [
			{ find: '#edition', replacement: EDITION_FILE },
			{ find: '#edition-figures', replacement: FIGURES },
			// the package.json "imports" (#lib, #figures, #specimen) resolve only from files under
			// web/; these aliases give an edition's own figures the same names
			{ find: /^#lib$/, replacement: path.join(WEB, 'src', 'lib', 'index.js') },
			{ find: /^#lib\//, replacement: path.join(WEB, 'src', 'lib') + '/' },
			{ find: /^#figures\//, replacement: path.join(WEB, 'src', 'figures') + '/' },
			{ find: /^#specimen$/, replacement: path.join(WEB, 'src', 'specimen', 'index.mjs') },
			{ find: /^#specimen\//, replacement: path.join(WEB, 'src', 'specimen') + '/' }
		]
	},
	server: { fs: { allow: [ROOT] }, port: 5173 }
});

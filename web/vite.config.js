import adapter from '@sveltejs/adapter-static';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
import path from 'node:path';

const ROOT = path.resolve(process.cwd(), '..');

/** Reload the page when book prose, Julia sources or exported artifacts change. */
function bookWatch() {
	const watched = [path.join(ROOT, 'book'), path.join(ROOT, 'src'), path.join(ROOT, 'artifacts'), path.join(process.cwd(), 'public', 'traces')];
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
			prerender: { entries: ['*'], handleMissingId: 'warn' },
			version: { name: 'picogpt' }
		}),
		bookWatch()
	],
	server: { fs: { allow: [ROOT] }, port: 5173 }
});

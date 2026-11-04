import { allFigures } from '#lib/server/book.js';
export function load() {
	const ids = Object.keys(import.meta.glob(['../../figures/**/fig-*.svelte', '!../../figures/**/*.key.svelte'])).map((p) => p.match(/(fig-[\w-]+)\.svelte$/)[1]);
	return { figures: allFigures().map(({ id, num, caption, chapter }) => ({ id, num, chapter, caption: caption.replace(/<[^>]+>/g, '').slice(0, 160), built: ids.includes(id) })) };
}

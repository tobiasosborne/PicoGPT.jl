import { listChapters } from '#lib/server/book.js';
export function load() {
	return { chapters: listChapters().map(({ num, slug, title, gist, exists, unnumbered }) => ({ num, slug, title, gist, exists, unnumbered: !!unnumbered })) };
}

import { listChapters, traceFor } from '#lib/server/book.js';
import { specimenData } from '#lib/server/specimen-data.js';

export function load() {
	return {
		chapters: listChapters().map(({ num, slug, title, gist, exists, unnumbered }) => ({ num, slug, title, gist, exists, unnumbered: !!unnumbered })),
		trace: traceFor('cover'),
		specimen: specimenData()
	};
}

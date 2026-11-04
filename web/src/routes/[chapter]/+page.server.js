import { error } from '@sveltejs/kit';
import { listChapters, renderChapter, neighbours, traceFor } from '#lib/server/book.js';
import { specimenData } from '#lib/server/specimen-data.js';

export function entries() {
	return listChapters().filter((c) => c.exists).map((c) => ({ chapter: c.slug }));
}

export function load({ params }) {
	const chapter = renderChapter(params.chapter);
	if (!chapter) error(404, `No chapter “${params.chapter}”`);
	return {
		chapter,
		nav: neighbours(params.chapter),
		trace: traceFor(params.chapter),
		specimen: specimenData()
	};
}

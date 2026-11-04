import { listChapters, renderChapter, traceFor } from '#lib/server/book.js';
import { specimenData } from '#lib/server/specimen-data.js';

// The whole book, in reading order, for the print edition (CSS Paged Media).
export function load() {
	const list = listChapters();
	const chapters = list.filter((c) => c.exists).map((c) => {
		const ch = renderChapter(c.slug);
		// in one document, links between chapters become in-document anchors
		for (const seg of ch.segments) if (seg.t === 'html') seg.html = seg.html.replace(/href="\.\.\/[\w-]+\/#/g, 'href="#').replace(/href="\.\.\/([\w-]+)\/"/g, 'href="#ch-$1"');
		return { ...ch, trace: traceFor(c.slug) };
	});
	return {
		chapters,
		contents: list.map(({ num, slug, title, gist, exists, unnumbered }) => ({ num, slug, title, gist, exists, unnumbered: !!unnumbered })),
		coverTrace: traceFor('cover'),
		specimen: specimenData(),
		built: new Date().toISOString().slice(0, 10)
	};
}

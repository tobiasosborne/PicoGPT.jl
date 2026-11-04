import { listChapters, renderChapter, traceFor } from '#lib/server/book.js';
import { specimenData } from '#lib/server/specimen-data.js';

// The whole book, in reading order, for the print edition (CSS Paged Media, print/build.mjs).
export function load() {
	const list = listChapters();
	const sectionsOf = {};
	const chapters = list.filter((c) => c.exists).map((c) => {
		const ch = renderChapter(c.slug);
		const sections = [];
		for (const seg of ch.segments) {
			if (seg.t !== 'html') continue;
			// in one document, links between chapters become in-document anchors
			seg.html = seg.html.replace(/href="\.\.\/[\w-]+\/#/g, 'href="#').replace(/href="\.\.\/([\w-]+)\/"/g, 'href="#ch-$1"');
			// the sections, for the contents
			for (const m of seg.html.matchAll(/<h2\b[^>]*\bid="([^"]+)"[^>]*>([\s\S]*?)<\/h2>/g)) {
				const num = m[2].match(/<span class="secnum">([^<]*)<\/span>/)?.[1] ?? '';
				sections.push({ id: m[1], num, html: m[2].replace(/<span class="secnum">[^<]*<\/span>\s*/, '').trim() });
			}
		}
		sectionsOf[c.slug] = sections;
		return { ...ch, trace: traceFor(c.slug) };
	});
	return {
		chapters,
		contents: list.map(({ num, slug, title, gist, exists, unnumbered }) => ({ num, slug, title, gist, exists, unnumbered: !!unnumbered, sections: sectionsOf[slug] ?? [] })),
		coverTrace: traceFor('cover'),
		specimen: specimenData(),
		// the proof date: SOURCE_DATE_EPOCH (reproducible builds) or today
		built: (process.env.SOURCE_DATE_EPOCH ? new Date(+process.env.SOURCE_DATE_EPOCH * 1000) : new Date()).toISOString().slice(0, 10)
	};
}

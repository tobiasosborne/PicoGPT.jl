import { listChapters, renderChapter, traceFor, bookIndex } from '#lib/server/book.js';
import { specimenData } from '#lib/server/specimen-data.js';
import { artifactsFor } from '#lib/server/artifacts.js';

// The whole book, in reading order, for the print edition (CSS Paged Media, print/build.mjs).
export function load() {
	const list = listChapters();
	const sectionsOf = {};
	// One document holds every chapter, so an id may occur twice: pandoc's automatic ones
	// ("exercises") always do, and authors' ids sometimes collide. The chapter that the book's
	// cross-reference index assigns an id to keeps it (references point there); every other
	// occurrence is prefixed with its chapter's slug.
	const rendered = list.filter((c) => c.exists).map((c) => ({ c, ch: renderChapter(c.slug) }));
	const seen = {};
	for (const { c, ch } of rendered) for (const seg of ch.segments) if (seg.t === 'html') for (const m of seg.html.matchAll(/\sid="([^"]+)"/g)) (seen[m[1]] ??= new Set()).add(c.slug);
	const where = bookIndex().data.where ?? {};
	const chapters = rendered.map(({ c, ch }) => {
		ch = { ...ch, segments: ch.segments.map((seg) => ({ ...seg })) };
		const sections = [];
		for (const seg of ch.segments) {
			if (seg.t !== 'html') continue;
			seg.html = seg.html.replace(/(\sid=")([^"]+)"/g, (all, pre, id) => (seen[id]?.size > 1 && where[id] !== String(c.num) ? `${pre}${c.slug}--${id}"` : all));
			// in one document, links between chapters become in-document anchors
			seg.html = seg.html.replace(/href="\.\.\/[\w-]+\/#/g, 'href="#').replace(/href="\.\.\/([\w-]+)\/"/g, 'href="#open-$1"');
			// the sections, for the contents
			for (const m of seg.html.matchAll(/<h2\b[^>]*\bid="([^"]+)"[^>]*>([\s\S]*?)<\/h2>/g)) {
				const num = m[2].match(/<span class="secnum">([^<]*)<\/span>/)?.[1] ?? '';
				sections.push({ id: m[1], num, html: m[2].replace(/<span class="secnum">[^<]*<\/span>\s*/, '').trim() });
			}
		}
		sectionsOf[c.slug] = sections;
		return { ...ch, trace: traceFor(c.slug) };
	});
	const clashes = Object.entries(seen).filter(([id, set]) => set.size > 1 && where[id] !== undefined).map(([id, set]) => `${id} (${[...set].join(', ')})`);
	if (clashes.length) console.warn(`\x1b[33m[book]\x1b[0m print: section ids used in more than one chapter: ${clashes.join('; ')}`);
	return {
		chapters,
		contents: list.map(({ num, slug, title, gist, exists, unnumbered }) => ({ num, slug, title, gist, exists, unnumbered: !!unnumbered, sections: sectionsOf[slug] ?? [] })),
		coverTrace: traceFor('cover'),
		specimen: specimenData(),
		artifacts: artifactsFor(chapters.flatMap((c) => c.figures)),
		// the proof date: SOURCE_DATE_EPOCH (reproducible builds) or today
		built: (process.env.SOURCE_DATE_EPOCH ? new Date(+process.env.SOURCE_DATE_EPOCH * 1000) : new Date()).toISOString().slice(0, 10)
	};
}

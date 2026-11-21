import { allFigures } from '#lib/server/book.js';
import { figureIds } from '#figures/registry.js';
export function load() {
	return { figures: allFigures().map(({ id, num, caption, chapter }) => ({ id, num, chapter, caption: caption.replace(/<[^>]+>/g, '').slice(0, 160), built: figureIds.includes(id) })) };
}

import { loadFigures } from '#figures/registry.js';
export async function load({ data }) {
	return { ...data, figures: await loadFigures(data.chapters.flatMap((c) => c.figures)) };
}

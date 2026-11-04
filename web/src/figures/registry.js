// registry.js — figure id → component, by file name.
//
// A figure `::: {.figure #fig-attention-reads}` is implemented by ANY file named
// `fig-attention-reads.svelte` under src/figures/ (by convention in the folder of
// the chapter that introduces it, e.g. src/figures/05-attention/). No list to edit.
// The module may export from <script module>:
//   meta = { size: 'main'|'wide'|'full', title }
//   Key  — a component drawing the "how to read" key, set in the margin beside the caption
//          (export it only from the figure where the glyph first appears in the book).

const modules = import.meta.glob(['./**/fig-*.svelte', '!./**/*.key.svelte']);

const byId = new Map(
	Object.entries(modules).map(([path, loader]) => [path.match(/\/(fig-[\w-]+)\.svelte$/)[1], { path, loader }])
);

export const figureIds = [...byId.keys()].sort();

/** Load the components a page needs. Missing ids are simply absent. */
export async function loadFigures(ids) {
	const out = {};
	await Promise.all(
		[...new Set(ids)].map(async (id) => {
			const entry = byId.get(id);
			if (!entry) return;
			const mod = await entry.loader();
			out[id] = { component: mod.default, meta: mod.meta ?? {}, Key: mod.Key ?? null };
		})
	);
	return out;
}

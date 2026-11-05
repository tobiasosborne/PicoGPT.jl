import { error } from '@sveltejs/kit';
import { allFigures } from '#lib/server/book.js';
import { specimenData } from '#lib/server/specimen-data.js';
import { artifactsFor } from '#lib/server/artifacts.js';

export function entries() {
	return allFigures().map((f) => ({ id: f.id }));
}

export function load({ params }) {
	const fig = allFigures().find((f) => f.id === params.id);
	if (!fig) error(404, `No figure ${params.id} in the book`);
	// Caption and note links stay chapter-relative here; the page resolves them against the base.
	return { fig, specimen: specimenData(), artifacts: artifactsFor([fig.id]) };
}

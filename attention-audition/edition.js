import { OUTLINE as bookOutline, BOOK as mainBook } from '../web/src/lib/editions/book.js';

export const OUTLINE = bookOutline.map((chapter) =>
	chapter.slug === 'attention' ? chapter : { ...chapter, referenceOnly: true }
);

export const BOOK = {
	...mainBook,
	title: 'Attention, from the top down',
	subtitle: 'A draft revision of Chapter 5 of PicoGPT, for its author to read',
	subject: 'The Attention chapter of PicoGPT. Draft print edition.'
};

export const PATHS = {
	chapters: 'attention-audition/book/chapters',
	figures: 'attention-audition/figures',
	traces: 'attention-audition/traces'
};

export const FRONT = {
	theta: false,
	facts: false,
	map: false,
	provenance: 'This Chapter 5 proof is typeset from the audition edition’s Markdown using the <i>PicoGPT</i> print pipeline.'
};

export const PRINT = {
	out: 'attention-audition/AttentionDraft.pdf',
	work: 'attention-audition/.work',
	proof: 'attention-audition/proof'
};

const MIXTURE = 'attention-audition/artifacts/mixture.json';

// Figures get the whole export except the readers' pair tables (p_by_pair, the bulk of
// the file); the figure readers keep theirs.
export function artifacts(ids, read) {
	if (!ids.length) return {};
	const mixture = read(MIXTURE);
	if (!mixture) return { mixture };
	return { mixture: { ...mixture, readers: mixture.readers.map(({ p_by_pair, ...r }) => r) } };
}

export function proseValues(read, mainValues) {
	const values = mainValues();
	let mixture;
	try {
		mixture = read(MIXTURE);
	} catch (error) {
		if (error.code !== 'ENOENT') throw error;
		return values;
	}
	const loss = mixture.summary.mean_cross_entropy;
	return {
		...values,
		mixtureReaderCount: String(mixture.reader_count),
		mixtureLossActual: loss.p.toFixed(4),
		mixtureLossArithmetic: loss.mix.toFixed(4),
		mixtureLossGeometric: loss.geo.toFixed(4),
		// mean total variation of the joint (two-head) mixtures from the model's prediction, over all readers
		mixtureTvArithmetic: mixture.summary.distances.tv_p_mix.mean.toFixed(2),
		mixtureTvGeometric: mixture.summary.distances.tv_p_geo.mean.toFixed(4)
	};
}

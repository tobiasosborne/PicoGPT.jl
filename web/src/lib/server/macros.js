// Book-wide TeX macros (Temml). Keep them few and obvious; prose should read as
// ordinary LaTeX. Documented in DESIGN.md §Typography/maths.
export const MACROS = {
	'\\R': '\\mathbb{R}',
	'\\T': '\\mathsf{T}',                    // transpose: A^\T
	'\\softmax': '\\operatorname{softmax}',
	'\\LN': '\\operatorname{LN}',
	'\\relu': '\\operatorname{relu}',
	'\\lse': '\\operatorname{lse}',
	'\\diag': '\\operatorname{diag}',
	'\\one': '\\mathbf{1}',
	'\\E': '\\mathbb{E}',
	'\\KL': '\\operatorname{KL}',
	'\\adj': '\\bar{#1}',                    // \adj{x} — the adjoint x̄
	'\\inner': '\\langle #1,\\, #2\\rangle'
};

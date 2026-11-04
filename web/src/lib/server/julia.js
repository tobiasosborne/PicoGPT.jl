// julia.js — lex, extract and highlight Julia source for `::: {.code …}` excerpts.
//
// The Julia sources are written by another hand and change often, so extraction
// works on tokens (strings, comments and brackets are understood), never on
// line-number guesses. Supported selectors (pandoc div attributes):
//
//   fn="name"        every method of `name` (long `function name(...) … end` and short
//                    `name(...) = …` forms), in file order. `method=2` picks one.
//   struct="Name"    a (mutable) struct definition
//   region="name"    the lines between `# region: name` and `# endregion`
//   lines="12-40"    an explicit line range (1-based, inclusive)
//   doc=true         include the docstring directly above a definition
//   comments=true    include the comment lines directly above a definition
//
// Every selector returns chunks {start, end, code} with 1-based line numbers so
// the book can cite `src/file.jl:start–end` exactly.

const OPENERS = new Set(['function', 'macro', 'if', 'for', 'while', 'let', 'begin', 'quote', 'struct', 'module', 'baremodule', 'do', 'try']);
const KEYWORDS = new Set([...OPENERS, 'end', 'else', 'elseif', 'catch', 'finally', 'return', 'in', 'isa', 'where', 'const', 'global', 'local', 'using', 'import', 'export', 'mutable', 'abstract', 'primitive', 'type', 'break', 'continue', 'public']);
const LITERALS = new Set(['true', 'false', 'nothing', 'missing', 'Inf', 'NaN', 'π', 'ℯ']);
const ID_START = /[\p{L}\p{Nl}_∇∂√]/u;
const ID_CONT = /[\p{L}\p{N}\p{M}\p{Pc}\p{Sk}!′″‴_⁰-ₜᵢ-ᵪⱼ]/u;
const ADJOINT = /[̄̅]/u; // combining macron / overline: x̄ names an adjoint

/** Tokenize Julia. Tokens: {type, text, start, end}. */
export function lex(src) {
	const out = [];
	let i = 0;
	const n = src.length;
	const push = (type, start, end) => out.push({ type, text: src.slice(start, end), start, end });
	const prevSig = () => {
		for (let k = out.length - 1; k >= 0; k--) if (out[k].type !== 'ws' && out[k].type !== 'nl') return out[k];
		return null;
	};
	while (i < n) {
		const c = src[i];
		const s = i;
		if (c === '\n') { push('nl', i, ++i); continue; }
		if (c === ' ' || c === '\t' || c === '\r') { while (i < n && /[ \t\r]/.test(src[i])) i++; push('ws', s, i); continue; }
		if (c === '#') {
			if (src[i + 1] === '=') { // nestable block comment
				let depth = 1; i += 2;
				while (i < n && depth > 0) {
					if (src[i] === '#' && src[i + 1] === '=') { depth++; i += 2; }
					else if (src[i] === '=' && src[i + 1] === '#') { depth--; i += 2; }
					else i++;
				}
				push('comment', s, i); continue;
			}
			while (i < n && src[i] !== '\n') i++;
			push('comment', s, i); continue;
		}
		if (c === '"' || c === '`') { i = readString(src, i); push('string', s, i); continue; }
		if (c === "'") {
			const p = prevSig();
			const transpose = p && p.end === i && (p.type === 'ident' || p.type === 'number' || p.type === 'literal' || (p.type === 'punct' && /[)\]}]/.test(p.text)) || (p.type === 'op' && p.text === "'") || (p.type === 'keyword' && p.text === 'end'));
			if (transpose) { push('op', i, ++i); continue; }
			i++;
			while (i < n && src[i] !== "'" && src[i] !== '\n') i += src[i] === '\\' ? 2 : 1;
			if (src[i] === "'") i++;
			push('char', s, i); continue;
		}
		if (c === '@') {
			i++;
			while (i < n && (ID_CONT.test(src[i]) || src[i] === '.')) i++;
			push('macro', s, i); continue;
		}
		if (/[0-9]/.test(c) || (c === '.' && /[0-9]/.test(src[i + 1] ?? ''))) {
			if (c === '0' && /[xob]/.test(src[i + 1] ?? '')) { i += 2; while (i < n && /[0-9a-fA-F_]/.test(src[i])) i++; }
			else {
				while (i < n && /[0-9_]/.test(src[i])) i++;
				if (src[i] === '.' && src[i + 1] !== '.' && !/[\p{L}_]/u.test(src[i + 1] ?? '')) { i++; while (i < n && /[0-9_]/.test(src[i])) i++; }
				if (/[eEf]/.test(src[i] ?? '') && /[-+0-9]/.test(src[i + 1] ?? '')) { i += 2; while (i < n && /[0-9]/.test(src[i])) i++; }
			}
			push('number', s, i); continue;
		}
		if (ID_START.test(c)) {
			i++;
			while (i < n && ID_CONT.test(src[i])) i++;
			const text = src.slice(s, i);
			if (src[i] === '"' && /^(raw|r|b|s|L|html|md|var|big|v)$/.test(text)) { i = readString(src, i); push('string', s, i); continue; }
			const p = prevSig();
			const field = p && p.type === 'op' && p.text === '.' && p.end === s;
			if (!field && KEYWORDS.has(text)) push('keyword', s, i);
			else if (LITERALS.has(text)) push('literal', s, i);
			else push('ident', s, i);
			continue;
		}
		if (c === ':' ) {
			const p = prevSig();
			const symbolish = (!p || (p.type === 'punct' && /[(\[{,;]/.test(p.text)) || (p.type === 'op' && p.text !== "'") || p.type === 'keyword') && ID_START.test(src[i + 1] ?? '');
			if (symbolish && src[i + 1] !== ':') { i++; while (i < n && ID_CONT.test(src[i])) i++; push('symbol', s, i); continue; }
		}
		if (/[()[\]{},;]/.test(c)) { push('punct', i, ++i); continue; }
		// operators: greedy run of operator characters
		i++;
		while (i < n && /[=+\-*/\\^<>!&|%.~:÷⋅×∘∈∉⊆⊂≤≥≠≈⊗⊕→←↦√∑∏]/.test(src[i]) && !/[()[\]{},;"'#\s]/.test(src[i])) i++;
		push('op', s, i);
	}
	return out;
}

function readString(src, i) {
	const q = src[i];
	const triple = src.startsWith(q.repeat(3), i);
	const close = triple ? q.repeat(3) : q;
	i += close.length;
	while (i < src.length) {
		if (src[i] === '\\') { i += 2; continue; }
		if (src[i] === '$' && src[i + 1] === '(') { // interpolation: skip balanced parens, nested strings
			let depth = 0;
			i++;
			while (i < src.length) {
				if (src[i] === '(') depth++;
				else if (src[i] === ')') { depth--; if (depth === 0) { i++; break; } }
				else if (src[i] === '"') { i = readString(src, i); continue; }
				i++;
			}
			continue;
		}
		if (src.startsWith(close, i)) return i + close.length;
		i++;
	}
	return i;
}

/** Find the token index of the `end` closing the opener at token index k (or -1). */
function matchEnd(tokens, k) {
	let depth = 0; // block depth
	let brackets = 0;
	for (let j = k; j < tokens.length; j++) {
		const t = tokens[j];
		if (t.type === 'punct') {
			if ('([{'.includes(t.text)) brackets++;
			else if (')]}'.includes(t.text)) brackets = Math.max(0, brackets - 1);
			continue;
		}
		if (t.type !== 'keyword' || brackets > 0) continue;
		if (OPENERS.has(t.text) || ((t.text === 'abstract' || t.text === 'primitive') && nextSig(tokens, j)?.text === 'type')) depth++;
		else if (t.text === 'end') { depth--; if (depth === 0) return j; }
	}
	return -1;
}

function nextSig(tokens, j) {
	for (let k = j + 1; k < tokens.length; k++) if (tokens[k].type !== 'ws' && tokens[k].type !== 'nl') return tokens[k];
	return null;
}
function nextSigIndex(tokens, j) {
	for (let k = j + 1; k < tokens.length; k++) if (tokens[k].type !== 'ws' && tokens[k].type !== 'nl') return k;
	return -1;
}

/** Bracket depth at each token, and whether each token starts a logical line. */
function structure(tokens) {
	const depth = new Int32Array(tokens.length);
	const lineStart = new Uint8Array(tokens.length);
	let d = 0, atStart = true;
	for (let j = 0; j < tokens.length; j++) {
		const t = tokens[j];
		if (t.type === 'punct' && ')]}'.includes(t.text)) d = Math.max(0, d - 1);
		depth[j] = d;
		if (t.type === 'punct' && '([{'.includes(t.text)) d++;
		if (t.type === 'nl' || (t.type === 'punct' && t.text === ';')) { if (d === 0) atStart = true; continue; }
		if (t.type === 'ws' || t.type === 'comment') continue;
		lineStart[j] = atStart ? 1 : 0;
		if (t.type !== 'macro') atStart = false; // `@inline f(x) = …` still starts a definition
	}
	return { depth, lineStart };
}

function lineOf(src, offset) {
	let line = 1;
	for (let i = 0; i < offset && i < src.length; i++) if (src[i] === '\n') line++;
	return line;
}

/** Name tokens at k: `name`, `Mod.name`, `Base.:+` → [lastNameTokenIndex, nameText]. */
function readName(tokens, k) {
	let j = k, name = '';
	if (tokens[j]?.type !== 'ident') return null;
	name = tokens[j].text;
	while (tokens[j + 1]?.type === 'op' && tokens[j + 1].text === '.' && tokens[j + 2]?.type === 'ident') { j += 2; name = tokens[j].text; }
	return [j, name];
}

function definitions(src, tokens) {
	const { depth, lineStart } = structure(tokens);
	const defs = [];
	for (let k = 0; k < tokens.length; k++) {
		const t = tokens[k];
		if (depth[k] !== 0) continue;
		// long form: function name(…) … end   /  macro name(…) … end
		if (t.type === 'keyword' && (t.text === 'function' || t.text === 'macro')) {
			const ni = nextSigIndex(tokens, k);
			const nm = ni >= 0 ? readName(tokens, ni) : null;
			const endIdx = matchEnd(tokens, k);
			if (nm && endIdx > 0) defs.push({ kind: t.text, name: nm[1], startTok: firstOnLine(tokens, k), endTok: endIdx });
			continue;
		}
		// struct / mutable struct
		if (t.type === 'keyword' && t.text === 'struct') {
			const ni = nextSigIndex(tokens, k);
			const nm = ni >= 0 ? readName(tokens, ni) : null;
			const endIdx = matchEnd(tokens, k);
			const prev = prevSigIndex(tokens, k);
			const startTok = prev >= 0 && tokens[prev].text === 'mutable' ? prev : k;
			if (nm && endIdx > 0) defs.push({ kind: 'struct', name: nm[1], startTok: firstOnLine(tokens, startTok), endTok: endIdx });
			continue;
		}
		// short form: name(args…) [where …] [:: T] = expr
		if (t.type === 'ident' && lineStart[k]) {
			const nm = readName(tokens, k);
			if (!nm) continue;
			let j = nm[0] + 1;
			if (tokens[j]?.type === 'punct' && tokens[j].text === '{') j = closeBracket(tokens, j) + 1; // f{T}(x)
			if (!(tokens[j]?.type === 'punct' && tokens[j].text === '(')) continue;
			j = closeBracket(tokens, j);
			if (j < 0) continue;
			// skip `where {…}` / `where T` / `::T` until '='
			let q = nextSigIndex(tokens, j);
			let guard = 0;
			while (q >= 0 && guard++ < 50) {
				const tq = tokens[q];
				if (tq.type === 'op' && tq.text === '=') break;
				if (tq.type === 'nl') { q = -1; break; }
				if (tq.type === 'punct' && '([{'.includes(tq.text)) { q = nextSigIndex(tokens, closeBracket(tokens, q)); continue; }
				if (tq.type === 'keyword' && tq.text === 'where' || tq.type === 'op' && (tq.text === '::' || tq.text === '<:') || tq.type === 'ident') { q = nextSigIndex(tokens, q); continue; }
				q = -1; break;
			}
			if (q < 0 || tokens[q].text !== '=') continue;
			const endTok = endOfExpression(tokens, q + 1);
			defs.push({ kind: 'short', name: nm[1], startTok: firstOnLine(tokens, k), endTok });
		}
	}
	for (const d of defs) {
		d.start = lineOf(src, tokens[d.startTok].start);
		d.end = lineOf(src, tokens[d.endTok].end);
	}
	return defs;
}

function prevSigIndex(tokens, j) {
	for (let k = j - 1; k >= 0; k--) if (tokens[k].type !== 'ws' && tokens[k].type !== 'nl') return k;
	return -1;
}
function firstOnLine(tokens, k) { // include leading macros (@inline) on the same line
	let j = k;
	while (j > 0 && tokens[j - 1].type !== 'nl') j--;
	return j;
}
function closeBracket(tokens, j) {
	let d = 0;
	for (let k = j; k < tokens.length; k++) {
		const t = tokens[k];
		if (t.type !== 'punct') continue;
		if ('([{'.includes(t.text)) d++;
		else if (')]}'.includes(t.text)) { d--; if (d === 0) return k; }
	}
	return -1;
}
/** End of a short-form body: the first newline at bracket depth 0 that does not continue the expression. */
function endOfExpression(tokens, j) {
	let d = 0, last = j;
	for (let k = j; k < tokens.length; k++) {
		const t = tokens[k];
		if (t.type === 'punct' && '([{'.includes(t.text)) d++;
		else if (t.type === 'punct' && ')]}'.includes(t.text)) d--;
		else if (t.type === 'keyword' && OPENERS.has(t.text) && d === 0) { const e = matchEnd(tokens, k); if (e > 0) { k = e; last = e; continue; } }
		if (t.type === 'nl' && d <= 0) {
			const p = tokens[prevSigIndex(tokens, k)];
			const continues = p && ((p.type === 'op' && p.text !== "'") || (p.type === 'punct' && p.text === ','));
			if (!continues) return last;
		}
		if (t.type !== 'ws' && t.type !== 'nl' && t.type !== 'comment') last = k;
	}
	return last;
}

/** Extend a definition upwards over its docstring and/or contiguous comment lines. */
function extendUp(lines, start, { doc, comments }) {
	let s = start;
	if (comments) while (s > 1 && /^\s*#/.test(lines[s - 2])) s--;
	if (doc && s > 1 && /"""\s*$/.test(lines[s - 2])) {
		let k = s - 2; // 0-based index of the closing """
		if (/^\s*"""\s*$/.test(lines[k])) { k--; while (k >= 0 && !/^\s*"""/.test(lines[k])) k--; }
		if (k >= 0) s = k + 1;
	} else if (doc && s > 1 && /^\s*".*"\s*$/.test(lines[s - 2])) s--;
	return s;
}

/**
 * Extract an excerpt. Returns {chunks:[{start,end,code}], label} or {error}.
 */
export function extract(src, spec) {
	const lines = src.split('\n');
	const tokens = lex(src);
	const opts = { doc: spec.doc === 'true' || spec.doc === true, comments: spec.comments === 'true' || spec.comments === true };
	let ranges = [];
	let label = '';
	if (spec.lines) {
		const m = String(spec.lines).match(/^(\d+)\s*[-–:]\s*(\d+)$/);
		if (!m) return { error: `bad lines="${spec.lines}"` };
		ranges = [[+m[1], Math.min(+m[2], lines.length)]];
		label = `lines ${m[1]}–${m[2]}`;
	} else if (spec.region) {
		const open = lines.findIndex(l => new RegExp(`^\\s*#\\s*region:?\\s*${escapeRe(spec.region)}\\s*$`).test(l));
		if (open < 0) return { error: `no "# region: ${spec.region}"` };
		const close = lines.findIndex((l, k) => k > open && /^\s*#\s*endregion\b/.test(l));
		if (close < 0) return { error: `region "${spec.region}" is not closed` };
		ranges = [[open + 2, close]];
		label = spec.region;
	} else if (spec.fn || spec.struct) {
		const name = spec.fn ?? spec.struct;
		const defs = definitions(src, tokens).filter(d => d.name === name && (spec.struct ? d.kind === 'struct' : d.kind !== 'struct'));
		if (!defs.length) return { error: `no ${spec.struct ? 'struct' : 'function'} named ${name}` };
		let chosen = defs;
		if (spec.method) {
			const m = +spec.method;
			if (!(m >= 1 && m <= defs.length)) return { error: `${name} has ${defs.length} method(s); method=${spec.method}` };
			chosen = [defs[m - 1]];
		}
		ranges = chosen.map(d => [extendUp(lines, d.start, opts), d.end]);
		label = spec.struct ? `struct ${name}` : name;
	} else return { error: 'no selector (fn, struct, region or lines)' };
	const chunks = ranges.map(([a, b]) => ({ start: a, end: b, code: dedent(lines.slice(a - 1, b)).join('\n') }));
	return { chunks, label };
}

function dedent(ls) {
	const indent = Math.min(...ls.filter(l => l.trim()).map(l => l.match(/^[ \t]*/)[0].length));
	return Number.isFinite(indent) ? ls.map(l => l.slice(indent)) : ls;
}
function escapeRe(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }

const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

/** Highlight a code string to HTML: spans with one-letter classes, styled in book.css. */
export function highlight(code) {
	const tokens = lex(code);
	let html = '';
	let defNext = false;
	for (let k = 0; k < tokens.length; k++) {
		const t = tokens[k];
		const x = esc(t.text);
		switch (t.type) {
			case 'comment': html += `<span class="c">${x}</span>`; break;
			case 'string': case 'char': html += `<span class="s">${x}</span>`; break;
			case 'number': case 'literal': html += `<span class="n">${x}</span>`; break;
			case 'macro': html += `<span class="m">${x}</span>`; break;
			case 'symbol': html += `<span class="y">${x}</span>`; break;
			case 'keyword':
				html += `<span class="k">${x}</span>`;
				defNext = t.text === 'function' || t.text === 'struct' || t.text === 'macro';
				break;
			case 'ident': {
				const adj = ADJOINT.test(t.text.normalize('NFD'));
				const cls = [defNext ? 'd' : '', adj ? 'a' : ''].filter(Boolean).join(' ');
				html += cls ? `<span class="${cls}">${x}</span>` : x;
				if (defNext && !(tokens[k + 1]?.type === 'op' && tokens[k + 1].text === '.')) defNext = false;
				break;
			}
			case 'ws': html += x; break;
			case 'nl': html += '\n'; break;
			default: html += x; if (t.type !== 'ws') defNext = defNext && t.text === '.';
		}
	}
	return html;
}

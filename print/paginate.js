// print/paginate.js — runs inside headless Chromium on the built /print/ page, after
// <html data-print-ready="1">, with paged.js (window.Paged) already loaded.
//
//   window.__picoPaginate({ xpages }) → { pages, ms, trim, meta, built, outline, xrefs, diagnostics }
//                                      | { error }
//
// It paginates the page's own HTML and CSS with the Paged.js Previewer (DESIGN.md §8: the
// polyfill does not work on a hydrated page) and adds what Paged.js 0.4 lacks:
//
//   1. Figure floats. A figure that does not fit in the rest of its page is taken out of the
//      flow and set at the head of the next page, and the text that follows fills the hole
//      (TeX's deferred float). A figure with only the tail of a paragraph above it on its page
//      moves above that tail, to the head of the page. Figures never split.
//   2. Running heads and folios, drawn as DOM on each page: the folio at the outer corner
//      over the notes column, the chapter (verso) or the first section on the page (recto)
//      over the main column. Openings get a drop folio; blank, front and plate pages none.
//   3. The first block on a page loses its top margin (Paged.js keeps it).
//   4. Cross-reference page numbers. `xpages` maps an xref's index to the page of its target;
//      build.mjs runs passes until those numbers stop changing (as LaTeX reruns).
//   5. An audit: overflow, split figures, colliding notes, holes, missing pieces, lost floats.
(() => {
	const PX_PER_MM = 96 / 25.4;

	// ── tuning ──────────────────────────────────────────────────────────────────
	const SNAP_TOP_LINES = 3.2;   // ≤ this much text above a figure on its page → figure to the page head
	const MIN_HOLE_LINES = 3.5;   // a figure that leaves a smaller gap simply moves to the next page
	const UNDERFULL = 0.22;       // report pages whose text stops this far (fraction) above the foot

	const diag = { errors: [], warnings: [], floats: [] };
	let refs = 0;
	const NOTES = []; // the margin notes' HTML, by mark
	const newRef = () => `pg-ref-${++refs}`; // fresh data-ref for nodes the float handler splits or moves
	const pageNo = (el) => {
		const p = el.closest('.pagedjs_page');
		return p ? +p.dataset.pageNumber : 0;
	};
	// plain text of a heading: no section number, no TeX annotations from the MathML
	const plain = (el) => {
		const c = el.cloneNode(true);
		c.querySelectorAll('.secnum, annotation, .opening__num').forEach((n) => n.remove());
		return c.textContent.replace(/\s+/g, ' ').trim();
	};
	const richHTML = (el) => {
		const c = el.cloneNode(true);
		c.querySelectorAll('.secnum').forEach((n) => n.remove());
		c.querySelectorAll('[id]').forEach((n) => n.removeAttribute('id'));
		return c.innerHTML.trim();
	};

	// ── pre-pass on the live, laid-out DOM ──────────────────────────────────────────
	function prepare(root, xpages) {
		const blockH = 206.3 * PX_PER_MM;
		const mainW = 104 * PX_PER_MM;
		// Paint servers shared by every figure (the covector hatch: +layout.svelte) live outside the
		// book's root; carry them in, or every hatched mark prints empty. They go inside the first
		// page's section, where a zero-size element cannot make a page of its own.
		const defsHost = root.querySelector('.pp') ?? root;
		for (const svg of document.querySelectorAll('svg')) {
			if (root.contains(svg) || !svg.querySelector('pattern[id], linearGradient[id], radialGradient[id], marker[id], symbol[id]')) continue;
			const c = svg.cloneNode(true);
			c.setAttribute('class', 'pg-defs');
			defsHost.prepend(c);
		}
		for (const h of root.querySelectorAll('.chapter-body h2')) h.dataset.rh = plain(h);
		// Margin notes leave the flow: each becomes an empty mark, and the note itself is set in
		// the margin of whichever page its mark lands on (placeNotes), after that page is built.
		// Paged.js never sees a note, so it can neither split one nor push text away for one.
		for (const n of root.querySelectorAll('.chapter-body .noted > .note, .chapter-body .sidenote')) {
			if (n.closest('.fig')) continue;
			const mark = document.createElement(n.classList.contains('sidenote') ? 'span' : 'div');
			mark.className = 'pg-note-mark';
			mark.dataset.note = String(NOTES.length);
			NOTES.push(n.outerHTML);
			n.replaceWith(mark);
		}
		// Freeze the dictionary into page-sized pairs of columns. Paged.js loses
		// entries when nested CSS columns fragment across pages. Measure the same
		// paragraphs at the same column width, retaining the original reading order.
		for (const dictionary of root.querySelectorAll('.dict')) {
			const items = [...dictionary.querySelectorAll('.dict__head, .dict__entry')];
			const units = [];
			for (let i = 0; i < items.length; i++) {
				if (items[i].classList.contains('dict__head')) {
					const start = document.createElement('div');
					start.append(items[i], items[++i]); units.push(start);
				} else units.push(items[i]);
			}
			dictionary.replaceChildren(); dictionary.style.columnCount = 'auto';
			// Reserve the width of the page numbers before target-counter is resolved.
			const hint = document.createElement('style');
			hint.textContent = '.print-edition-root .dict__where::after { content: ", p. 000"; }';
			document.head.append(hint);
			// The dictionary starts with its heading and introduction on the first page.
			// Reserve that space when filling its first two columns.
			let firstReserve = 0;
			for (let n = dictionary.previousElementSibling; n; n = n.previousElementSibling) {
				const style = getComputedStyle(n);
				firstReserve += n.getBoundingClientRect().height + parseFloat(style.marginTop) + parseFloat(style.marginBottom);
				if (n.matches('h2')) break;
			}
			const pages = [];
			let columns, col, side = 0;
			const nextPage = () => {
				columns = document.createElement('div'); columns.className = 'pg-dict-columns';
				for (let k = 0; k < 2; k++) { const c = document.createElement('div'); c.className = 'pg-dict-column'; columns.append(c); }
				dictionary.append(columns); pages.push(columns); side = 0; col = columns.children[0];
			};
			nextPage();
			for (const unit of units) {
				col.append(unit);
				if (col.getBoundingClientRect().height > blockH - 8 * PX_PER_MM - (pages.length === 1 ? firstReserve : 0) && col.children.length > 1) {
					unit.remove();
					if (side === 0) { side = 1; col = columns.children[1]; } else nextPage();
					col.append(unit);
				}
			}
			hint.remove();
			const groups = pages.map((columns) => {
				const group = document.createElement('div'); group.className = 'dict pg-dict-page';
				group.setAttribute('role', 'list'); group.append(columns); return group;
			});
			dictionary.replaceWith(...groups);
		}
		for (const body of root.querySelectorAll('.chapter-body')) {
			// the chapter's first paragraph opens with a line of small capitals
			const first = body.querySelector(':scope > p, :scope > .noted > p');
			if (first) first.classList.add('pg-firstpara');
			// a paragraph keeps the indent it has in the unbroken text (on a page of its own, the
			// sibling selectors that give it would see the wrong neighbours)
			for (const p of body.querySelectorAll(':scope > p, :scope > .noted > p')) p.classList.add(parseFloat(getComputedStyle(p).textIndent) > 0 ? 'pg-ti' : 'pg-ti0');
			// display maths wider than the main column spans the text block (and is reported if
			// it is wider still)
			for (const eq of body.querySelectorAll(':scope > .eq, :scope > p .eq-scroll, :scope > .noted > .eq')) {
				const r = document.createRange();
				r.selectNodeContents(eq);
				const w = r.getBoundingClientRect().width;
				if (w > mainW + 1) {
					const block = eq.closest('.eq') ?? eq;
					block.classList.add('pg-eq-wide');
					if (w > 153 * PX_PER_MM) diag.warnings.push(`an equation in ${body.closest('[data-chapter]')?.dataset.chapter} is ${(w / PX_PER_MM).toFixed(0)} mm wide, wider than the text block: “${plain(eq).slice(0, 60)}”`);
				}
			}
		}
		// a code excerpt whose longest line fits the main column sits in it (and moves with it on
		// versos); a longer one spans the text block
		// (a code excerpt with a margin note is unwrapped first: the note is already a mark that
		// anchors to the block after it, and the excerpt is then laid out like any other)
		for (const code of root.querySelectorAll('.chapter-body > .noted > .code')) code.parentElement.replaceWith(...code.parentElement.childNodes);
		for (const code of root.querySelectorAll('.chapter-body > .code')) {
			let w = 0;
			for (const pre of code.querySelectorAll('pre')) {
				const r = document.createRange();
				r.selectNodeContents(pre);
				w = Math.max(w, r.getBoundingClientRect().width);
			}
			if (w <= mainW + 0.5) { code.classList.add('pg-code-main'); continue; }
			// longer than the text block too: set the excerpt a little smaller rather than let a
			// source line run off the page (the lines are the file's own; they are never rewrapped)
			const avail = code.querySelector('pre')?.getBoundingClientRect().width ?? 0;
			if (avail && w > avail + 0.5) {
				const fs = parseFloat(getComputedStyle(code).getPropertyValue('--code-fs')) || 7.1;
				const to = Math.max(5.8, Math.floor(fs * (avail / w) * 20) / 20);
				code.style.setProperty('--code-fs', `${to}pt`);
				diag.floats.push({ id: code.dataset.file ?? 'code', how: `set at ${to} pt so its longest line fits the text block` });
				if (to === 5.8 && fs * (avail / w) < 5.8) diag.warnings.push(`a code excerpt (${code.querySelector('.code__head')?.textContent.trim().slice(0, 50)}) is too wide for the text block even at 5.8 pt`);
			}
		}
		// Hatching as vector lines. Chromium's PDF backend cannot write the rotated SVG hatch
		// pattern natively and paints each hatched mark with a page-sized 72 dpi raster tile.
		// Each hatch fill becomes the same lines (spacing, angle, ink and weight read from the
		// pattern) under the mark's own outline: cut to rectangles and circles exactly, clipped
		// for any other shape. The mark's opacity goes to stroke-opacity (native in PDF; a
		// translucent group would be rasterised again).
		const NS = 'http://www.w3.org/2000/svg';
		let hk = 0;
		const segs = (shape, el, n, d, gap) => {
			// lines n·p = c, c = k·gap, cut to the shape; returns [x1, y1, x2, y2][]
			const out = [];
			if (shape === 'rect') {
				const x0 = +el.getAttribute('x') || 0, y0 = +el.getAttribute('y') || 0;
				const x1 = x0 + (+el.getAttribute('width') || 0), y1 = y0 + (+el.getAttribute('height') || 0);
				const cs = [x0 * n[0] + y0 * n[1], x1 * n[0] + y0 * n[1], x0 * n[0] + y1 * n[1], x1 * n[0] + y1 * n[1]];
				for (let c = Math.ceil(Math.min(...cs) / gap) * gap; c <= Math.max(...cs); c += gap) {
					// the line through q = c·n, direction d; clip t to the rectangle (Liang–Barsky)
					const q = [c * n[0], c * n[1]];
					let t0 = -1e9, t1 = 1e9;
					for (const [qi, di, lo, hi] of [[q[0], d[0], x0, x1], [q[1], d[1], y0, y1]]) {
						if (Math.abs(di) < 1e-12) { if (qi < lo || qi > hi) { t0 = 1; t1 = 0; } continue; }
						const a = (lo - qi) / di, b = (hi - qi) / di;
						t0 = Math.max(t0, Math.min(a, b)); t1 = Math.min(t1, Math.max(a, b));
					}
					if (t1 > t0) out.push([q[0] + t0 * d[0], q[1] + t0 * d[1], q[0] + t1 * d[0], q[1] + t1 * d[1]]);
				}
			} else if (shape === 'circle') {
				const cx = +el.getAttribute('cx') || 0, cy = +el.getAttribute('cy') || 0, r = +el.getAttribute('r') || 0;
				const c0 = cx * n[0] + cy * n[1];
				for (let c = Math.ceil((c0 - r) / gap) * gap; c <= c0 + r; c += gap) {
					const off = c - c0, h = Math.sqrt(Math.max(0, r * r - off * off));
					const m = [cx + off * n[0], cy + off * n[1]];
					if (h > 0) out.push([m[0] - h * d[0], m[1] - h * d[1], m[0] + h * d[0], m[1] + h * d[1]]);
				}
			}
			return out;
		};
		const patOf = (v) => { const id = v.match(/url\("?[^")#]*#([^")]+)"?\)/)?.[1]; const p = id && document.getElementById(id); return p?.tagName === 'pattern' ? p : null; };
		// a hatched stroke (a band along a curve) becomes the same ink, dashed into short bars
		// across the band: still the 1-form's stack of planes, and vector
		for (const el of root.querySelectorAll('svg path, svg line, svg polyline, svg rect, svg circle, svg ellipse, svg polygon')) {
			if (el.closest('pattern, clipPath, mask, defs, .pg-defs')) continue;
			const pat = patOf(getComputedStyle(el).stroke);
			if (!pat) continue;
			const line = pat.querySelector('line');
			const gap = parseFloat(pat.getAttribute('width')) || 3.2;
			const w = line ? parseFloat(getComputedStyle(line).strokeWidth) || 1 : 1;
			el.style.stroke = line ? getComputedStyle(line).stroke : 'currentColor';
			el.style.strokeDasharray = `${w.toFixed(2)} ${(gap - w).toFixed(2)}`;
			el.style.strokeLinecap = 'butt';
		}
		for (const el of root.querySelectorAll('svg rect, svg path, svg circle, svg ellipse, svg polygon')) {
			if (el.closest('pattern, clipPath, mask, defs, .pg-defs')) continue;
			const cs = getComputedStyle(el);
			const id = cs.fill.match(/url\("?[^")#]*#([^")]+)"?\)/)?.[1];
			const pat = id && document.getElementById(id);
			if (!pat || pat.tagName !== 'pattern') continue;
			const line = pat.querySelector('line');
			const gap = parseFloat(pat.getAttribute('width')) || 3.2;
			const th = (+(pat.getAttribute('patternTransform')?.match(/rotate\(([-\d.]+)/)?.[1] ?? 0)) * Math.PI / 180;
			const n = [Math.cos(th), Math.sin(th)], d = [-Math.sin(th), Math.cos(th)];
			const g = document.createElementNS(NS, 'g');
			g.setAttribute('class', 'pg-hatch');
			if (el.getAttribute('transform')) g.setAttribute('transform', el.getAttribute('transform'));
			g.setAttribute('stroke', line ? getComputedStyle(line).stroke : cs.stroke);
			g.setAttribute('stroke-width', line ? getComputedStyle(line).strokeWidth : '1');
			g.setAttribute('stroke-opacity', String(parseFloat(cs.opacity) * parseFloat(cs.fillOpacity)));
			g.setAttribute('fill', 'none');
			const shape = el.tagName.toLowerCase();
			let lines;
			if (shape === 'rect' && !(+el.getAttribute('rx'))) lines = segs('rect', el, n, d, gap);
			else if (shape === 'circle') lines = segs('circle', el, n, d, gap);
			else {
				// any other shape: lines across its box, clipped to it
				const svg = el.ownerSVGElement;
				let defs = svg.querySelector(':scope > defs');
				if (!defs) { defs = document.createElementNS(NS, 'defs'); svg.prepend(defs); }
				const clip = document.createElementNS(NS, 'clipPath');
				clip.id = `pg-hatch-${++hk}`;
				const sh = el.cloneNode(false);
				for (const at of ['transform', 'class', 'style', 'id', 'fill', 'stroke']) sh.removeAttribute(at);
				clip.append(sh);
				defs.append(clip);
				g.setAttribute('clip-path', `url(#${clip.id})`);
				const b = el.getBBox();
				const box = document.createElementNS(NS, 'rect');
				box.setAttribute('x', b.x); box.setAttribute('y', b.y); box.setAttribute('width', b.width); box.setAttribute('height', b.height);
				lines = segs('rect', box, n, d, gap);
			}
			for (const [x1, y1, x2, y2] of lines) {
				const l = document.createElementNS(NS, 'line');
				l.setAttribute('x1', x1.toFixed(2)); l.setAttribute('y1', y1.toFixed(2)); l.setAttribute('x2', x2.toFixed(2)); l.setAttribute('y2', y2.toFixed(2));
				g.append(l);
			}
			el.before(g);
			el.style.fill = 'none';
		}
		// Blend modes and group opacity together are rasterised by Chromium's PDF backend (a
		// page-sized 120 dpi image). In a drawing that has any translucent mark, the blend modes
		// go: marks simply overlap. Elsewhere the overprint (multiply) stays, as vector.
		let blends = 0;
		for (const svg of root.querySelectorAll('svg')) {
			const els = [svg, ...svg.querySelectorAll('*')];
			const cs = els.map((e) => getComputedStyle(e));
			if (!cs.some((c) => parseFloat(c.opacity) < 1)) continue;
			cs.forEach((c, i) => { if (c.mixBlendMode !== 'normal') { els[i].style.mixBlendMode = 'normal'; blends++; } });
		}
		for (const el of root.querySelectorAll('.fx')) if (getComputedStyle(el).mixBlendMode !== 'normal' && el.querySelector('svg [style*="mix-blend-mode: normal"]')) el.style.mixBlendMode = 'normal';
		if (blends) diag.floats.push({ id: 'print', how: `${blends} blend mode(s) dropped in drawings with translucent marks (Chromium would rasterise them)` });
		// Code may break across pages, but only between slices of at least three lines, each
		// slice carrying its own line numbers (so numbers and code stay level). Highlighting spans
		// that cross a slice edge are closed and reopened.
		for (const chunk of root.querySelectorAll('.chapter-body .code .code__chunk')) {
			const numsEl = chunk.querySelector('.code__nums');
			const codeEl = chunk.querySelector('pre > code');
			if (!numsEl || !codeEl) continue;
			const nums = [...numsEl.children];
			let lines = codeEl.innerHTML.split('\n');
			if (lines.length === nums.length + 1 && !lines[lines.length - 1].trim()) lines = lines.slice(0, -1);
			if (lines.length !== nums.length || lines.length < 7) continue;
			const opens = [];
			let stack = [];
			for (const l of lines) {
				opens.push([...stack]);
				for (const m of l.matchAll(/<span\b[^>]*>|<\/span>/g)) m[0][1] === '/' ? stack.pop() : stack.push(m[0]);
			}
			opens.push(stack);
			const cuts = [];
			for (let a = 0; a + 6 <= lines.length; a += 3) cuts.push(a + 3);
			const bounds = [0, ...cuts, lines.length];
			const frag = document.createDocumentFragment();
			for (let i = 0; i + 1 < bounds.length; i++) {
				const [a, b] = [bounds[i], bounds[i + 1]];
				const c = chunk.cloneNode(false);
				c.classList.add('pg-slice');
				const ns = numsEl.cloneNode(false);
				for (const n of nums.slice(a, b)) ns.append(n.cloneNode(true));
				const pre = codeEl.parentElement.cloneNode(false);
				const code = codeEl.cloneNode(false);
				code.innerHTML = opens[a].join('') + lines.slice(a, b).join('\n') + '</span>'.repeat(opens[b].length);
				pre.append(code);
				c.append(ns, pre);
				frag.append(c);
			}
			chunk.replaceWith(frag);
		}
		// a figure taller than the text block is set smaller (its author is told)
		for (const fig of root.querySelectorAll('.chapter-body > .fig')) {
			const h = fig.getBoundingClientRect().height;
			if (h <= blockH - 4) continue;
			// a figure that fills its width scales with it; anything else (HTML panels, SVGs of fixed
			// size) needs zoom as well
			const body = fig.querySelector('.fig__body');
			const target = blockH - 3 * PX_PER_MM;
			const H = () => fig.getBoundingClientRect().height;
			const B = () => body.getBoundingClientRect().height;
			let k = 1;
			for (let i = 0; i < 12 && H() > target; i++) {
				const before = H();
				k *= Math.min(0.985, Math.max(0.5, (target - (before - B())) / B()));
				body.style.width = `${(k * 100).toFixed(2)}%`;
				if (i === 0 && H() >= before - 1) { body.style.width = ''; k = 1; break; } // does not scale with its width
			}
			// content of a fixed size: zoom
			for (let i = 0; i < 12 && H() > target; i++) {
				k *= Math.min(0.985, Math.max(0.5, (target - (H() - B())) / B()));
				body.style.zoom = String(k);
			}
			const fits = fig.getBoundingClientRect().height <= target;
			(fits ? diag.warnings : diag.errors).push(`${fig.dataset.fig}: ${(h / PX_PER_MM).toFixed(0)} mm tall, more than the ${(blockH / PX_PER_MM).toFixed(0)} mm text block; ${fits ? `printed at ${(k * 100).toFixed(0)} % (its labels too)` : 'it cannot be made to fit'}`);
		}
		// measured height of every block in a chapter's flow: does a deferred figure find text
		// enough to fill the hole it leaves?
		for (const body of root.querySelectorAll('.chapter-body')) {
			for (const el of body.children) {
				const r = el.getBoundingClientRect();
				const cs = getComputedStyle(el);
				el.dataset.pgH = Math.round(r.height + parseFloat(cs.marginTop) + parseFloat(cs.marginBottom));
			}
			// an empty marker before each figure lets the layout decide the figure's place
			// before the figure itself is laid out
			for (const fig of body.querySelectorAll(':scope > .fig')) {
				const a = document.createElement('div');
				a.className = 'pg-anchor';
				fig.before(a);
			}
		}
		// cross-references: number them; annotate those whose target is on another spread
		let k = 0;
		for (const a of root.querySelectorAll('.chapter-body a.xref[href^="#"]')) {
			const xi = String(k++);
			a.dataset.xi = xi;
			const p = xpages?.[xi];
			if (!p) continue;
			const before = a.previousSibling?.textContent ?? '';
			const span = document.createElement('span');
			span.className = 'pg-xpage';
			span.textContent = /\(\s*$/.test(before) ? `, p. ${p}` : ` (p. ${p})`;
			a.append(span);
		}
	}

	// ── 1. figure floats ──────────────────────────────────────────────────────────
	class PicoFloats extends window.Paged.Handler {
		constructor(chunker, polisher, caller) {
			super(chunker, polisher, caller);
			this.pending = [];   // figures taken out of the flow, waiting for a later page
			this.wrapper = null;
			this.layout = null;
			this.lineH = 13.6 * 96 / 72;
		}

		onPageLayout(wrapper, breakToken, layout) {
			if (wrapper !== this.wrapper) this.flushed = false;
			this.wrapper = wrapper;
			this.layout = layout;
		}

		rendered(sourceEl) {
			return this.wrapper && this.wrapper.querySelector(`[data-ref="${sourceEl.dataset.ref}"]`);
		}

		// text after the figure that could fill the hole it leaves: up to the chapter's exercises
		// (a figure drifts past them only rather than leave more than half a page empty)
		following(fig, pastExercises = false) {
			let h = 0;
			for (let n = fig.nextElementSibling; n; n = n.nextElementSibling) {
				if (!pastExercises && (n.classList.contains('exercise') || (n.tagName === 'H2' && /exercises/i.test(n.textContent) && n.nextElementSibling?.classList.contains('exercise')))) break;
				h += +(n.dataset.pgH || 0);
			}
			return h;
		}

		// A heading stays with what follows it: two lines of text, or the figure it introduces
		// (unless text follows that figure, in which case the figure may float on). If they do
		// not fit, the page breaks before the heading.
		keepWithNext(h) {
			// A deferred figure can move this heading to a roomier page. Recompute our
			// own keep decision there instead of retaining the previous page's break.
			if (h.dataset.pgKeepBreak) { delete h.dataset.breakBefore; delete h.dataset.pgKeepBreak; }
			const body = this.rendered(h.parentNode);
			if (!body || !body.firstChild) return;
			const b = this.layout.bounds;
			const beyond = (rect) => rect.left >= b.right - 1;
			let next = h.nextElementSibling;
			while (next && next.classList.contains('pg-anchor')) next = next.nextElementSibling;
			const probe = h.cloneNode(true);
			probe.removeAttribute('id');
			body.appendChild(probe);
			const hr = probe.getBoundingClientRect();
			let need = 0;
			let extra = null;
			if (next && next.classList.contains('fig')) {
				let after = next.nextElementSibling;
				while (after && after.classList.contains('pg-anchor')) after = after.nextElementSibling;
				if (!after || /^H[1-6]$/.test(after.tagName) || after.classList.contains('fig') || after.classList.contains('exercise')) extra = next.cloneNode(true);
				else need = 2.2 * this.lineH; // the figure can float on; the text after it stays with the heading
			} else if (next && next.classList.contains('code')) need = 4 * this.lineH;
			else if (next) need = 2.2 * this.lineH;
			let bottom = hr.bottom + need;
			if (extra) {
				extra.removeAttribute('id');
				body.appendChild(extra);
				const er = extra.getBoundingClientRect();
				bottom = beyond(er) ? Infinity : er.bottom;
				extra.remove();
			}
			probe.remove();
			if (beyond(hr)) return; // already overflowing: the break will fall earlier
			if (bottom > b.bottom + 0.5 && hr.top - b.top > 3 * this.lineH) { h.dataset.breakBefore = 'page'; h.dataset.pgKeepBreak = '1'; }
		}

		layoutNode(node) {
			// Paged.js does not keep nested two-column groups intact on its own. Decide
			// before laying one out, while a break can still fall before the whole group.
			if (node.nodeType === 1 && node.classList.contains('dict') && node.parentNode?.classList?.contains('chapter-body')) {
				if (node.dataset.pgDictBreak) { delete node.dataset.breakBefore; delete node.dataset.pgDictBreak; }
				const body = this.rendered(node.parentNode);
				if (!body) return;
				const probe = node.cloneNode(true);
				body.append(probe);
				const r = probe.getBoundingClientRect(), b = this.layout.bounds;
				probe.remove();
				if (r.left < b.right - 1 && r.bottom > b.bottom - 2 && r.top > b.top + 2) {
					node.dataset.breakBefore = 'page'; node.dataset.pgDictBreak = '1';
				}
				return;
			}
			if (node.nodeType === 1 && (node.tagName === 'H2' || node.tagName === 'H3') && node.parentNode?.classList?.contains('chapter-body')) return this.keepWithNext(node);
			if (node.nodeType !== 1 || !node.classList.contains('pg-anchor')) return;
			const fig = node.nextElementSibling;
			if (!fig || !fig.classList.contains('fig')) return;
			const parent = node.parentNode;
			const body = this.rendered(parent);
			if (!body || !body.firstChild) return; // the figure would open the page: nothing to decide
			const b = this.layout.bounds;
			// Paged.js lays a page out as the first of a row of CSS columns: whatever lands in a
			// later column (left ≥ the page's right edge) has overflowed the page.
			const beyond = (rect) => rect.left >= b.right - 1;
			// where the figure's slot begins: an empty probe that clears the margin notes, as figures do
			const slot = document.createElement('div');
			slot.style.cssText = 'height:0;margin:0;padding:0;border:0;clear:both';
			body.appendChild(slot);
			const s0 = slot.getBoundingClientRect();
			slot.remove();
			if (beyond(s0)) return; // the text before the figure already overflows: decide on the next page
			// the figure where it stands
			const probe = fig.cloneNode(true);
			probe.removeAttribute('id');
			body.appendChild(probe);
			const r = probe.getBoundingClientRect();
			probe.remove();
			const slotTop = s0.top;
			const above = slotTop - b.top;
			const fits = !beyond(r) && r.bottom <= b.bottom + 0.5;
			const id = fig.id || fig.dataset.fig || '(figure)';

			if (fits) {
				// only the tail of a paragraph above it: move the figure to the head of the page
				const bodyAtTop = body.getBoundingClientRect().top - b.top < 2;
				const afterHeading = /^H[1-6]$/.test(node.previousElementSibling?.tagName ?? '');
				if (above > 1 && above < SNAP_TOP_LINES * this.lineH && bodyAtTop && !afterHeading) {
					const c = fig.cloneNode(true);
					c.classList.add('pg-float', 'pg-top');
					body.insertBefore(c, body.firstChild);
					const rc = c.getBoundingClientRect();
					if (!beyond(rc) && rc.bottom <= b.bottom + 0.5) {
						fig.remove();
						diag.floats.push({ id, how: 'to the head of its page' });
					} else c.remove();
				}
				return;
			}
			if (r.height + 2 > b.height) {
				diag.warnings.push(`${id}: taller than the text block (${(r.height / PX_PER_MM).toFixed(0)} mm > ${(b.height / PX_PER_MM).toFixed(0)} mm); it will overflow`);
				return;
			}
			const hole = b.bottom - slotTop;
			if (hole < MIN_HOLE_LINES * this.lineH) return; // a small gap: let it go to the next page
			if (this.following(fig, hole > 0.5 * b.height) < hole) return; // too little text left to fill it
			fig.remove();
			this.pending.push({ fig, parent, id });
			diag.floats.push({ id, how: 'deferred to the head of the next page' });
		}

		// End of a page: the figures deferred on it go into the source exactly where the next
		// page begins (splitting the paragraph there if the page broke inside it), and the next
		// page's break token is pointed at them, so Paged.js lays them out as ordinary content.
		afterPageLayout(pageEl, page, breakToken) {
			if (!this.pending.length || !breakToken || !breakToken.node) return;
			const parent = this.pending[0].parent;
			const figs = [];
			while (this.pending.length && this.pending[0].parent === parent) figs.push(this.pending.shift().fig);
			const tok = breakToken.node;
			let at = null; // insert before this child of parent (null: at the end of the chapter)
			if (tok !== parent && parent.contains(tok)) {
				let top = tok;
				while (top.parentNode !== parent) top = top.parentNode;
				const before = document.createRange();
				before.setStart(top, 0);
				before.setEnd(tok, breakToken.offset || 0);
				if (!before.toString().trim() && !before.cloneContents().querySelector('img, svg, math')) {
					at = top;
					// Paged.js fingerprints every break token to detect loops; this one is about to be
					// replaced, so retire its fingerprint, or a later break at the same node looks like a loop
					const retire = tok.nodeType === 1 ? tok : tok.parentElement;
					if (retire && retire !== parent) retire.dataset.ref = newRef();
				}
				else {
					// split `top` at the break: the part already set stays, the rest continues after the figures
					const rest = document.createRange();
					rest.setStart(tok, breakToken.offset || 0);
					rest.setEndAfter(top.lastChild);
					const cont = top.cloneNode(false);
					cont.removeAttribute('id');
					cont.classList.add('pg-cont');
					cont.append(rest.extractContents());
					const refs = new Set([...top.querySelectorAll('[data-ref]')].map((e) => e.dataset.ref).concat(top.dataset.ref));
					for (const e of [cont, ...cont.querySelectorAll('[data-ref]')]) if (refs.has(e.dataset.ref)) e.dataset.ref = newRef();
					top.after(cont);
					at = cont;
				}
			}
			figs.forEach((f, i) => {
				f.classList.add('pg-top');
				if (i) {
					const a = document.createElement('div');
					a.className = 'pg-anchor';
					parent.insertBefore(a, at);
				}
				parent.insertBefore(f, at);
			});
			breakToken.node = figs[0];
			breakToken.offset = 0;
		}

		afterRendered() {
			for (const q of this.pending) diag.errors.push(`${q.id}: deferred figure never placed (lost at the end of its chapter)`);
			for (const h of CARRY) diag.errors.push(`a margin note found no page with room for it: “${h.replace(/<[^>]+>/g, '').slice(0, 70)}”`);
		}
	}

	// ── words are not divided across a page turn ────────────────────────────────────
	// Paged.js breaks a hyphenated word where Chromium hyphenated it, then appends its own
	// hyphen and the line can re-wrap out of the page. The break moves back to the word's start.
	class PicoWords extends window.Paged.Handler {
		onBreakToken(breakToken, overflow) {
			if (!breakToken || !overflow || !breakToken.node || breakToken.node.nodeType !== 3 || !breakToken.offset) return;
			if (overflow.startContainer.nodeType !== 3) return;
			const t = breakToken.node.textContent;
			const isW = (c) => !!c && /[\p{L}\p{N}\u00AD'’]/u.test(c);
			let o = breakToken.offset;
			if (!(isW(t[o - 1]) && isW(t[o]))) return;
			while (o > 0 && isW(t[o - 1])) o--;
			const delta = breakToken.offset - o;
			if (o === 0 || overflow.startOffset - delta < 0) return;
			overflow.setStart(overflow.startContainer, overflow.startOffset - delta);
			breakToken.offset = o;
		}
	}

	// ── 3. the first block of a page sits at the head of the text block ─────────────
	class PicoPageTop extends window.Paged.Handler {
		onPageLayout(wrapper) {
			if (wrapper !== this.w) { this.w = wrapper; this.done = false; }
		}
		renderNode(clone) {
			if (this.done || clone.nodeType !== 1 || !clone.parentElement) return;
			if (!clone.parentElement.classList.contains('chapter-body')) {
				if (clone.matches('.opening, .pp')) this.done = true; // a page that opens with these keeps them as they are
				return;
			}
			if (clone.classList.contains('pg-anchor')) return;
			this.done = true;
			let el = clone, first = true;
			while (el && el !== this.w) {
				let p = el.previousElementSibling;
				while (p && (p.classList.contains('pg-anchor') || p.classList.contains('rh-clear') || p.classList.contains('rh-chapter'))) p = p.previousElementSibling;
				if (p) { first = false; break; }
				el = el.parentElement;
			}
			if (first && !clone.dataset.splitFrom) clone.classList.add('pg-first');
		}
	}

	// ── margin notes: beside their text, on their page, never on top of anything ────
	// In the paged book a margin note is absolutely positioned at its anchor's height
	// (print.css), so it never drags its paragraph to the next page. Here, on each page, the
	// notes are stacked top to bottom with a small gap, moved off any full-width block
	// (wide figures, code, the names table) that crosses the notes column, and the whole
	// stack is pushed up from the foot if it runs past it.
	const CARRY = []; // notes that found no room on their page, for the next one
	function placeNotes(pageEl, content, verso) {
		const A = content.getBoundingClientRect();
		const n = pageEl.dataset.pageNumber;
		const flowing = !!content.querySelector('.chapter-body') && !content.querySelector('.opening:not([data-split-from]), .pp');
		const marks = [...content.querySelectorAll('.pg-note-mark')].filter((m) => m.getBoundingClientRect().left < A.right - 1);
		const carried = flowing ? CARRY.splice(0) : [];
		if (!marks.length && !carried.length) return;
		const host = content.querySelector('.print-edition-root') ?? content;
		const tmp = document.createElement('div');
		const make = (html, mark) => {
			tmp.innerHTML = html;
			const el = tmp.firstElementChild;
			el.classList.add('pg-note');
			host.appendChild(el);
			return { el, html, mark };
		};
		const colL = verso ? A.left : A.left + (104 + 5) * PX_PER_MM;
		const colR = colL + 44 * PX_PER_MM;
		const gap = 2.4 * PX_PER_MM;
		const near = 8 * 13.6 * 96 / 72;
		// obstacles: anything else in the notes column
		const obst = [...content.querySelectorAll('.fig, .code, .names, .dict, .eq.pg-eq-wide, .eq-scroll.pg-eq-wide, .exercise__label, .opening, .pg-toc, h2 .secnum, h3 .secnum')]
			.map((e) => e.classList.contains('fig') ? (e.querySelector('.fig__notes') && !e.classList.contains('fig--wide') && !e.classList.contains('fig--full') ? e.querySelector('.fig__notes') : e) : e)
			.map((e) => e.getBoundingClientRect())
			.filter((r) => r.height && r.right > colL + 1 && r.left < colR - 1 && r.left < A.right - 1)
			.sort((a, b) => a.top - b.top);
		const clash = (t, h) => obst.find((o) => t < o.bottom + gap && t + h > o.top - gap);
		// a block note sits level with the top of the paragraph it precedes; a sidenote level
		// with the line of its number; a carried note at the head of the page
		const wantOf = (m) => {
			if (!m) return A.top;
			const a = m.tagName === 'DIV' ? m.nextElementSibling ?? m : m.previousElementSibling ?? m;
			return Math.max(A.top, Math.min(A.bottom, a.getBoundingClientRect().top));
		};
		let items = [...carried.map((h) => make(h, null)), ...marks.map((m) => make(NOTES[+m.dataset.note], m))];
		for (const it of items) { it.h = it.el.getBoundingClientRect().height; it.want = wantOf(it.mark); }

		const place = () => {
			// downwards from each anchor, around obstacles
			let floor = A.top;
			for (const it of items) {
				let t = Math.max(it.want, floor);
				for (let o = clash(t, it.h); o; o = clash(t, it.h)) {
					const up = o.top - gap - it.h;
					if (up >= floor && it.want - up < near) { t = up; break; }
					t = o.bottom + gap;
				}
				it.top = t;
				floor = t + it.h + gap;
			}
			// the foot of the page pushes the stack up
			let ceil = A.bottom;
			for (let i = items.length - 1; i >= 0; i--) {
				const it = items[i];
				if (it.top + it.h <= ceil) { ceil = it.top - gap; continue; }
				let t = ceil - it.h;
				for (let o = clash(t, it.h); o; o = clash(t, it.h)) t = o.top - gap - it.h;
				it.top = t;
				ceil = t - gap;
			}
			return items.every((it, i) => it.top >= A.top - 0.5 && it.top + it.h <= A.bottom + 0.5 && !clash(it.top + 0.5, it.h - 1) && (!i || it.top >= items[i - 1].top + items[i - 1].h));
		};
		// what does not fit goes on to the next page (beside the rest of its paragraph, usually)
		while (items.length && !place()) {
			const last = items.pop();
			last.el.remove();
			CARRY.unshift(last.html);
		}
		if (CARRY.length) diag.floats.push({ id: `p. ${n}`, how: `${CARRY.length} margin note(s) carried to the next page` });
		if (carried.length && !items.length) diag.warnings.push(`p. ${n}: a margin note carried from the previous page found no room here either`);
		let moved = 0;
		for (const it of items) {
			const st = it.el.style;
			const cb = it.el.offsetParent?.getBoundingClientRect() ?? A;
			st.left = `${colL - cb.left}px`;
			st.top = `${it.top - cb.top}px`;
			if (Math.abs(it.top - it.want) > 2) moved++;
		}
		if (moved) diag.floats.push({ id: `p. ${n}`, how: `${moved} margin note(s) moved off their anchor line` });
	}

	// ── 2. page classes, running heads, folios ─────────────────────────────────────
	class PicoPages extends window.Paged.Handler {
		constructor(chunker, polisher, caller) {
			super(chunker, polisher, caller);
			this.chapter = null;   // { num, html }
			this.section = '';     // html of the section in force
		}
		afterPageLayout(pageEl) {
			const content = pageEl.querySelector('.pagedjs_page_content');
			const box = pageEl.querySelector('.pagedjs_pagebox');
			if (!content || !box) return;
			const n = +pageEl.dataset.pageNumber;
			const verso = pageEl.classList.contains('pagedjs_left_page');
			const opening = content.querySelector('.opening:not([data-split-from])');
			const plate = content.querySelector('.pp--frontis');
			const front = content.querySelector('.pp--halftitle, .pp--title, .pp--colophon, .pp--contents');
			const blank = pageEl.classList.contains('pagedjs_blank_page') || !content.textContent.trim() && !content.querySelector('img, svg');
			if (opening) {
				pageEl.classList.add('pg-opening');
				const name = opening.querySelector('.opening__name');
				this.chapter = { num: opening.querySelector('.opening__num')?.textContent.trim() ?? '', html: name ? richHTML(name) : '' };
				this.section = '';
			}
			placeNotes(pageEl, content, verso);
			if (plate) pageEl.classList.add('pg-frontis');
			if (front) pageEl.classList.add('pg-front');
			if (blank) pageEl.classList.add('pg-blank');
			const heads = [...content.querySelectorAll('.chapter-body h2:not([data-split-from])')];
			const sectionHere = heads.length ? richHTML(heads[0]) : this.section;
			if (heads.length) this.section = richHTML(heads[heads.length - 1]);
			if (blank || plate || front || !this.chapter) return;

			const side = verso ? 'verso' : 'recto';
			const folio = `<span class="pg-folio">${n}</span>`;
			if (opening) {
				const foot = document.createElement('div');
				foot.className = `pg-foot pg-foot--${side}`;
				foot.innerHTML = verso ? `${folio}<span></span>` : `<span></span>${folio}`;
				box.appendChild(foot);
				return;
			}
			const head = document.createElement('div');
			head.className = `pg-head pg-head--${side}`;
			const ch = this.chapter;
			head.innerHTML = verso
				? `${folio}<span class="pg-rh">${ch.num ? `<span class="pg-rh__num">${ch.num}</span>` : ''}${ch.html}</span>`
				: `<span class="pg-rh">${sectionHere || ch.html}</span>${folio}`;
			box.appendChild(head);
		}
	}

	// ── the audit ───────────────────────────────────────────────────────────────────
	function audit() {
		const pages = [...document.querySelectorAll('.pagedjs_page')];
		const fmtMM = (px) => `${(px / PX_PER_MM).toFixed(1)} mm`;
		// Ghosts: when a page cannot take a block at all, Paged.js may leave its clone in the
		// invisible overflow column while also starting the next page with it. A clone that
		// continues on the next page is removed; anything else out there is lost text (an error).
		pages.forEach((pg, i) => {
			const area = pg.querySelector('.pagedjs_page_content');
			const next = pages[i + 1];
			if (!area || !next) return;
			const A = area.getBoundingClientRect();
			for (const el of [...area.querySelectorAll('.chapter-body > *')]) {
				if (el.getBoundingClientRect().left < A.right - 1) continue;
				const ref = el.dataset.ref;
				if (ref && next.querySelector(`[data-ref="${ref}"], [data-split-from="${ref}"]`)) el.remove();
			}
		});
		for (const pg of pages) {
			const n = +pg.dataset.pageNumber;
			const area = pg.querySelector('.pagedjs_page_content');
			if (!area) continue;
			const A = area.getBoundingClientRect();
			const seen = new Set();
			const say = (msg) => { if (!seen.has(msg)) { seen.add(msg); diag.warnings.push(msg); } };
			const blocks = [...area.querySelectorAll('p, li, h1, h2, h3, h4, .eq, figure, table, pre, aside, .sidenote, svg, img, math[display="block"], .exercise__label')];
			for (const el of blocks) {
				if (el.closest('.opening__num, .secnum, .pg-toc__num, .opening__title, .pp-title')) continue;
				if (el.parentElement.closest('svg, math')) continue;
				const r = el.getBoundingClientRect();
				if (!r.width && !r.height) continue;
				const what = el.closest('[data-fig]')?.dataset.fig ?? el.closest('.chapter-body [id], .pp [id]')?.id ?? `“${el.textContent.replace(/\s+/g, ' ').trim().slice(0, 40)}”${el.className ? ` (.${String(el.className.baseVal ?? el.className).split(' ')[0]})` : ''}`;
				const tag = el.tagName.toLowerCase();
				if (r.left >= A.right - 1) { diag.errors.push(`p. ${n}: ${tag} ${what} fell outside the page (lost)`); continue; }
				if (r.right > A.right + 1.5) say(`p. ${n}: ${tag} in ${what} runs ${fmtMM(r.right - A.right)} into the right margin`);
				if (r.left < A.left - 1.5) say(`p. ${n}: ${tag} in ${what} runs ${fmtMM(A.left - r.left)} into the left margin`);
				if (r.bottom > A.bottom + 1.5) say(`p. ${n}: ${tag} in ${what} runs ${fmtMM(r.bottom - A.bottom)} below the text block`);
			}
			for (const f of area.querySelectorAll('figure.fig[data-split-from], figure.fig[data-split-to]')) say(`p. ${n}: figure ${f.dataset.fig} is split across pages`);
			// widows and orphans: a single line of a paragraph alone at the head or foot of a page
			const lh = 13.6 * 96 / 72;
			for (const p of area.querySelectorAll('.chapter-body p[data-split-from], .chapter-body p[data-split-to]')) {
				const r = p.getBoundingClientRect();
				if (r.left >= A.right - 1 || Math.round(r.height / lh) > 1) continue;
				say(`p. ${n}: ${p.dataset.splitTo ? 'orphan (one line of a paragraph at the foot)' : 'widow (one line of a paragraph at the head)'}: “${p.textContent.trim().slice(0, 50)}”`);
			}
			// margin notes that collide with each other or with full-width blocks
			const notes = [...area.querySelectorAll('.note, .sidenote, .exercise__label')].filter((x) => !x.closest('.fig__notes')).sort((a, b) => a.getBoundingClientRect().top - b.getBoundingClientRect().top);
			for (let i = 1; i < notes.length; i++) {
				const a = notes[i - 1].getBoundingClientRect(), b = notes[i].getBoundingClientRect();
				if (b.top < a.bottom - 1 && Math.abs(a.left - b.left) < 4) say(`p. ${n}: two margin notes overlap`);
			}
			const wide = [...area.querySelectorAll('.fig--wide, .fig--full, .code, .names, .dict, .fig__notes')];
			for (const note of notes) {
				const r = note.getBoundingClientRect();
				for (const w of wide) {
					const q = w.getBoundingClientRect();
					if (r.top < q.bottom - 1 && r.bottom > q.top + 1 && r.left < q.right - 1 && r.right > q.left + 1) say(`p. ${n}: a margin note overlaps ${w.closest('[data-fig]')?.dataset.fig ?? w.className.split(' ')[0]}`);
				}
			}
			// holes: text that stops well above the foot of a page that the chapter continues past
			const next = pages[pages.indexOf(pg) + 1];
			const forced = !next || next.classList.contains('pagedjs_blank_page') || next.querySelector('.opening:not([data-split-from]), .pp');
			if (!forced && !pg.classList.contains('pg-front') && !pg.classList.contains('pg-frontis') && !pg.classList.contains('pg-blank')) {
				let bottom = A.top;
				for (const el of area.querySelectorAll('.chapter-body > *, .opening')) {
					const r = el.getBoundingClientRect();
					if (r.height) bottom = Math.max(bottom, r.bottom);
				}
				const gap = (A.bottom - bottom) / A.height;
				if (gap > UNDERFULL) {
					const nextFirst = next.querySelector('.chapter-body > *:not(.pg-anchor)');
					const why = nextFirst?.matches('.fig') ? ` (figure ${nextFirst.dataset.fig} did not fit)` : nextFirst?.matches('.code') ? ' (a code excerpt did not fit)' : nextFirst?.matches('.exercise, .names, .dict, table') ? ` (a ${nextFirst.className.split(' ')[0]} block did not fit)` : '';
					say(`p. ${n}: page ${Math.round(gap * 100)} % empty at the foot${why}`);
				}
			}
		}
		const refs = new Set();
		for (const el of document.querySelectorAll('.pagedjs_pages svg [fill^="url(#"], .pagedjs_pages svg [stroke^="url(#"]')) for (const at of ['fill', 'stroke']) { const m = el.getAttribute(at)?.match(/^url\(#([^)]+)\)/); if (m) refs.add(m[1]); }
		const here = location.href.split('#')[0];
		for (const el of document.querySelectorAll('.pagedjs_pages svg *')) {
			for (const v of [getComputedStyle(el).fill, getComputedStyle(el).stroke]) {
				const m = v.match(/url\("?([^")#]*)#([^")]+)"?\)/);
				if (!m) continue;
				if (m[1] && m[1] !== here) diag.errors.push(`a figure paints with ${m[0]}, outside the document (nothing will print)`);
				refs.add(m[2]);
			}
		}
		for (const id of refs) if (!document.getElementById(id)) diag.errors.push(`a figure paints with url(#${id}), which is not defined in the book`);
		const ids = {};
		for (const el of document.querySelectorAll('.pagedjs_pages [id]')) ids[el.id] = (ids[el.id] ?? 0) + 1;
		for (const [id, k] of Object.entries(ids)) if (k > 1) diag.warnings.push(`the id “${id}” occurs ${k} times: links to it are ambiguous`);
		const missing = (sel, what, text) => document.querySelectorAll(`.pagedjs_pages ${sel}`).forEach((el) => diag.warnings.push(`p. ${pageNo(el)}: ${what}: ${text(el)}`));
		missing('.xref--missing', 'unresolved cross-reference', (el) => el.textContent.trim());
		missing('.fig__pending', 'figure in preparation (no component)', (el) => el.closest('[data-fig]')?.dataset.fig ?? '?');
		missing('.code--missing', 'code excerpt missing', (el) => `${el.dataset.file} ${el.dataset.sel}`);
		missing('.tex-error', 'maths error', (el) => el.textContent.trim().slice(0, 80));
		for (const img of document.querySelectorAll('.pagedjs_pages img')) if (!img.complete || !img.naturalWidth) diag.errors.push(`p. ${pageNo(img)}: image failed to load: ${img.getAttribute('src')}`);
		for (const f of document.fonts) if (f.status === 'error') diag.errors.push(`font failed to load: ${f.family} ${f.style} ${f.weight}`);
	}

	// ── the outline (PDF bookmarks) and the cross-reference map ──────────────────────
	const at = (el) => {
		const pg = el.closest('.pagedjs_page');
		return { page: +pg.dataset.pageNumber - 1, y: el.getBoundingClientRect().top - pg.getBoundingClientRect().top };
	};
	function outline() {
		const out = [];
		const title = document.querySelector('.pagedjs_pages .pp--title');
		if (title) out.push({ title: 'Title page', ...at(title) });
		const toc = document.querySelector('.pagedjs_pages .pp--contents');
		if (toc) out.push({ title: 'Contents', ...at(toc) });
		for (const op of document.querySelectorAll('.pagedjs_pages .opening:not([data-split-from])')) {
			const num = op.querySelector('.opening__num')?.textContent.trim();
			const name = plain(op.querySelector('.opening__name') ?? op);
			const item = { title: num ? `${num}  ${name}` : name, ...at(op), children: [] };
			const slug = op.closest('[data-chapter]')?.dataset.chapter;
			for (const h of document.querySelectorAll(`.pagedjs_pages [data-chapter="${slug}"] .chapter-body h2:not([data-split-from])`)) {
				const sn = h.querySelector('.secnum')?.textContent.trim();
				item.children.push({ title: `${sn ? sn + '  ' : ''}${h.dataset.rh ?? plain(h)}`, ...at(h) });
			}
			out.push(item);
		}
		return out;
	}
	function xrefs() {
		const out = {};
		for (const a of document.querySelectorAll('.pagedjs_pages a.xref[data-xi]')) {
			if (out[a.dataset.xi]) continue;
			const id = decodeURIComponent(a.getAttribute('href').slice(1));
			if (!/^(fig|sec|ex)-/.test(id)) continue;
			const t = document.getElementById(id) ?? document.querySelector(`.pagedjs_pages [data-id="${CSS.escape(id)}"]`);
			if (!t) continue;
			out[a.dataset.xi] = { page: pageNo(a), target: pageNo(t) };
		}
		return out;
	}

	window.__picoPaginate = async (opts = {}) => {
		const t0 = performance.now();
		try {
			const root = document.querySelector('.print-edition-root');
			if (!root) return { error: 'no .print-edition-root on the page' };
			// The page's own CSS, in cascade order. (Paged.js 0.4's CSS parser cannot read
			// media-query range syntax, which the minifier emits: `(width<=74rem)` becomes
			// `(max-width:74rem)`. Paged.js then drops every @media block but print, so
			// screen breakpoints never reach the book.)
			const ranges = (css) => css.replace(/\(\s*(width|height)\s*(<=|>=|<|>)\s*([^)]+?)\s*\)/g, (_, dim, op, v) => `(${op[0] === '<' ? 'max' : 'min'}-${dim}:${v})`);
			const sheets = [];
			for (const el of document.querySelectorAll('link[rel="stylesheet"], style')) {
				if (el.tagName === 'LINK') sheets.push({ [el.href]: ranges(await (await fetch(el.href)).text()) });
				else if (el.textContent.trim()) sheets.push({ [`${location.href}#style-${sheets.length}`]: ranges(el.textContent) });
			}
			// the pre-pass and the snapshot of the HTML happen in one task: a figure resized here
			// must not get the chance to lay itself out again (its ResizeObserver runs at the next frame)
			prepare(root, opts.xpages);
			const meta = { ...root.dataset };
			const html = root.outerHTML;
			// empty the page; Paged.js re-renders the book from its HTML into page boxes
			document.querySelectorAll('link[rel="stylesheet"], style, script').forEach((el) => el.remove());
			document.body.innerHTML = '';
			document.documentElement.classList.add('pg-paged');
			const holder = document.createElement('div');
			document.body.appendChild(holder);

			window.Paged.registerHandlers(PicoWords, PicoFloats, PicoPageTop, PicoPages);
			const previewer = new window.Paged.Previewer({ hyphenGlyph: '-' });
			const flow = await previewer.preview(html, sheets, holder);
			// Paged.js resolves every url() in the CSS against its stylesheet, which turns the
			// paint reference fill: url(#hatch-covector) into ….css#hatch-covector: nothing paints.
			// Fragment-only references are restored (paint does not change the layout).
			for (const st of document.querySelectorAll('style')) {
				const t = st.textContent;
				const u = t.replace(/url\((["']?)[^)"'#]*?\.css#([\w-]+)\1\)/g, 'url(#$2)');
				if (u !== t) st.textContent = u;
			}
			await document.fonts.ready;
			await Promise.all([...document.querySelectorAll('img')].map((img) => (img.complete ? null : new Promise((r) => { img.onload = img.onerror = r; }))));
			audit();
			return {
				pages: flow.total,
				ms: performance.now() - t0,
				trim: { w: +meta.trimW, h: +meta.trimH },
				meta: { title: meta.title, subtitle: meta.subtitle, author: meta.author },
				built: meta.built,
				outline: outline(),
				xrefs: xrefs(),
				diagnostics: diag
			};
		} catch (e) {
			return { error: `${e.message}\n${(e.stack || '').split('\n').slice(0, 6).join('\n')}` };
		}
	};
})();

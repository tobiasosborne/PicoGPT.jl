<script>
	// The contents as a map of the model. Each chapter is placed at the station of the
	// computation it explains, so reading down the page follows the forward pass:
	//   text → embed → attention ⊕ → MLP ⊕ → readout → loss
	// The rail on the left is the model drawn in the book's inks: the forward stem in value
	// ink, running down; the pullback in covector ink, rising beside it from the seed ℓ̄ = 1
	// to the embedding; the update closing the loop; and sampling, dotted, feeding its
	// output back to the text. Numerals keep the book's reading order. Chapters that exist
	// link; planned ones are grey and say "in preparation" (nothing links to absent text).
	// The rail of each row is its own SVG whose verticals run the row's full height, so
	// the lanes join from row to row whatever the text wraps to.
	import { resolve } from '$app/paths';
	let { chapters, hash = false } = $props();
	const href = (c) => (hash ? `#open-${c.slug}` : resolve(`/${c.slug}/`));

	// The model, station by station. lanes: which lanes pass through the row
	//   f: forward stem   c: covector stem   s: sampling loop
	// node: what sits on the stem in this row. Rows are listed in the order of the model.
	const MAP = {
		prologue: { station: 'the specimen', maths: '<i>θ</i> ∈ ℝ<sup><i>p</i></sup>', node: 'theta', lanes: '' },
		task: { station: 'text', maths: '<i>x</i> ∈ 𝒜<sup><i>T</i></sup>', node: 'text', lanes: 'fS' },
		vectors: { station: 'embed', maths: '<i>X</i> = <i>E</i>[:, <i>x</i>] + <i>P</i>', node: 'dot', lanes: 'fCs' },
		attention: { station: 'attention', maths: '<i>X</i> ← <i>X</i> + attn(LN <i>X</i>)', node: 'branch', lanes: 'fcs', block: 'top' },
		block: { station: 'MLP', maths: '<i>X</i> ← <i>X</i> + MLP(LN <i>X</i>)', node: 'branch', lanes: 'fcs', block: 'bottom' },
		energies: { station: 'readout', maths: '<i>p</i> = softmax(<i>W</i> LN <i>X</i> + <i>b</i>)', node: 'dot', lanes: 'fcsT' },
		surprise: { station: 'loss', maths: '<i>ℓ</i> = −⟨log <i>p<sub>y</sub></i>⟩', node: 'end', lanes: 'Fcs' },
		'pulling-back': { station: 'pullback', maths: '<i class="bar">x</i> = <i>Df</i>(<i>x</i>)<sup>T</sup> <i class="bar">y</i>', node: 'seed', lanes: 'Ks' },
		learning: { station: 'update', maths: '<i>θ</i> ← <i>θ</i> − <i>η</i> <i class="bar">θ</i>', node: 'loop', lanes: 's' },
		speaking: { station: 'sampling', maths: '<i>x</i><sub><i>T</i>+1</sub> ∼ <i>p</i>', node: 'sample', lanes: 'Z' },
		epilogue: { station: '', maths: '', node: '', lanes: '' }
	};
	const ORDER = ['prologue', 'task', 'vectors', 'attention', 'block', 'energies', 'surprise', 'pulling-back', 'learning', 'speaking', 'epilogue'];
	const rows = $derived([
		...ORDER.map((slug) => chapters.find((c) => c.slug === slug)).filter(Boolean),
		...chapters.filter((c) => !ORDER.includes(c.slug))
	]);
	const m = (c) => MAP[c.slug] ?? { station: '', maths: '', node: '', lanes: '' };
	// lane x positions in the rail (px)
	const XS = 13, XF = 44, XC = 50, NY = 27; // NY: node height, level with the title
	const has = (c, k) => m(c).lanes.includes(k);
</script>

<div class="map">
	<p class="map__note label">The chapters, placed where they act in the model. Read them in the order of their numbers.</p>
	<ol class="toc map__list">
		{#each rows as c (c.slug)}
			{@const s = m(c)}
			<li class="map__row" class:map__row--live={c.exists} class:map__row--block={s.block} value={c.unnumbered ? undefined : c.num}>
				<div class="map__rail" aria-hidden="true">
					<svg class="map__svg" width="96" height="100%">
						<!-- sampling: dotted, from 'speaking' up to the text (Z start, S end with arrow, s through) -->
						{#if has(c, 's')}<line class="map__samp" x1={XS} x2={XS} y1="0" y2="100%" />{/if}
						{#if has(c, 'Z')}<line class="map__samp" x1={XS} x2={XS} y1="0" y2={NY} />{/if}
						{#if has(c, 'S')}
							<line class="map__samp" x1={XS} x2={XS} y1={NY} y2="100%" />
							<path class="map__samp" d="M{XS},{NY} H{XF - 9}" />
							<path class="map__samphead" d="M{XF - 14},{NY - 3.5} L{XF - 9},{NY} L{XF - 14},{NY + 3.5}" />
						{/if}
						{#if has(c, 'T')}<path class="map__samp" d="M{XF},{NY} H{XS}" />{/if}
						<!-- forward stem: f through, F ends here at the loss -->
						{#if has(c, 'f')}<line class="map__fwd" x1={XF} x2={XF} y1={s.node === 'text' ? NY : 0} y2="100%" />{/if}
						{#if has(c, 'F')}<line class="map__fwd" x1={XF} x2={XF} y1="0" y2={NY} />{/if}
						<!-- covector stem: c through, C ends at the embedding, K starts at the seed -->
						{#if has(c, 'c')}<line class="map__cov" x1={XC} x2={XC} y1="0" y2="100%" /><line class="map__covbars" x1={XC} x2={XC} y1="0" y2="100%" />{/if}
						{#if has(c, 'C')}<line class="map__cov" x1={XC} x2={XC} y1={NY + 2} y2="100%" /><line class="map__covbars" x1={XC} x2={XC} y1={NY + 10} y2="100%" /><path class="map__covhead" d="M{XC - 3.5},{NY + 8} L{XC},{NY + 2} L{XC + 3.5},{NY + 8}" />{/if}
						{#if has(c, 'K')}<line class="map__cov" x1={XC} x2={XC} y1="0" y2={NY} /><line class="map__covbars" x1={XC} x2={XC} y1="0" y2={NY - 3} />{/if}
						<!-- the node -->
						{#if s.node === 'text'}
							<circle class="map__textnode" cx={XF} cy={NY} r="5" />
						{:else if s.node === 'dot'}
							<circle class="map__dot" cx={XF} cy={NY} r="3.4" />
						{:else if s.node === 'branch'}
							<!-- the branch leaves the stem, does its work, and merges back at ⊕ -->
							<path class="map__branch" d="M{XF},{NY - 14} C{XF + 12},{NY - 14} {XF + 30},{NY - 10} {XF + 30},{NY} S{XF + 12},{NY + 14} {XF + 5.5},{NY + 14}" />
							<rect class="map__box" x={XF + 25} y={NY - 5} width="10" height="10" />
							<circle class="map__plus" cx={XF} cy={NY + 14} r="5" />
							<path class="map__plusmark" d="M{XF - 3},{NY + 14} H{XF + 3} M{XF},{NY + 11} V{NY + 17}" />
						{:else if s.node === 'end'}
							<rect class="map__end" x={XF - 4} y={NY - 4} width="8" height="8" />
						{:else if s.node === 'seed'}
							<rect class="map__seed" x={XC - 5} y={NY - 1.2} width="10" height="2.4" />
						{:else if s.node === 'loop'}
							<path class="map__loopc" d="M{XF + 3},{NY - 9} A9,9 0 0 1 {XF + 3},{NY + 9}" />
							<path class="map__loopv" d="M{XF + 3},{NY + 9} A9,9 0 0 1 {XF + 3},{NY - 9}" />
							<path class="map__loophead" d="M{XF - 0.5},{NY - 12.5} L{XF + 3.6},{NY - 9} L{XF - 0.5},{NY - 5.5}" />
						{:else if s.node === 'sample'}
							<circle class="map__sample" cx={XS} cy={NY} r="4" />
						{/if}
					</svg>
					{#if s.block === 'top'}<span class="map__times">× <i>L</i></span>{/if}
				</div>
				<span class="toc__num map__num" aria-hidden="true">{c.unnumbered ? '' : c.num}</span>
				<div class="map__text">
					{#if c.exists}
						<a class="toc__title map__title" href={href(c)}>{c.title}</a>
					{:else}
						<span class="toc__title map__title">{c.title}</span>
					{/if}
					<span class="toc__gist map__gist">{c.gist}</span>
				</div>
				<div class="map__where">
					{#if s.station}<span class="map__station">{s.station}</span>{/if}
					{#if s.maths}<span class="map__maths">{@html s.maths}</span>{/if}
					<span class="map__state label">{c.exists ? '' : 'in preparation'}</span>
				</div>
				{#if hash && c.exists}<a class="toc__page" href={href(c)} aria-hidden="true"></a>{/if}
			</li>
		{/each}
	</ol>
	<p class="map__legend label" aria-hidden="true">
		<span><svg width="26" height="10"><line class="map__fwd" x1="2" x2="24" y1="5" y2="5" /></svg> forward, in value ink, down</span>
		<span><svg width="26" height="10"><line class="map__cov" x1="2" x2="24" y1="5" y2="5" /><line class="map__covbars" x1="2" x2="24" y1="5" y2="5" /></svg> pullback, in covector ink, up</span>
		<span><svg width="26" height="10"><line class="map__samp" x1="2" x2="24" y1="5" y2="5" /></svg> sampling, back to the text</span>
	</p>
</div>

<style>
	.map { width: 100%; }
	.map__note { color: var(--ink-3); margin: 0 0 0.9rem; }
	.map__list { list-style: none; margin: 0; padding: 0; }
	.map__row {
		display: grid;
		grid-template-columns: 96px 2.6rem minmax(0, 1fr) 15rem;
		column-gap: 1rem;
		align-items: stretch;
		border-top: var(--hair) solid var(--rule-2);
	}
	.map__row:last-child { border-bottom: var(--hair) solid var(--rule-2); }
	.map__row--block + .map__row--block { border-top-style: dotted; }
	.map__rail { position: relative; min-height: 3.6rem; }
	.map__svg { position: absolute; inset: 0; width: 96px; height: 100%; overflow: visible; }
	.map__times { position: absolute; left: 82px; top: 100%; transform: translateY(-50%); font-family: var(--sans); font-size: var(--fs-micro); color: var(--ink-3); white-space: nowrap; background: var(--paper); padding: 0 0.15em; }
	.map__times i { font-family: var(--serif); font-size: 1.15em; }
	.map__num {
		padding: 0.95rem 0 1rem;
		font-size: 2.1rem;
		line-height: 0.8;
		font-weight: 220;
		color: var(--ink-3);
		text-align: right;
		font-variant-numeric: lining-nums;
		font-variation-settings: 'opsz' 60;
	}
	.map__text { padding: 0.95rem 0 1.05rem; }
	.map__title { display: block; font-size: 1.42rem; line-height: 1.2; text-decoration: none; color: var(--ink-3); letter-spacing: -0.004em; }
	.map__row--live .map__title, .map__row--live .map__num { color: var(--ink); }
	a.map__title:hover { color: var(--value-deep); }
	.map__gist { display: block; margin-top: 0.22rem; font-size: var(--fs-small); line-height: 1.42; color: var(--ink-2); max-width: 36em; }
	.map__row:not(.map__row--live) .map__gist { color: var(--ink-3); }
	.map__where { padding: 1.05rem 0 1rem; display: flex; flex-direction: column; gap: 0.15rem; text-align: right; }
	.map__station { font-variant-caps: all-small-caps; letter-spacing: 0.08em; font-size: var(--fs-small); color: var(--ink-2); }
	.map__maths { font-family: var(--serif); font-size: 0.98rem; color: var(--ink); white-space: nowrap; }
	.map__maths :global(i) { font-style: italic; }
	.map__maths :global(i.bar) { text-decoration: overline; text-decoration-thickness: 0.06em; text-underline-offset: 0; }
	.map__maths :global(sup), .map__maths :global(sub) { font-size: 0.68em; line-height: 0; }
	.map__state { color: var(--ink-3); }
	.map__row:not(.map__row--live) .map__maths { color: var(--ink-3); }

	/* the rail: the model in the book's inks */
	.map__fwd { stroke: var(--value); stroke-width: 1.6px; }
	.map__cov { stroke: var(--covector); stroke-width: 0.9px; }
	.map__covbars { stroke: var(--covector); stroke-width: 5px; stroke-dasharray: 0.9 5.1; }
	.map__samp { stroke: var(--ink-3); stroke-width: 0.9px; stroke-dasharray: 0.6 3; stroke-linecap: round; fill: none; }
	.map__samphead { fill: none; stroke: var(--ink-3); stroke-width: 0.9px; }
	.map__covhead { fill: none; stroke: var(--covector); stroke-width: 1px; }
	.map__textnode { fill: var(--paper); stroke: var(--ink); stroke-width: 1.2px; }
	.map__dot { fill: var(--value); }
	.map__branch { fill: none; stroke: var(--value); stroke-width: 1.2px; }
	.map__box { fill: var(--paper); stroke: var(--value); stroke-width: 1.1px; }
	.map__plus { fill: var(--paper); stroke: var(--value); stroke-width: 1.2px; }
	.map__plusmark { stroke: var(--value); stroke-width: 1.1px; }
	.map__end { fill: var(--value); }
	.map__seed { fill: var(--covector); }
		.map__loopc { fill: none; stroke: var(--covector); stroke-width: 1.2px; }
	.map__loopv { fill: none; stroke: var(--value); stroke-width: 1.2px; }
	.map__loophead { fill: none; stroke: var(--covector); stroke-width: 1.1px; }
	.map__sample { fill: var(--paper); stroke: var(--ink-2); stroke-width: 1px; stroke-dasharray: 1 1.6; }
	.map__svg { mix-blend-mode: var(--overprint); }

	.map__legend { display: flex; flex-wrap: wrap; gap: 0.4rem 1.6rem; margin: 0.9rem 0 0; color: var(--ink-3); }
	.map__legend span { display: inline-flex; align-items: center; gap: 0.4rem; }

	@media (max-width: 52rem) {
		.map__row { grid-template-columns: 64px 2rem minmax(0, 1fr); column-gap: 0.7rem; }
		.map__where { grid-column: 3; padding: 0 0 0.9rem; text-align: left; flex-direction: row; flex-wrap: wrap; gap: 0.2rem 0.8rem; align-items: baseline; }
		.map__text { padding-bottom: 0.35rem; }
		.map__rail { grid-row: span 2; }
		.map__num { grid-row: span 2; font-size: 1.6rem; }
		.map__svg { width: 64px; }
		.map__times { left: 58px; }
	}
</style>

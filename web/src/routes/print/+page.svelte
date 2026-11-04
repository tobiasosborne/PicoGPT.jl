<script>
	// The print edition: the same HTML as the web book, every figure in its static
	// state, in reading order, under print.css. print/build.mjs waits for
	// <html data-print-ready="1">, paginates with Paged.js, prints. See DESIGN.md §8.
	import { onMount, setContext } from 'svelte';
	import { resolve } from '$app/paths';
	import { BOOK } from '#lib/outline.js';
	import { makeSpecimen } from '#lib/model/specimen.js';
	import { fmt } from '#figures/primitives/scales.js';
	import ChapterOpening from '#lib/shell/ChapterOpening.svelte';
	import ChapterBody from '#lib/shell/ChapterBody.svelte';
	import ThetaField from '#lib/shell/ThetaField.svelte';
	import Contents from '#lib/shell/Contents.svelte';

	let { data } = $props();
	const specimen = $derived(makeSpecimen(data.specimen));
	setContext('specimen', () => specimen);
	setContext('artifacts', () => data.artifacts ?? {});
	const cfg = $derived(specimen.config);
	const loss0 = $derived(specimen.loss?.[0]);
	const lossN = $derived(specimen.loss?.at(-1));

	onMount(async () => {
		document.documentElement.classList.add('print-edition');
		await document.fonts.ready;
		await Promise.all([...document.images].map((img) => (img.complete ? null : new Promise((r) => { img.onload = img.onerror = r; }))));
		// two frames: let measured figures settle at their print width
		await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
		document.documentElement.dataset.printReady = '1';
		window.dispatchEvent(new Event('pico:print-ready'));
	});
</script>

<svelte:head>
	<title>{BOOK.title} — print edition</title>
	<!-- print.css applies on screen here too, so this page is a faithful proof of the book -->
	<meta name="pico-edition" content="print" />
</svelte:head>

{#snippet plate(trace, label)}
	<div class="pp-plate"><img src={resolve(`/${trace.src}`)} alt={trace.title ?? ''} /></div>
	{#if trace.title || trace.desc}
		<!-- the trace's own <title> and <desc>: what was computed, from what (DESIGN.md §9) -->
		<p class="pp-plate__caption"><span class="pp-plate__label">{label}</span> <span class="pp-plate__title">{@html trace.title ?? ''}</span> {@html trace.desc ?? ''}</p>
	{/if}
{/snippet}

<div class="print-edition-root" data-title={BOOK.title} data-subtitle={BOOK.subtitle} data-author={BOOK.author} data-built={data.built} data-trim-w="189" data-trim-h="246">
	<!-- i. half-title (recto) -->
	<section class="pp pp--halftitle"><p class="pp-halftitle">{BOOK.title}</p></section>

	<!-- ii. frontispiece (verso), facing the title page -->
	<section class="pp pp--frontis pp--cover-frontis">
		{#if data.coverTrace}{@render plate(data.coverTrace, 'Frontispiece.')}{/if}
	</section>

	<!-- iii. title page (recto) -->
	<section class="pp pp--title">
		<header class="pp-titlehead">
			<h1 class="pp-title">{BOOK.title}</h1>
			<p class="pp-subtitle">{BOOK.subtitle}</p>
		</header>
		<ThetaField {specimen} />
		<p class="pp-author">{BOOK.author}</p>
	</section>

	<!-- iv. colophon (verso) -->
	<section class="pp pp--colophon">
		<dl class="pp-facts">
			<div><dt>The specimen</dt><dd>a transformer with {specimen.paramCount} parameters: <i>V</i> = {cfg.V}, <i>D</i> = {cfg.D}, <i>H</i> = {cfg.H}, <i>L</i> = {cfg.L}, <i>T</i> = {cfg.T}</dd></div>
			<div><dt>trained on</dt><dd>“{specimen.corpus}”</dd></div>
			{#if Number.isFinite(loss0)}<div><dt>its loss</dt><dd>{fmt(loss0, 3)} → {fmt(lossN, 3)} nats in {specimen.provenance.steps ?? specimen.loss.length} steps{#if specimen.provenance.floor != null}; the text’s own ambiguity allows no less than {fmt(specimen.provenance.floor, 3)}{/if}</dd></div>{/if}
			{#if specimen.provenance.greedy}<div><dt>asked to continue “to be or”</dt><dd>“{specimen.provenance.greedy}”</dd></div>{/if}
		</dl>
		<p>Every number in every figure comes from the specimen, {specimen.provenance.kind === 'artifact' ? 'as exported by the Julia code to artifacts/specimen.json' : 'trained by the JavaScript engine (provisional: the Julia export was not present when this proof was made)'}. The code is at {BOOK.repo.replace('https://', '')}.</p>
		<p>Two spot inks carry one meaning each throughout the book: <span class="ink-value">value ink</span> for quantities computed going forward, parameters and activations; <span class="ink-covector">covector ink</span> for adjoints x̄ pulled back from the loss. Printed in one colour, value marks are solid and covector marks are barred or hatched.</p>
		<p>Set in Literata, with mathematics in XCharter Math, labels in Archivo and code in JuliaMono, all under the SIL Open Font License. Typeset from the same HTML as the web edition, by Paged.js in Chromium.</p>
		<p class="pp-built">Proof of {data.built}.</p>
	</section>

	<!-- v. contents (recto) -->
	<section class="pp pp--contents">
		<h2 class="pp-h">Contents</h2>
		<ol class="pg-toc">
			{#each data.contents as c (c.num)}
				<li class="pg-toc__ch" class:pg-toc__ch--planned={!c.exists}>
					<span class="pg-toc__num">{c.unnumbered ? '' : c.num}</span>
					<p class="pg-toc__head">
						{#if c.exists}
							<a class="pg-toc__title" href="#open-{c.slug}">{c.title}</a><a class="pg-toc__page" href="#open-{c.slug}" aria-hidden="true"></a>
						{:else}
							<span class="pg-toc__title">{c.title}</span><span class="pg-toc__state">in preparation</span>
						{/if}
					</p>
					<p class="pg-toc__gist">{c.gist}</p>
					{#if c.sections.length}
						<p class="pg-toc__secs">
							{#each c.sections as s, k (s.id)}<span class="pg-toc__sec"><a href="#{s.id}">{@html s.html}</a>&nbsp;<a class="pg-toc__spage" href="#{s.id}" aria-hidden="true"></a></span>{#if k < c.sections.length - 1}<span class="pg-toc__sep" aria-hidden="true"></span>{/if}{/each}
						</p>
					{/if}
				</li>
			{/each}
		</ol>
	</section>

	<section class="pp pp--map">
		<h2 class="pp-h">A map of the model</h2>
		<Contents chapters={data.contents} hash />
	</section>

	{#each data.chapters as ch (ch.meta.slug)}
		<article class="chapter print-chapter" class:print-chapter--frontis={ch.trace} id="ch-{ch.meta.slug}" data-chapter={ch.meta.slug}>
			{#if ch.trace}
				<section class="pp pp--frontis pp--chapter-frontis">{@render plate(ch.trace, ch.meta.num !== '' ? `Chapter ${ch.meta.num}.` : 'Frontispiece.')}</section>
			{/if}
			<ChapterOpening meta={ch.meta} frontis={false} />
			<ChapterBody chapter={ch} figures={data.figures} mode="print" />
		</article>
	{/each}
</div>

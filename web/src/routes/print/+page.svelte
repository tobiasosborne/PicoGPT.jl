<script>
	// The print edition: the same HTML as the web book, every figure in its static
	// state, in reading order, under print.css. A PDF tool (Paged.js in headless
	// Chromium) waits for <html data-print-ready="1">, paginates, prints. See DESIGN.md §8.
	import { onMount, setContext } from 'svelte';
	import { resolve } from '$app/paths';
	import { BOOK } from '#lib/outline.js';
	import { makeSpecimen } from '#lib/model/specimen.js';
	import ChapterOpening from '#lib/shell/ChapterOpening.svelte';
	import ChapterBody from '#lib/shell/ChapterBody.svelte';
	import ThetaField from '#lib/shell/ThetaField.svelte';
	import Contents from '#lib/shell/Contents.svelte';

	let { data } = $props();
	const specimen = $derived(makeSpecimen(data.specimen));
	setContext('specimen', () => specimen);

	onMount(async () => {
		document.documentElement.classList.add('print-edition');
		await document.fonts.ready;
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

<div class="print-edition-root">
	<!-- i. half-title (recto) -->
	<section class="pp pp--halftitle"><p class="pp-halftitle">{BOOK.title}</p></section>

	<!-- ii. frontispiece (verso), facing the title page -->
	<section class="pp pp--frontis">
		{#if data.coverTrace}<img src={resolve(`/${data.coverTrace.src}`)} alt="" />{/if}
	</section>

	<!-- iii. title page (recto) -->
	<section class="pp pp--title">
		<h1 class="pp-title">{BOOK.title}</h1>
		<p class="pp-subtitle">{BOOK.subtitle}</p>
		<ThetaField {specimen} />
		<p class="pp-author">{BOOK.author}</p>
	</section>

	<!-- iv. colophon (verso) -->
	<section class="pp pp--colophon">
		<p>Set in Literata, with mathematics in XCharter Math, labels in Archivo and code in JuliaMono, all under the SIL Open Font License.</p>
		<p>Two spot inks carry one meaning each throughout the book: <span class="ink-value">value ink</span> for quantities computed going forward, parameters and activations; <span class="ink-covector">covector ink</span> for adjoints x̄ pulled back from the loss. Printed in one colour, value marks are solid and covector marks are barred or hatched.</p>
		<p>Every number in every figure comes from the specimen, the {specimen.paramCount}-parameter model of chapter 0, {specimen.provenance.kind === 'artifact' ? 'as exported by the Julia code to artifacts/specimen.json' : 'trained by the JavaScript engine (provisional: the Julia export was not present when this proof was made)'}.</p>
		<p class="pp-built">Proof of {data.built}.</p>
	</section>

	<!-- v. contents -->
	<section class="pp pp--contents">
		<h2 class="pp-h">Contents</h2>
		<Contents chapters={data.contents} hash />
	</section>

	{#each data.chapters as ch (ch.meta.slug)}
		<article class="chapter print-chapter" class:print-chapter--frontis={ch.trace} id="ch-{ch.meta.slug}" data-chapter={ch.meta.slug}>
			<section class="pp pp--frontis pp--chapter-frontis" class:pp--blank={!ch.trace}>
				{#if ch.trace}<img src={resolve(`/${ch.trace.src}`)} alt="" />{/if}
			</section>
			<ChapterOpening meta={ch.meta} frontis={false} />
			<ChapterBody chapter={ch} figures={data.figures} mode="print" />
		</article>
	{/each}
</div>

<script>
	import { resolve } from '$app/paths';
	import { BOOK } from '#lib/outline.js';
	import { makeSpecimen } from '#lib/model/specimen.js';
	import { fmt } from '#figures/primitives/scales.js';
	import ThetaField from '#lib/shell/ThetaField.svelte';
	import Contents from '#lib/shell/Contents.svelte';
	import InkToggle from '#lib/shell/InkToggle.svelte';

	let { data } = $props();
	const specimen = $derived(makeSpecimen(data.specimen));
	const first = $derived(data.chapters.find((c) => c.exists));
	const loss0 = $derived(specimen.loss[0]);
	const lossN = $derived(specimen.loss.at(-1));
	const cfg = $derived(specimen.config);
</script>

<svelte:head>
	<title>{BOOK.title} — {BOOK.subtitle}</title>
	<meta name="description" content="{BOOK.subtitle}. A from-scratch GPT in Julia, explained for physicists and mathematicians, with every figure drawn from a real model." />
</svelte:head>

<div class="cover">
	<div class="cover__tools label"><a href={resolve('/contents/')}>Contents</a><InkToggle /></div>
	<figure class="cover__verso" aria-label="Frontispiece">
		{#if data.trace}
			<img src={resolve(`/${data.trace.src}`)} alt={data.trace.title ?? 'Every parameter of the specimen over the course of its training'} />
		{/if}
	</figure>
	<main class="cover__recto">
		<header class="cover__head">
			<h1 class="cover__title">{BOOK.title}</h1>
			<p class="cover__subtitle">{BOOK.subtitle}</p>
		</header>
		<ThetaField {specimen} />
		<dl class="cover__facts label">
			<div><dt>trained on</dt><dd>“{specimen.corpus}”</dd></div>
			<div><dt>shape</dt><dd><i>V</i> = {cfg.V}, <i>D</i> = {cfg.D}, <i>H</i> = {cfg.H}, <i>L</i> = {cfg.L}, <i>T</i> = {cfg.T}</dd></div>
			{#if Number.isFinite(loss0)}<div><dt>loss</dt><dd>{fmt(loss0, 3)} → {fmt(lossN, 3)} nats in {specimen.provenance.steps ?? specimen.loss.length} steps{#if specimen.provenance.floor != null}; the text’s own ambiguity allows no less than {fmt(specimen.provenance.floor, 3)}{/if}</dd></div>{/if}
			{#if specimen.provenance.greedy}<div><dt>asked to continue “to be or”</dt><dd>“{specimen.provenance.greedy}”</dd></div>{/if}
			<div><dt>numbers</dt><dd>{specimen.provenance.kind === 'artifact' ? 'every one from artifacts/specimen.json' : 'provisional: trained by the JS engine until the Julia export exists'}</dd></div>
		</dl>
		<footer class="cover__foot">
			<p class="cover__author">{BOOK.author}</p>
			{#if first}<a class="cover__begin" href={resolve(`/${first.slug}/`)}>Begin with chapter {first.num}: {first.title} →</a>{/if}
		</footer>
	</main>
</div>

<section class="book contents-section" id="contents" aria-labelledby="contents-h">
	<h2 id="contents-h" class="contents-section__h">Contents</h2>
	<Contents chapters={data.chapters} />
</section>

<script>
	import { setContext } from 'svelte';
	import { makeSpecimen } from '#lib/model/specimen.js';
	import { BOOK } from '#lib/outline.js';
	import RunningHead from '#lib/shell/RunningHead.svelte';
	import ChapterOpening from '#lib/shell/ChapterOpening.svelte';
	import ChapterBody from '#lib/shell/ChapterBody.svelte';
	import ChapterNav from '#lib/shell/ChapterNav.svelte';

	let { data } = $props();
	const specimen = $derived(makeSpecimen(data.specimen));
	setContext('specimen', () => specimen);
	setContext('artifacts', () => data.artifacts ?? {});
	const meta = $derived(data.chapter.meta);
</script>

<svelte:head>
	<title>{meta.num ? `${meta.num} · ` : ''}{meta.title} — {BOOK.title}</title>
	<meta name="description" content={meta.claim?.replace(/<[^>]+>/g, '') ?? ''} />
</svelte:head>

<RunningHead {meta} nav={data.nav} />
<main>
	<article class="chapter" id={meta.id} data-chapter={meta.slug}>
		<ChapterOpening {meta} trace={data.trace} />
		<ChapterBody chapter={data.chapter} figures={data.figures} mode="interactive" />
	</article>
	<ChapterNav nav={data.nav} />
</main>

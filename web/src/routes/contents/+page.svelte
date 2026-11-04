<script>
	import { BOOK } from '#lib/outline.js';
	import RunningHead from '#lib/shell/RunningHead.svelte';
	import Contents from '#lib/shell/Contents.svelte';
	let { data } = $props();
</script>

<svelte:head><title>Contents — {BOOK.title}</title></svelte:head>
<RunningHead meta={{ num: '', title: 'Contents' }} />
<main class="book contents-section contents-section--page">
	<h1 class="contents-section__h">Contents</h1>
	<Contents chapters={data.chapters} />
</main>

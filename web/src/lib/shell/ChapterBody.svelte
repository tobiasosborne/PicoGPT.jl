<script>
	// The prose of one chapter: HTML segments from pandoc, with each figure slot
	// mounted as its component. mode: 'interactive' (web) | 'print' (static).
	import Figure from './Figure.svelte';

	let { chapter, figures = {}, mode = 'interactive' } = $props();
</script>

<div class="book chapter-body flow">
	{#each chapter.segments as seg, k (k)}
		{#if seg.t === 'html'}
			{@html seg.html}
		{:else}
			<Figure {...seg} entry={figures[seg.id]} {mode} chapter={chapter.meta} />
		{/if}
	{/each}
</div>

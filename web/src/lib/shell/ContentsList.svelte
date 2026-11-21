<script>
	// The contents as a plain list (numeral, title, gist), for an edition whose chapters
	// are not stations of the model (Contents.svelte draws that map for the main book).
	import { resolve } from '$app/paths';
	let { chapters, hash = false } = $props();
	const href = (c) => (hash ? `#open-${c.slug}` : resolve(`/${c.slug}/`));
</script>

<ol class="toc">
	{#each chapters as c (c.slug)}
		<li class="toc__item" class:toc__item--live={c.exists} value={c.unnumbered ? undefined : c.num}>
			<span class="toc__num" aria-hidden="true">{c.unnumbered ? '' : c.num}</span>
			<div>
				{#if c.exists}
					<a class="toc__title" href={href(c)}>{c.title}</a>
				{:else}
					<span class="toc__title">{c.title}</span>
				{/if}
				<span class="toc__gist">{c.gist}</span>
			</div>
			<span class="toc__state label">{c.exists ? '' : 'in preparation'}</span>
		</li>
	{/each}
</ol>

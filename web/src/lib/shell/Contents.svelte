<script>
	// The contents: ten chapters and an epilogue. Chapters that exist are links;
	// planned ones say so (no link to text that is not there yet).
	import { resolve } from '$app/paths';
	let { chapters, hash = false } = $props();
	const href = (c) => (hash ? `#open-${c.slug}` : resolve(`/${c.slug}/`));
</script>

<ol class="toc">
	{#each chapters as c (c.num)}
		<li class="toc__item" class:toc__item--live={c.exists}>
			<span class="toc__num" aria-hidden="true">{c.unnumbered ? '' : c.num}</span>
			<div class="toc__text">
				{#if c.exists}
					<a class="toc__title" href={href(c)}>{c.title}</a>
				{:else}
					<span class="toc__title">{c.title}</span>
				{/if}
				<span class="toc__gist">{c.gist}</span>
			</div>
			<span class="toc__state label">{c.exists ? 'read' : 'in preparation'}</span>
			{#if hash && c.exists}<a class="toc__page" href={href(c)} aria-hidden="true"></a>{/if}
		</li>
	{/each}
</ol>

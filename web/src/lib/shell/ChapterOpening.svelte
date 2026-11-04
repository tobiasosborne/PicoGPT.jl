<script>
	import Frontispiece from './Frontispiece.svelte';
	let { meta, trace = null, frontis = true } = $props();
	const numberWords = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten'];
</script>

<header class="opening" id="open-{meta.slug}">
	{#if frontis}<Frontispiece {trace} slug={meta.slug} />{/if}
	<div class="opening__text book">
		<!-- running-head marker for the print edition (string-set), invisible on screen -->
		<span class="rh-clear" aria-hidden="true"></span>
		<span class="rh-chapter" aria-hidden="true">{meta.num ? `${meta.num}\u2003` : ''}{meta.title}</span>
		{#if meta.num !== ''}
			<p class="opening__kicker"><span class="opening__word">Chapter {numberWords[+meta.num] ?? meta.num}</span></p>
		{/if}
		<h1 class="opening__title">
			{#if meta.num !== ''}<span class="opening__num" aria-hidden="true">{meta.num}</span>{/if}
			<span class="opening__name">{@html meta.titleHtml ?? meta.title}</span>
		</h1>
		{#if meta.claim}
			<p class="claim opening__claim">{@html meta.claim}</p>
		{/if}
	</div>
</header>

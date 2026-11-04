<script>
	// The web's running head: what a printed page carries at its top edge
	// (book, chapter, current section), plus the way out (contents) and the ink choice.
	import { onMount } from 'svelte';
	import { resolve } from '$app/paths';
	import { BOOK } from '#lib/outline.js';
	import InkToggle from './InkToggle.svelte';

	let { meta, nav = null } = $props();
	let section = $state('');
	let scrolled = $state(false);

	onMount(() => {
		const heads = [...document.querySelectorAll('.chapter-body h2')];
		const onScroll = () => {
			scrolled = window.scrollY > 120;
			let current = '';
			for (const h of heads) if (h.getBoundingClientRect().top < 110) current = h.textContent.trim();
			section = current;
		};
		onScroll();
		window.addEventListener('scroll', onScroll, { passive: true });
		return () => window.removeEventListener('scroll', onScroll);
	});
</script>

<header class="rhead" class:rhead--scrolled={scrolled}>
	<a class="rhead__book" href={resolve('/')}>{BOOK.title}</a>
	<span class="rhead__where label">
		{#if meta}<span class="rhead__chapter">{meta.num ? `${meta.num} ` : ''}{meta.title}</span>{/if}
		{#if section}<span class="rhead__section">{section}</span>{/if}
	</span>
	<span class="rhead__tools label">
		{#if nav?.prev}<a href={resolve(`/${nav.prev.slug}/`)} aria-label="Previous chapter" title="Previous chapter">←</a>{/if}
		<a href={resolve('/contents/')}>Contents</a>
		{#if nav?.next}<a href={resolve(`/${nav.next.slug}/`)} aria-label="Next chapter" title="Next chapter">→</a>{/if}
		<InkToggle />
	</span>
</header>

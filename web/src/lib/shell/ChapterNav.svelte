<script>
	import { resolve } from '$app/paths';
	let { nav } = $props();
</script>

<nav class="chapnav book" aria-label="Chapters">
	<div class="chapnav__prev">
		{#if nav.prev}
			<a href={resolve(`/${nav.prev.slug}/`)} rel="prev"><span class="chapnav__dir label">Previous</span><span class="chapnav__title"><span class="chapnav__num">{nav.prev.num}</span> {nav.prev.title}</span></a>
		{/if}
	</div>
	<div class="chapnav__toc"><a href={resolve('/contents/')} class="label">Contents</a></div>
	<div class="chapnav__next">
		{#if nav.next}
			<a href={resolve(`/${nav.next.slug}/`)} rel="next"><span class="chapnav__dir label">Next</span><span class="chapnav__title"><span class="chapnav__num">{nav.next.num}</span> {nav.next.title}</span></a>
		{/if}
	</div>
</nav>

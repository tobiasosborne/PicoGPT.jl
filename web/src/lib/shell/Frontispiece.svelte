<script>
	// The frontispiece slot (BRIEF §11): a plotter-style trace of a real computation,
	// public/traces/<slug>.svg. Absent file → nothing on the web (a quiet outline in
	// dev, so authors see where it goes), a blank verso in print.
	import { resolve } from '$app/paths';
	import { dev } from '$app/env';

	let { trace = null, slug, variant = 'chapter' } = $props();
</script>

{#if trace}
	<figure class="frontis frontis--{variant}" style:--aspect={trace.aspect ?? 1.5}>
		<img src={resolve(`/${trace.src}`)} alt={trace.title ?? ''} loading={variant === 'cover' ? 'eager' : 'lazy'} decoding="async" />
		{#if trace.title || trace.desc}
			<figcaption class="frontis__caption label">{trace.title ?? ''}{trace.title && trace.desc ? ' — ' : ''}{trace.desc ?? ''}</figcaption>
		{/if}
	</figure>
{:else if dev}
	<div class="frontis frontis--{variant} frontis--empty" aria-hidden="true">
		<span class="label">frontispiece · public/traces/{slug}.svg</span>
	</div>
{:else}
	<div class="frontis frontis--{variant} frontis--none" aria-hidden="true"></div>
{/if}

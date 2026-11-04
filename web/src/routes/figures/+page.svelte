<script>
	import { resolve } from '$app/paths';
	import RunningHead from '#lib/shell/RunningHead.svelte';
	let { data } = $props();
</script>

<svelte:head><title>Figures — PicoGPT</title></svelte:head>
<RunningHead meta={{ num: '', title: 'Figures' }} />
<main class="book contents-section contents-section--page">
	<h1 class="contents-section__h">Figures in the book</h1>
	<ol class="toc">
		{#each data.figures as f (f.id)}
			<li class="toc__item" class:toc__item--live={f.built}>
				<span class="toc__num" style="font-size:1.1rem">{f.num}</span>
				<div class="toc__text">
					<a class="toc__title" href={resolve(`/figures/${f.id}/`)}>{f.id}</a>
					<span class="toc__gist">{f.caption}</span>
				</div>
				<span class="toc__state label">{f.built ? 'built' : 'in preparation'}</span>
			</li>
		{/each}
	</ol>
</main>

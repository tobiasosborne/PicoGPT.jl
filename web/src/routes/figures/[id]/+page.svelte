<script>
	// One figure, alone: for figure builders, for review, and for films.
	//   /figures/<id>/            interactive
	//   /figures/<id>/?print      static (print) state
	//   /figures/<id>/?film&t=0.5 static frame at time t; window.__pico.setT(t) changes it
	// <html data-figure-ready="1"> is set once fonts are loaded and the figure has laid out.
	import { onMount, setContext } from 'svelte';
	import { makeSpecimen } from '#lib/model/specimen.js';
	import Figure from '#lib/shell/Figure.svelte';
	import RunningHead from '#lib/shell/RunningHead.svelte';

	let { data } = $props();
	const specimen = $derived(makeSpecimen(data.specimen));
	setContext('specimen', () => specimen);

	let mode = $state('interactive');
	let t = $state(null);
	const attrs = $derived(t == null ? data.fig.attrs : { ...data.fig.attrs, t: String(t) });

	onMount(async () => {
		const q = new URLSearchParams(location.search);
		if (q.has('print')) mode = 'print';
		if (q.has('film')) {
			document.documentElement.classList.add('film');
			t = q.has('t') ? +q.get('t') : 0;
		}
		window.__pico = { setT: (x) => (t = x), get t() { return t; } };
		await document.fonts.ready;
		await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
		document.documentElement.dataset.figureReady = '1';
	});
</script>

<svelte:head><title>{data.fig.id} — PicoGPT figures</title></svelte:head>
{#if mode !== 'print' && t == null}<RunningHead meta={{ num: data.fig.chapter.num, title: data.fig.chapter.title }} />{/if}
<main class="figure-solo" class:print-edition-root={mode === 'print'}>
	<p class="figure-solo__meta label">{data.fig.id} · chapter {data.fig.chapter.num} · {mode}{t != null ? ` · t = ${t}` : ''}</p>
	<div class="book flow">
		<Figure {...data.fig} {attrs} entry={data.figures[data.fig.id]} {mode} />
	</div>
</main>

<style>
	.figure-solo { padding: 3rem 0 5rem; }
	.figure-solo__meta { width: min(var(--wide), 100% - 2 * var(--pad)); margin: 0 auto 1rem; color: var(--ink-3); }
	:global(html.film) .figure-solo__meta { display: none; }
</style>

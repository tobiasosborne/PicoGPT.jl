<script>
	// One figure, alone: for figure builders, for review, and for films.
	//   /figures/<id>/            interactive
	//   /figures/<id>/?print      static (print) state
	//   /figures/<id>/?film&t=0.5 static frame at time t; window.__pico.setT(t) changes it
	// <html data-figure-ready="1"> is set once fonts are loaded and the figure has laid out.
	import { resolve } from '$app/paths';
	import { onMount, setContext, tick } from 'svelte';
	import { makeSpecimen } from '#lib/model/specimen.js';
	import Figure from '#lib/shell/Figure.svelte';
	import RunningHead from '#lib/shell/RunningHead.svelte';

	let { data } = $props();
	const specimen = $derived(makeSpecimen(data.specimen));
	setContext('specimen', () => specimen);
	setContext('artifacts', () => data.artifacts ?? {});

	let mode = $state('interactive');
	let t = $state(null);
	const figureLinks = (html) => html.replace(/href="\.\.\/([^"#]*)(#[^"]*)?"/g, (_, route, hash = '') => `href="${resolve(`/${route}`)}${hash}"`).replace(/href="#([^"]*)"/g, (_, hash) => `href="${resolve(`/${data.fig.chapter.slug}/`)}#${hash}"`);
	const attrs = $derived(t == null ? data.fig.attrs : { ...data.fig.attrs, t: String(t) });

	onMount(async () => {
		const q = new URLSearchParams(location.search);
		if (q.has('print')) mode = 'print';
		if (q.has('film')) {
			document.documentElement.classList.add('film');
			t = q.has('t') ? +q.get('t') : 0;
		}
		window.__pico = {
			async setT(x) {
				if (!Number.isFinite(x)) throw new TypeError('Figure time must be finite');
				t = Math.max(0, Math.min(1, x));
				await tick();
				await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
			},
			get t() { return t; }
		};
		await document.fonts.ready;
		await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
		document.documentElement.dataset.figureReady = '1';
	});
</script>

<svelte:head><title>{data.fig.id} — PicoGPT figures</title></svelte:head>
{#if mode !== 'print' && t == null}<RunningHead meta={{ num: data.fig.chapter.num, title: data.fig.chapter.title }} />{/if}
<main class="figure-solo" class:print-edition-root={mode === 'print'}>
	{#if t != null}
		<header class="film-heading">
			<p class="label">PicoGPT · Chapter {data.fig.chapter.num}</p>
			<h1>{data.figures[data.fig.id]?.meta?.title ?? data.fig.id}</h1>
		</header>
	{/if}
	<p class="figure-solo__meta label">{data.fig.id} · chapter {data.fig.chapter.num} · {mode}{t != null ? ` · t = ${t}` : ''}</p>
	<div class="book flow">
		<Figure {...data.fig} caption={figureLinks(data.fig.caption)} notes={data.fig.notes.map(figureLinks)} {attrs} entry={data.figures[data.fig.id]} {mode} />
	</div>
</main>

<style>
	.figure-solo { padding: 3rem 0 5rem; }
	.figure-solo__meta { width: min(var(--wide), 100% - 2 * var(--pad)); margin: 0 auto 1rem; color: var(--ink-3); }
	:global(html.film) .figure-solo__meta { display: none; }
	.film-heading { width: min(1040px, 100% - 96px); margin: 0 auto 24px; }
	.film-heading p { color: var(--ink-3); font-size: 11px; letter-spacing: 0.1em; }
	.film-heading h1 { font-size: 32px; line-height: 1.3; margin-top: 8px; }
	:global(html.film) .figure-solo { padding: 36px 0 0; }
	:global(html.film) .book { width: 1040px; max-width: calc(100% - 96px); }
	:global(html.film .fig) { width: 100%; margin: 0; }
	:global(html.film .fx-live__controls),
	:global(html.film .fx-ladder__reset), :global(html.film .fx-fb__theta:not(.is-on)) { display: none; }
	:global(html.film .fx-fb__theta.is-on) { text-decoration: none; }
	:global(html.film *), :global(html.film *::before), :global(html.film *::after) { transition: none !important; animation: none !important; caret-color: transparent; }
</style>

<script module>
	export const meta = { size: 'wide', title: 'The Shakespeare model at four temperatures' };
</script>

<script>
	// Figure: the trained Shakespeare model writing from the same prompt at each exported
	// temperature τ (last checkpoint). Set as text columns. Words (runs of two or more
	// letters) that never occur in the training text are underlined with dots; above each
	// column, the share of words that do, as a bar. Data: artifacts/shakespeare/samples.json,
	// word counts computed on the server from artifacts/data/tinyshakespeare.txt.
	import { getContext } from 'svelte';
	import { fmt } from '../primitives/scales.js';

	let { attrs = {}, mode = 'interactive' } = $props();
	const data = $derived(getContext('artifacts')?.()?.shakespeare?.samples ?? null);
	const step = $derived(data ? (attrs.step ? +attrs.step : Math.max(...data.steps)) : null);
	const cols = $derived(data ? data.temperatures.map((tau) => data.samples.find((s) => s.step === step && s.temperature === tau)).filter(Boolean) : []);
	const chars = $derived(+(attrs.chars ?? 330));

	/** The sample as runs: {text, unknown} — unknown words underlined. Cut at a line end near `chars`. */
	function runs(s) {
		let text = s.text.replace(/^\n+/, '');
		const off = s.text.length - text.length;
		let cut = text.lastIndexOf('\n', chars);
		if (cut < chars * 0.6) cut = text.lastIndexOf(' ', chars);
		text = text.slice(0, cut > 0 ? cut : chars);
		const out = [];
		let at = 0;
		for (const [a0, b0] of s.words?.unknown ?? []) {
			const a = a0 - off, b = b0 - off;
			if (b <= 0 || a >= text.length) continue;
			if (a > at) out.push({ text: text.slice(at, a), unknown: false });
			out.push({ text: text.slice(Math.max(a, at), Math.min(b, text.length)), unknown: true });
			at = Math.min(b, text.length);
		}
		if (at < text.length) out.push({ text: text.slice(at), unknown: false });
		return out;
	}
	const share = (s) => (s.words && s.words.total ? s.words.known / s.words.total : null);
</script>

<div class="fx fx-stemp" class:is-print={mode === 'print'} role="img" aria-label="The Shakespeare model at step {step}, writing at temperatures {cols.map((c) => `${c.temperature} (${share(c) != null ? Math.round(100 * share(c)) + '% real words' : ''})`).join(', ')}">
	{#if !data}
		<p class="fig__pending">The Shakespeare model’s samples (artifacts/shakespeare/samples.json) are not present in this build.</p>
	{:else}
		<div class="fx-stemp__grid" style:--n={cols.length}>
			{#each cols as c (c.temperature)}
				{@const sh = share(c)}
				<section class="fx-stemp__col">
					<p class="fx-stemp__head"><span class="fx-stemp__tau">τ = {c.temperature}</span>{#if c.temperature === 1}<span class="fx-stemp__note"> the model’s own</span>{/if}</p>
					{#if sh != null}
						<div class="fx-stemp__bar" aria-hidden="true"><span style:width="{sh * 100}%"></span></div>
						<p class="fx-stemp__words">{c.words.known} of {c.words.total} words real ({Math.round(sh * 100)} %)</p>
					{/if}
					<p class="fx-stemp__text">{#each runs(c) as r, k (k)}{#if r.unknown}<span class="fx-stemp__unk">{r.text}</span>{:else}{r.text}{/if}{/each}</p>
				</section>
			{/each}
		</div>
		<p class="fx-stemp__foot">Step {step}. Every sample starts from a newline; underlined with dots: words that never occur in the training text.</p>
	{/if}
</div>

<style>
	.fx-stemp__grid { display: grid; grid-template-columns: repeat(var(--n), minmax(0, 1fr)); gap: 0 1.4rem; }
	@media (max-width: 720px) { .fx-stemp:not(.is-print) .fx-stemp__grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1.2rem 1rem; } }
	.fx-stemp__col { min-width: 0; }
	.fx-stemp__head { font-family: var(--sans); font-size: var(--fs-label); margin: 0; color: var(--ink); text-indent: 0; }
	.fx-stemp__tau { font-weight: 650; }
	.fx-stemp__note { color: var(--ink-3); margin-left: 0.45em; }
	.fx-stemp__bar { height: 5px; margin: 0.45rem 0 0.25rem; border-bottom: var(--hair) solid var(--ink-3); position: relative; }
	.fx-stemp__bar span { display: block; height: 100%; background: var(--value); }
	.fx-stemp__words { font-family: var(--sans); font-size: var(--fs-micro); color: var(--ink-2); margin: 0 0 0.6rem; text-indent: 0; font-variant-numeric: tabular-nums lining-nums; }
	.fx-stemp__text { font-family: var(--serif); font-size: 0.8rem; line-height: 1.38; white-space: pre-wrap; overflow-wrap: anywhere; margin: 0; color: var(--ink); text-indent: 0; hyphens: none; }
	.fx-stemp__unk { text-decoration: underline dotted var(--ink-3); text-decoration-thickness: 1.2px; text-underline-offset: 3px; }
	.fx-stemp__foot { font-family: var(--sans); font-size: var(--fs-micro); color: var(--ink-3); margin: 0.8rem 0 0; text-indent: 0; }
</style>

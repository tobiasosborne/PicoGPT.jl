<script module>
	export const meta = { size: 'wide', title: 'Where the large model recites its training text' };
</script>

<script>
	import { getContext } from 'svelte';
	let { mode = 'interactive' } = $props();
	const data = $derived(getContext('artifacts')?.()?.overfitting);
	const record = $derived(data?.memorised);
	// Select the final-model sample with the longest exact match, across both sources.
	const sample = $derived(record?.samples.filter((s) => s.step === data.large.final.step).reduce((a, b) => longest(b) > longest(a) ? b : a, null));
	const longest = (s) => Math.max(0, ...(s?.spans ?? []).map((p) => p.end - p.start));
	const covered = $derived(sample ? Array.from(sample.text, (_, i) => sample.spans.some((p) => p.start <= i && i < p.end)) : []);
	const runs = $derived.by(() => {
		if (!sample) return [];
		const out = [];
		for (let i = 0; i < sample.text.length; i++) {
			if (!i || covered[i] !== covered[i - 1]) out.push({ text: '', matched: covered[i] });
			out.at(-1).text += sample.text[i];
		}
		return out;
	});
</script>

<div class="fx fx-memorised" class:is-print={mode === 'print'}>
	{#if !sample}
		<p class="fig__pending">The precomputed matches are not present in this build.</p>
	{:else}
		<div class="memorised__grid">
			<section>
				<p class="memorised__head">Step {sample.step.toLocaleString('en-GB')} · τ = {sample.temperature} · seed {sample.seed}</p>
				<p class="memorised__text">{#each runs as run, i (i)}{#if run.matched}<span class="memorised__match">{run.text}</span>{:else}{run.text}{/if}{/each}</p>
			</section>
			<div class="memorised__notes">
				<p class="memorised__head">An exact match</p>
				<p class="memorised__stat">{covered.filter(Boolean).length}<span> of {sample.text.length} characters underlined</span></p>
				<p>Underlined: every character in a maximal exact match of at least {record.minimum} characters to the training prefix. Case, punctuation and line breaks count.</p>
				<p>Longest single match: {longest(sample)} characters. Overlapping matches share one underline.</p>
				<p>Selected for the longest match among the recorded final-model samples. This is an exhibit of copying, not an estimate of how often it happens.</p>
			</div>
		</div>
	{/if}
</div>

<style>
	.memorised__grid { display: grid; grid-template-columns: 1.15fr 1fr; gap: 3rem; }
	.memorised__head { font-family: var(--sans); font-size: var(--fs-label); color: var(--ink-2); margin: 0; text-indent: 0; font-variant-numeric: tabular-nums lining-nums; }
	.memorised__text { font-family: var(--serif); font-size: 0.92rem; line-height: 1.6; white-space: pre-wrap; overflow-wrap: anywhere; margin: 0; text-indent: 0; hyphens: none; }
	.memorised__match { text-decoration: underline; text-decoration-color: var(--value); text-decoration-thickness: 1px; text-underline-offset: 4px; }
	.memorised__notes { font-size: 0.84rem; line-height: 1.5; color: var(--ink-2); }
	.memorised__notes p { text-indent: 0; margin: 0 0 0.8rem; }
	.memorised__notes .memorised__stat { font-family: var(--sans); font-size: 2rem; line-height: 1.1; margin: 0.6rem 0 1.2rem; color: var(--ink); }
	.memorised__stat span { display: block; font-size: var(--fs-label); color: var(--ink-2); margin-top: 0.4rem; }
	.is-print .memorised__grid { gap: 24px; }
	.is-print .memorised__text { font-size: var(--fs-body); }
	.is-print .memorised__notes { font-size: var(--fs-small); }
	.is-print .memorised__notes .memorised__stat { font-size: 22px; margin: 0.6em 0 0.8em; }
	@media (max-width: 700px) { .fx-memorised:not(.is-print) .memorised__grid { grid-template-columns: 1fr; gap: 1.8rem; } }
</style>

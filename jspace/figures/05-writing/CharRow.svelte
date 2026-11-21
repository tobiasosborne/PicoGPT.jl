<script>
	// A row of the vocabulary's characters, set in Literata, for choosing one of them:
	// click a character, or ←/→ when the row has focus. The chosen character is set bold
	// and underlined; `marks` sets others in bold without the rule (e.g. the partner of a
	// swap). Only drawn in interactive mode; print figures name their choice in a label.
	import { showChar, clamp } from '#figures/primitives/scales.js';

	let { vocab = [], value = $bindable(0), marks = [], x = 0, y = 0, step = 22, name = '', label = 'character', disabled = [] } = $props();

	function key(e) {
		if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
		let k = value;
		do k = clamp(k + (e.key === 'ArrowRight' ? 1 : -1), 0, vocab.length - 1);
		while (disabled.includes(k) && k > 0 && k < vocab.length - 1);
		if (!disabled.includes(k)) value = k;
		e.preventDefault();
	}
</script>

<g class="crow" role="radiogroup" aria-label={label} tabindex="0" onkeydown={key}>
	{#if name}<text class="lbl crow__name" x={x - 10} y={y} text-anchor="end">{name}</text>{/if}
	{#each vocab as ch, a (a)}
		{@const cx = x + step * (a + 0.5)}
		<text class="tok crow__ch" class:is-on={a === value} class:is-mark={marks.includes(a)} class:is-off={disabled.includes(a)} x={cx} y={y} text-anchor="middle">{showChar(ch)}</text>
		{#if a === value}<line class="crow__rule" x1={cx - step * 0.32} x2={cx + step * 0.32} y1={y + 5} y2={y + 5} />{/if}
		<rect class="hit" x={cx - step / 2} y={y - 18} width={step} height={26} role="radio" aria-checked={a === value} aria-label="‘{showChar(ch)}’" tabindex="-1"
			onclick={() => { if (!disabled.includes(a)) value = a; }} onkeydown={() => {}} />
	{/each}
</g>

<style>
	.crow:focus { outline: none; }
	.crow:focus-visible .crow__ch { text-decoration: underline dotted; }
	.crow__ch { font-size: 18px; fill: var(--ink-3); cursor: pointer; transition: fill var(--t-quick); }
	.crow__ch.is-on, .crow__ch.is-mark { fill: var(--ink); font-weight: 650; }
	.crow__ch.is-off { opacity: 0.35; cursor: default; }
	.crow__rule { stroke: var(--ink); stroke-width: 1.2px; }
	.crow__name { font-family: var(--serif); font-style: italic; font-size: 14px; fill: var(--ink-2); }
</style>

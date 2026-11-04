<script>
	// The frame every figure lives in: number, caption, margin notes, and the
	// component itself. The component is a pure function of (specimen, attrs, state);
	// in mode="print" it must render its static state with no interactive affordances.
	let { id, num, caption = '', notes = [], attrs = {}, classes = [], entry = null, mode = 'interactive', chapter = null } = $props();

	const size = $derived(attrs.size ?? entry?.meta?.size ?? 'main');
	const Component = $derived(entry?.component);
	const Key = $derived(entry?.Key ?? null);
</script>

<figure class="fig fig--{size} {classes.join(' ')}" {id} data-fig={id} data-mode={mode}>
	<div class="fig__body">
		{#if Component}
			<Component {attrs} {mode} {id} />
		{:else}
			<div class="fig__pending" role="note">
				<span class="label">{id}</span>
				<span>Figure in preparation.</span>
			</div>
		{/if}
	</div>
	<figcaption class="fig__foot">
		{#if caption}
			<div class="fig__caption"><span class="fig__num">Figure {num}</span>{@html caption}</div>
		{/if}
		{#if notes.length || Key}
			<div class="fig__notes">
				{#if Key}<Key {attrs} {mode} />{/if}
				{@html notes.join('')}
			</div>
		{/if}
	</figcaption>
</figure>

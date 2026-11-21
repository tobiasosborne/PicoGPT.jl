<script module>
	import Figure from '#figures/05-attention/fig-attention-triangle.svelte';
	export { meta, Key } from '#figures/05-attention/fig-attention-triangle.svelte';
</script>

<script>
	let props = $props();
</script>

<Figure {...props} />

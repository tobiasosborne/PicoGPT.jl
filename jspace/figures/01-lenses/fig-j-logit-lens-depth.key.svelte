<script>
	// How to read the depth axis, the first time it appears: one block of the Shakespeare
	// model's axis (depths 0, ½, 1), enlarged, with what each tick means.
	import { getContext } from 'svelte';
	import HowToRead from '#figures/primitives/HowToRead.svelte';
	import DepthAxis from '../primitives/DepthAxis.svelte';

	const sh = $derived(getContext('artifacts')()?.jspace?.shakespeare ?? null);
	const depths = $derived(sh ? sh.depth.slice(0, 3) : [0, 0.5, 1]);
	const kinds = $derived(sh ? sh.depth_kind.slice(0, 3) : ['embed', 'attention', 'mlp']);
	const x0 = 40, w = 140, y = 92;
</script>

<HowToRead width={244} height={150} notes={[
	{ ax: x0, ay: y, side: 'above', len: 66, text: '0: the stream after\nthe embedding', align: 'start', dx: -26 },
	{ ax: x0 + w / 2, ay: y, side: 'above', len: 36, text: '½: after the attention\nupdate of block 1' },
	{ ax: x0 + w, ay: y, side: 'above', len: 10, text: '1: after its MLP update', align: 'end', dx: 46 }
]}>
	{#snippet glyph()}
		<g class="fx">
			<DepthAxis {depths} {kinds} x={x0} width={w} {y} title="" />
		</g>
	{/snippet}
</HowToRead>

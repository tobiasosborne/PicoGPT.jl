<script>
	// How to read the residual stem, drawn from the figure's own selected column.
	import { getContext } from 'svelte';
	import StemKey from '../primitives/StemKey.svelte';
	import { stemWindow, rms } from '../primitives/stem.js';
	import { clamp } from '../primitives/scales.js';

	let { attrs = {} } = $props();
	const specimen = $derived(getContext('specimen')());
	const gt = $derived(specimen.lab.gradTrace(specimen.theta));
	const win = $derived(clamp((+attrs.window || 1) - 1, 0, gt.windows.length - 1));
	const col = $derived(clamp((+attrs.column || 8) - 1, 0, specimen.config.T - 1));
	const c = $derived(stemWindow(gt, win, specimen.config).cols[col]);
	// the key's widths use the figure's one scale: 12 px for the widest column of any window
	const k = $derived.by(() => {
		let m = 0;
		for (let b = 0; b < gt.windows.length; b++) for (const q of stemWindow(gt, b, specimen.config).cols) m = Math.max(m, rms(q.x0), rms(q.a), rms(q.mid), rms(q.y), rms(q.out));
		return 12 / m;
	});
</script>

<StemKey x0={c.x0} a={c.a} mid={c.mid} {k} />

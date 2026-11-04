<script>
	// How to read the pullback (covector) stem, drawn from the figure's own selected column
	// at θ₀ (or the trained θ, attrs theta="trained").
	import { getContext } from 'svelte';
	import PullbackKey from '../primitives/PullbackKey.svelte';
	import { stemWindow, rms } from '../primitives/stem.js';
	import { clamp, fmt } from '../primitives/scales.js';

	let { attrs = {} } = $props();
	const specimen = $derived(getContext('specimen')());
	const theta = $derived(attrs.theta === 'trained' ? specimen.theta : specimen.theta0);
	const gt = $derived(specimen.lab.gradTrace(theta));
	const win = $derived(clamp((+attrs.window || 1) - 1, 0, gt.windows.length - 1));
	const col = $derived(clamp((+attrs.column || 8) - 1, 0, specimen.config.T - 1));
	const all = $derived(gt.windows.map((_, b) => stemWindow(gt, b, specimen.config)));
	const c = $derived(all[win].cols[col]);
	// the figure's covector width scale: 13 px for the widest adjoint of any window
	const k = $derived(13 / Math.max(...all.flatMap((s) => s.cols.flatMap((q) => [rms(q.bx0), rms(q.batt), rms(q.bmid), rms(q.bmlp), rms(q.bout)]))));
	function sci(x) {
		if (!(x > 0)) return '0';
		const e = Math.floor(Math.log10(x)), m = x / 10 ** e;
		const sup = String(e).replace('-', '⁻').replace(/\d/g, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+d]);
		return e >= -1 && e <= 1 ? fmt(x, 2) : `${fmt(m, 1)}×10${sup}`;
	}
</script>

<PullbackKey bx0={c.bx0} batt={c.batt} bmid={c.bmid} {k} {sci} />

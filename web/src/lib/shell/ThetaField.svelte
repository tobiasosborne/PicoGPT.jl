<script>
	// θ, set as type: every parameter of the specimen in the order the code stores
	// them (BRIEF §3), as a justified run of numbers. The weight of the type is
	// |θ_i|; each block of θ starts with its name. Point at a number to read its
	// name. Every mark here is one real parameter.
	import { fmtDense, fmt } from '#figures/primitives/scales.js';

	let { specimen, theta = specimen.theta, digits = 2 } = $props();

	const SUB = { q: 'q', k: 'k', v: 'v', o: 'o' };
	function pretty(name) {
		const base = name.split('.').at(-1);
		const m = base.match(/^(gamma|beta)(\d|f)$/);
		if (m) return { sym: m[1] === 'gamma' ? 'γ' : 'β', sub: m[2] };
		const w = base.match(/^([Wb])(q|k|v|o|1|2|out)$/);
		if (w) return { sym: w[1], sub: w[2] };
		return { sym: base, sub: '' };
	}
	const blocks = $derived(
		specimen.params(theta).map((p) => ({
			...p,
			label: pretty(p.name),
			ref: Math.max(...p.values.map(Math.abs)) || 1
		}))
	);
	// one reference for the whole field, so weights are comparable across blocks
	const global = $derived(Math.max(...theta.map(Math.abs)) || 1);
	const weightOf = (v) => Math.round(110 + 760 * Math.pow(Math.min(1, Math.abs(v) / global), 0.55));

	let hover = $state(null);
	function describe(b, k) {
		const v = b.values[k];
		const idx = b.shape.length === 2 ? `[${(k % b.shape[0]) + 1}, ${Math.floor(k / b.shape[0]) + 1}]` : `[${k + 1}]`;
		return { name: `${b.label.sym}${b.label.sub ? '_' + b.label.sub : ''}${idx}`, value: fmt(v, 4), at: b.offset + k + 1 };
	}
	function over(e) {
		const el = e.target.closest?.('[data-k]');
		if (!el) return;
		const b = blocks[+el.dataset.b];
		hover = describe(b, +el.dataset.k);
	}
	function key(e) {
		let k = (hover?.at ?? 1) - 1;
		if (e.key === 'ArrowRight' || e.key === 'ArrowDown') k++;
		else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') k--;
		else if (e.key === 'Home') k = 0;
		else if (e.key === 'End') k = specimen.paramCount - 1;
		else return;
		e.preventDefault(); k = Math.max(0, Math.min(specimen.paramCount - 1, k));
		const b = blocks.find((b) => k >= b.offset && k < b.offset + b.size);
		hover = describe(b, k - b.offset);
	}
</script>

<div class="theta">
	<div class="theta__field" role="slider" tabindex="0" aria-label="trained parameters in storage order" aria-valuemin="1" aria-valuemax={specimen.paramCount} aria-valuenow={hover?.at ?? 1} aria-valuetext={hover ? `${hover.name} = ${hover.value}` : undefined} onkeydown={key} onfocus={() => { if (!hover) hover = describe(blocks[0], 0); }} onpointerover={over} onpointerleave={() => (hover = null)}>
		{#each blocks as b, bi (b.name)}
			<span class="theta__name" title={b.name}><i>{b.label.sym}</i>{#if b.label.sub}<sub>{b.label.sub}</sub>{/if}</span>
			{#each b.values as v, k (k)}
				<span class="theta__n" data-b={bi} data-k={k} style:font-weight={weightOf(v)}>{fmtDense(v, digits)}</span>{' '}
			{/each}
		{/each}
	</div>
	<p class="theta__readout label" aria-live="polite">
		{#if hover}
			<span class="theta__hname">{hover.name}</span> = {hover.value} <span class="theta__at">· θ<sub>{hover.at}</sub> of {specimen.paramCount}</span>
		{:else}
			<span class="theta__hint">θ ∈ ℝ<sup>{specimen.paramCount}</sup>: every parameter of the trained specimen, in the order the code keeps them, read across before down (unlike the computation). The weight of the type is |θ<sub>i</sub>|.</span>
		{/if}
	</p>
</div>

<script>
	// Time as a thing you can hold. t ∈ [0,1] is the figure's only clock; every
	// animated figure is a pure function of t. Play runs t forward at a constant
	// rate; under prefers-reduced-motion it steps from phase to phase instead.
	// Keyboard: ←/→ step between phases, Home/End, Space plays.
	import { onDestroy } from 'svelte';
	import { reducedMotion, clamp } from './scales.js';

	let { t = $bindable(1), phases = [], duration = 2400, label = 'time', ink = 'value' } = $props();

	let track = $state();
	let playing = $state(false);
	let raf = 0;

	const stops = $derived([0, ...phases.map((p) => p.at), 1].filter((v, i, a) => a.indexOf(v) === i).sort((a, b) => a - b));

	function setFromPointer(e) {
		const r = track.getBoundingClientRect();
		t = clamp((e.clientX - r.left) / r.width, 0, 1);
	}
	function down(e) {
		stop();
		track.setPointerCapture(e.pointerId);
		setFromPointer(e);
	}
	function move(e) { if (track.hasPointerCapture?.(e.pointerId)) setFromPointer(e); }
	function nextStop(dir) {
		const eps = 1e-6;
		return dir > 0 ? (stops.find((s) => s > t + eps) ?? 1) : ([...stops].reverse().find((s) => s < t - eps) ?? 0);
	}
	function key(e) {
		if (e.key === 'ArrowRight' || e.key === 'ArrowUp') { t = nextStop(1); e.preventDefault(); }
		else if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') { t = nextStop(-1); e.preventDefault(); }
		else if (e.key === 'Home') { t = 0; e.preventDefault(); }
		else if (e.key === 'End') { t = 1; e.preventDefault(); }
		else if (e.key === ' ') { toggle(); e.preventDefault(); }
	}
	function stop() { playing = false; if (typeof cancelAnimationFrame !== 'undefined') cancelAnimationFrame(raf); }
	function toggle() {
		if (playing) return stop();
		if (t >= 1 - 1e-6) t = 0;
		if (reducedMotion()) { t = nextStop(1); return; }
		playing = true;
		let last = performance.now();
		const frame = (now) => {
			if (!playing) return;
			t = Math.min(1, t + (now - last) / duration);
			last = now;
			if (t >= 1) { playing = false; return; }
			raf = requestAnimationFrame(frame);
		};
		raf = requestAnimationFrame(frame);
	}
	onDestroy(stop);
	const current = $derived([...phases].reverse().find((p) => t >= p.at - 1e-6)?.label ?? phases[0]?.label ?? '');
</script>

<div class="scrub scrub--{ink}" role="group" aria-label={label}>
	<button class="scrub__play" onclick={toggle} aria-label={playing ? 'Pause' : 'Play'}>
		{#if playing}
			<svg viewBox="0 0 12 12" width="10" height="10" aria-hidden="true"><rect x="2.6" y="2" width="2.2" height="8" /><rect x="7.2" y="2" width="2.2" height="8" /></svg>
		{:else}
			<svg viewBox="0 0 12 12" width="10" height="10" aria-hidden="true"><path d="M3.4 1.8 L10 6 L3.4 10.2 Z" /></svg>
		{/if}
	</button>
	<div
		class="scrub__track"
		bind:this={track}
		role="slider"
		tabindex="0"
		aria-label={label}
		aria-valuemin="0"
		aria-valuemax="1"
		aria-valuenow={t}
		aria-valuetext={current}
		onpointerdown={down}
		onpointermove={move}
		onkeydown={key}
	>
		<span class="scrub__rail"></span>
		<span class="scrub__done" style:width="{t * 100}%"></span>
		{#each phases as p (p.at)}
			<span class="scrub__phase" class:scrub__phase--past={t >= p.at - 1e-6} style:left="{p.at * 100}%"><span class="scrub__label label">{p.label}</span></span>
		{/each}
		<span class="scrub__head" style:left="{t * 100}%"></span>
	</div>
</div>

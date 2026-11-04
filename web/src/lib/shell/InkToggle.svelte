<script>
	// Paper or night. Defaults to the reader's system setting; a choice is remembered.
	import { onMount } from 'svelte';
	let ink = $state('auto');
	onMount(() => { ink = document.documentElement.dataset.ink ?? 'auto'; });
	function toggle() {
		const dark = ink === 'night' || (ink === 'auto' && matchMedia('(prefers-color-scheme: dark)').matches);
		ink = dark ? 'paper' : 'night';
		document.documentElement.dataset.ink = ink;
		try { localStorage.setItem('pico-ink', ink); } catch {}
	}
</script>

<button class="inktoggle" onclick={toggle} aria-label="Switch between paper and night" title="Paper / night">
	<svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true"><circle cx="8" cy="8" r="6.2" fill="none" stroke="currentColor" stroke-width="1" /><path d="M8 1.8 A6.2 6.2 0 0 1 8 14.2 Z" fill="currentColor" /></svg>
</button>

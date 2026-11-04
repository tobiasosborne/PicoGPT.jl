<script>
	import '#lib/styles/fonts.css';
	import '#lib/styles/tokens.css';
	import '#lib/styles/base.css';
	import '#lib/styles/book.css';
	import '#lib/styles/figures.css';
	import '#lib/styles/shell.css';
	import '#lib/styles/print.css';

	let { children } = $props();
</script>

<!-- shared SVG paint servers: the covector hatch (the "stack of planes" of a 1-form) -->
<svg width="0" height="0" style="position:absolute" aria-hidden="true" focusable="false">
	<defs>
		<pattern id="hatch-covector" width="3.2" height="3.2" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
			<line x1="0" y1="0" x2="0" y2="3.2" style="stroke: var(--covector); stroke-width: 1.15" />
		</pattern>
		<pattern id="hatch-ink" width="3.2" height="3.2" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
			<line x1="0" y1="0" x2="0" y2="3.2" style="stroke: var(--ink-3); stroke-width: 0.8" />
		</pattern>
	</defs>
</svg>

{@render children()}

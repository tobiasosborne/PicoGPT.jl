<script>
	// How to read the loss curve, drawn from the book's own run (artifacts/specimen.json).
	import { getContext } from 'svelte';
	import LossChartKey from '../primitives/LossChartKey.svelte';
	const specimen = getContext('specimen')();
</script>

<LossChartKey values={specimen.loss} lnV={Math.log(specimen.config.V)} floor={specimen.lab.floor} xMax={200} />

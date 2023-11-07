<script lang="ts">
	import { type Tracer, traceRange } from '$lib/trace';
	import SpanTree from './SpanTree.svelte';

	/**
	 * The Tracer you want to render
	 */
	export let trace: Tracer;

	$: range = traceRange(trace);
</script>

<!--
    @component
    Timeline and Tree view of the given Tracer
-->

<div class="grid content-start border">
	<div class="overflow-y-auto">
		{#if range}
			<SpanTree entries={trace.entries} {range} />
		{:else}
			<p class="text-sm">No trace available</p>
		{/if}
	</div>
</div>

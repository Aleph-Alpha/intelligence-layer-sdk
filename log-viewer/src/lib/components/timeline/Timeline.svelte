<script lang="ts">
	import { type DebugLog, logRange } from '$lib/log';
	import SpanTree from './SpanTree.svelte';

	/**
	 * The Debug Log you want to render
	 */
	export let log: DebugLog;

	$: range = logRange(log);
</script>

<!--
    @component
    Timeline and Tree view of the given DebugLog
-->

<div class="grid content-start border">
	<div class="bg-gray-950 px-2 py-1 text-sm font-extrabold text-white">
		{log.name}
	</div>

	<div class="overflow-y-auto">
		{#if range}
			<SpanTree logs={log.logs} {range} />
		{:else}
			<p class="text-sm">No logs available</p>
		{/if}
	</div>
</div>

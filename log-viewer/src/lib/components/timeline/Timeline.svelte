<script lang="ts">
	import { differenceInMilliseconds } from 'date-fns';
	import { type DebugLog, logRange, renderDuration } from '../../log';
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

{#if range}
	<div class="grid grid-cols-3 grid-rows-1 bg-gray-950 text-sm font-extrabold text-white">
		<div class="col-span-1 border-r border-white px-2 py-1">
			{log.name}
		</div>
		<div class="col-span-2 flex items-center justify-between px-2 py-1 text-xs">
			<span>0</span><span>{renderDuration(differenceInMilliseconds(range.to, range.from))}</span>
		</div>
	</div>

	<SpanTree logs={log.logs} {range} />
{:else}
	<p class="text-sm">No logs available</p>
{/if}

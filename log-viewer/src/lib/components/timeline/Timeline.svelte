<script lang="ts">
	import type { DebugLog } from '../../log';
	import LogDurations from './LogDurations.svelte';
	import SpanTree from './SpanTree.svelte';

	/**
	 * The Debug Log you want to render
	 */
	export let log: DebugLog;
</script>

<!--
    @component
    Timeline and Tree view of the given DebugLog
-->

<div class="grid grid-cols-3 grid-rows-1">
	<div class="col-span-1">
		<SpanTree logs={log.logs} />
	</div>
	<div class="col-span-2"><LogDurations logs={log.logs} /></div>
</div>

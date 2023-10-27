<script lang="ts">
	import { type DebugLog, logRange } from '../../log';
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

<SpanTree logs={log.logs} {range} />

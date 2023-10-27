<script lang="ts">
	import type { LogEntry } from '$lib/log';
	import JsonValue from './JsonValue.svelte';

	/**
	 * A LogEntry to render
	 */
	export let log: LogEntry;
</script>

<!--
    @component
    Renders a single log entry
-->

<div class="grid grid-cols-1">
	<div class="grid grid-cols-1 gap-0.5 bg-gray-950 p-2 text-white">
		<span class="text-xs font-extrabold">{log.timestamp}</span>
		<span>{log.message}</span>
	</div>
	<JsonValue value={log.value} />
</div>

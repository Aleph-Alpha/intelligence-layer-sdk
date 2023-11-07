<script lang="ts">
	import { isLogEntry, type Entry } from '$lib/trace';
	import LogEntry from './LogEntry.svelte';

	/**
	 * Any value that can be passed to JSON.stringify
	 */
	export let entries: Entry[];

	$: logs = entries.filter(isLogEntry);
</script>

<!--
    @component
    Renders any log entries within this list of spans and logs.
-->

{#if entries.length}
	<div class="grid grid-cols-1 gap-2">
		{#each logs as log}
			<LogEntry {log} />
		{/each}
	</div>
{/if}

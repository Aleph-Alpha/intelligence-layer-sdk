<script lang="ts">
	import { compareAsc } from 'date-fns';
	import { type Entry, isSpan } from '../../log';
	import SpanDuration from './SpanDuration.svelte';

	/**
	 * The list of log entries to render in the tree.
	 */
	export let logs: Entry[];

	// Filter out LogEntry's, only show the Span/TaskSpan in the tree
	$: spans = logs.filter(isSpan);

	let logTimes: Date[];
	$: {
		logTimes = logs.reduce<Date[]>((acc, i) => {
			if ('timestamp' in i) acc.push(new Date(i.timestamp));
			if ('start_timestamp' in i) acc.push(new Date(i.start_timestamp));
			if ('end_timestamp' in i) acc.push(new Date(i.end_timestamp));
			return acc;
		}, []);
		logTimes.sort(compareAsc);
		console.log(logTimes);
	}
</script>

<!--
    @compone nt
    A timeline of all sub-span durations
-->
<div class="w-full">
	{#each spans as span}
		<SpanDuration {span} runStart={logTimes[0]} runEnd={logTimes[logTimes.length - 1]} />
	{/each}
</div>

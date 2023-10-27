<script lang="ts">
	import { type Entry, type TimeRange, isSpan } from '../../log';
	import SpanRow from './SpanRow.svelte';

	/**
	 * The list of log entries to render in the tree.
	 */
	export let logs: Entry[];
	/**
	 * How deeply nested is this sub-tree?
	 */
	export let level: number = 0;
	/**
	 * The duration of the entire run of the logger
	 */
	export let range: TimeRange;

	// Filter out LogEntry's, only show the Span/TaskSpan in the tree
	$: spans = logs.filter(isSpan);
</script>

<!--
    @component
    A tree-view of Spans/TaskSpans to show their names and nested structure.

    This is a recursive component that builds itself up by creating sub-trees.
-->
{#if spans.length}
	<ul class:border-t={level === 0}>
		{#each spans as span}
			<li>
				<SpanRow {span} {level} {range} />
				<svelte:self logs={span.logs} level={level + 1} {range} />
			</li>
		{/each}
	</ul>
{:else if level === 0}
	<p class="text-sm">No spans available</p>
{/if}

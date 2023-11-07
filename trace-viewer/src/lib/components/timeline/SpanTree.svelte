<script lang="ts">
	import { type Entry, type TimeRange, isSpan } from '$lib/trace';
	import SpanRow from './SpanRow.svelte';

	/**
	 * The list of entries to render in the tree.
	 */
	export let entries: Entry[];
	/**
	 * How deeply nested is this sub-tree?
	 */
	export let level = 0;
	/**
	 * The duration of the entire run of the tracer
	 */
	export let range: TimeRange;

	// Filter out LogEntry's, only show the Span/TaskSpan in the tree
	$: spans = entries.filter(isSpan);
</script>

<!--
    @component
    A tree-view of Spans/TaskSpans to show their names and nested structure.

    This is a recursive component that builds itself up by creating sub-trees.
-->
{#if spans.length}
	<ul class:border-b={level === 0}>
		{#each spans as span}
			<li>
				<SpanRow {span} {level} {range} />
				<svelte:self entries={span.entries} level={level + 1} {range} />
			</li>
		{/each}
	</ul>
{/if}

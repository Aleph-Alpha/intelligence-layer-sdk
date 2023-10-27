<script lang="ts">
	import { type Entry, isSpan } from '../../log';

	/**
	 * The list of log entries to render in the tree.
	 */
	export let logs: Entry[];
	/**
	 * How deeply nested is this sub-tree?
	 */
	export let level: number = 0;

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
		{#each spans as log}
			<li>
				<button class="group w-full border-b text-left" style="padding-left: {level}em"
					><span class="block border-l border-gray-300 px-2 py-1 text-sm group-hover:bg-gray-50"
						>{log.name}</span
					></button
				>
				<svelte:self logs={log.logs} level={level + 1} />
			</li>
		{/each}
	</ul>
{/if}

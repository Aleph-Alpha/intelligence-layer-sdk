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
				<button
					class="group h-8 w-full border-b bg-gray-50 text-left"
					style="padding-left: {level}em"
				>
					<span
						class="block border-l border-gray-300 bg-white px-2 py-1 text-sm group-hover:bg-gray-100"
					>
						{log.name}
					</span>
				</button>
				<svelte:self logs={log.logs} level={level + 1} />
			</li>
		{/each}
	</ul>
{:else if level === 0}
	<p class="text-sm">No spans available</p>
{/if}

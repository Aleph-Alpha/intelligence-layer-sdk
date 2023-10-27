<script lang="ts">
	import type { SpanEntry, TimeRange } from '../../log';
	import SpanDuration from './SpanDuration.svelte';

	/**
	 * The Span to render in the tree.
	 */
	export let span: SpanEntry;
	/**
	 * How deeply nested is this sub-tree?
	 */
	export let level: number = 0;
	/**
	 * The duration of the entire run of the logger
	 */
	export let range: TimeRange;
</script>

<!--
    @component
    A tree-view of Spans/TaskSpans to show their names and nested structure.

    This is a recursive component that builds itself up by creating sub-trees.
-->

<div class="group grid grid-cols-3 grid-rows-1 items-stretch border-b bg-gray-50">
	<div class="col-span-1 grid items-stretch">
		<button class="group grid w-full items-stretch text-left" style="padding-left: {level}em">
			<span
				class="grid items-center border-l border-r border-gray-300 bg-white px-2 text-sm group-hover:bg-gray-100"
			>
				<span>{span.name}</span>
			</span>
		</button>
	</div>
	<div class="col-span-2 bg-white px-2 group-hover:bg-gray-100">
		<SpanDuration {span} {range} />
	</div>
</div>

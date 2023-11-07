<script lang="ts">
	import { differenceInMilliseconds } from 'date-fns';
	import { activeSpan } from '$lib/active';
	import { type SpanEntry, type TimeRange, renderDuration } from '$lib/trace';

	/**
	 * A Span or TaskSpan to show the duration of
	 */
	export let span: SpanEntry;
	/**
	 * The duration of the entire run of the trace
	 */
	export let range: TimeRange;

	$: spanStart = new Date(span.start_timestamp);
	$: spanOffset = differenceInMilliseconds(spanStart, range.from);
	$: spanLength = differenceInMilliseconds(new Date(span.end_timestamp), spanStart);
	$: runLength = differenceInMilliseconds(range.to, range.from);
</script>

<!--
    @component
    A view of Span durations in relation to the entire duration of the trace.

    This is a recursive component that builds itself up by creating the same component for sub-spans.
-->
<button
	class="bg-accent-400 py-0.5 text-right text-xs font-extrabold text-gray-950 shadow outline-none ring-1 ring-gray-950/20 hover:bg-accent-500"
	on:click={() => activeSpan.set(span)}
	style="margin-left: {Math.round((spanOffset / runLength) * 100)}%; width:{Math.round(
		(spanLength / runLength) * 100
	)}%;"
>
	<span class="px-1">{renderDuration(spanLength)}</span>
</button>

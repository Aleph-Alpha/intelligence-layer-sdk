<script lang="ts">
	import { differenceInMilliseconds } from 'date-fns';
	import { type SpanEntry, isSpan } from '../../log';

	/**
	 * A Span or TaskSpan to show the duration of
	 */
	export let span: SpanEntry;
	/**
	 * The start of the entire run of the logger
	 */
	export let runStart: Date;
	/**
	 * The end of the entire run of the logger
	 */
	export let runEnd: Date;

	$: spanStart = new Date(span.start_timestamp);
	$: spanOffset = differenceInMilliseconds(spanStart, runStart);
	$: spanLength = differenceInMilliseconds(new Date(span.end_timestamp), spanStart);
	$: runLength = differenceInMilliseconds(runEnd, runStart);

	// Filter out LogEntry's, only show the Span/TaskSpan in the tree
	$: childSpans = span.logs.filter(isSpan);

	function renderDuration(spanLength: number): string {
		let unit = 'ms';
		let length = spanLength;
		if (length > 1000) {
			length /= 1000;
			unit = 's';
			if (length > 60) {
				length /= 60;
				unit = 'min';
				if (length > 60) {
					length /= 60;
					unit = 'h';
				}
			}
		}

		return `${length.toLocaleString('en-US')}${unit}`;
	}
</script>

<!--
    @component
    A view of Span durations in relation to the entire duration of the log.

    This is a recursive component that builds itself up by creating the same component for sub-spans.
-->
<div class="h-8 w-full border-t py-0.5 last:border-b hover:bg-gray-50">
	<button
		class="bg-accent-400 py-1 text-right text-xs font-extrabold text-gray-950 shadow outline-none ring-1 ring-gray-950/20 hover:bg-accent-500"
		style="margin-left: {Math.round((spanOffset / runLength) * 100)}%; width:{Math.round(
			(spanLength / runLength) * 100
		)}%;"
	>
		<span class="px-1">{renderDuration(spanLength)}</span>
	</button>
</div>
{#each childSpans as span}
	<svelte:self {span} {runStart} {runEnd} />
{/each}

<script lang="ts">
	import { differenceInMilliseconds } from 'date-fns';
	import { type SpanEntry, type TimeRange, isSpan } from '../../log';

	/**
	 * A Span or TaskSpan to show the duration of
	 */
	export let span: SpanEntry;
	/**
	 * The duration of the entire run of the logger
	 */
	export let range: TimeRange;

	$: spanStart = new Date(span.start_timestamp);
	$: spanOffset = differenceInMilliseconds(spanStart, range.from);
	$: spanLength = differenceInMilliseconds(new Date(span.end_timestamp), spanStart);
	$: runLength = differenceInMilliseconds(range.to, range.from);

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
<div class="w-full pb-1">
	<button
		class="bg-accent-400 py-0.5 text-right text-xs font-extrabold text-gray-950 shadow outline-none ring-1 ring-gray-950/20 hover:bg-accent-500"
		style="margin-left: {Math.round((spanOffset / runLength) * 100)}%; width:{Math.round(
			(spanLength / runLength) * 100
		)}%;"
	>
		<span class="px-1">{renderDuration(spanLength)}</span>
	</button>
</div>

<script lang="ts">
	import type { SpanEntry } from '$lib/log';
	import JsonValue from './JsonValue.svelte';
	import LogEntries from './LogEntries.svelte';

	/**
	 * A Span or Task Span to be rendered
	 */
	export let span: SpanEntry;
</script>

<!--
    @component
    Renders the various parts of a Span, whether it is a specialized TaskSpan, or just a normal one.
-->
{#if 'input' in span}
	<JsonValue value={span.input} />
{/if}

{#if 'output' in span}
	<JsonValue value={span.output} />
{/if}

<LogEntries logs={span.logs} />

<div>
	<p>Start: {span.start_timestamp}</p>
	<p>End: {span.end_timestamp}</p>
</div>

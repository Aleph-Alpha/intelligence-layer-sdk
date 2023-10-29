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

<div>
	<p>
		<span class="inline-block w-12">Start:</span><span>{span.start_timestamp}</span>
	</p>
	<p>
		<span class="inline-block w-12">End:</span><span>{span.end_timestamp}</span>
	</p>
</div>

<LogEntries logs={span.logs} />

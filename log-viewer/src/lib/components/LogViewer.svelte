<script lang="ts">
	import { activeSpan } from '$lib/active';
	import Timeline from '$lib/components/timeline/Timeline.svelte';
	import type { DebugLog } from '$lib/log';

	/**
	 * The Debug Log you want to render
	 */
	export let log: DebugLog;
</script>

<!--
    @component
    Renders a DebugLog as a timeline, as well as allowing for specifying a specific span and viewing further details.
-->
<Timeline {log} />

{#if $activeSpan}
	{$activeSpan.name}
{/if}

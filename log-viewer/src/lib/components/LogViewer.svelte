<script lang="ts">
	import { activeSpan } from '$lib/active';
	import Timeline from '$lib/components/timeline/Timeline.svelte';
	import type { DebugLog } from '$lib/log';
	import SpanDetail from './span-detail/SpanDetail.svelte';

	/**
	 * The Debug Log you want to render
	 */
	export let log: DebugLog;
</script>

<!--
    @component
    Renders a DebugLog as a timeline, as well as allowing for specifying a specific span and viewing further details.
-->
<div class="grid max-h-[calc(100vh-48px)] shrink grow grid-rows-2 gap-2">
	<Timeline {log} />
	<SpanDetail span={$activeSpan} />
</div>

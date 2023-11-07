<script lang="ts">
	import { activeSpan } from '$lib/active';
	import Timeline from '$lib/components/timeline/Timeline.svelte';
	import type { Tracer } from '$lib/trace';
	import SpanDetail from './span-detail/SpanDetail.svelte';

	/**
	 * The Debug Log you want to render
	 */
	export let trace: Tracer;
</script>

<!--
    @component
    Renders a DebugLog as a timeline, as well as allowing for specifying a specific span and viewing further details.
-->
<div class="grid max-h-[calc(100vh-48px)] shrink grow grid-rows-2 gap-2">
	<Timeline {trace} />
	<SpanDetail span={$activeSpan} />
</div>

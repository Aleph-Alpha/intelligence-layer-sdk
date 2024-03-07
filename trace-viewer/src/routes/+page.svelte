<script lang="ts">
	import { activeTrace } from '$lib/active';
	import TracePicker from '$lib/components/TracePicker.svelte';
	import TraceViewer from '$lib/components/TraceViewer.svelte';
	import type { PageData } from './$types';
	export let data: PageData;
</script>

<div class="flex h-screen flex-col gap-2 p-2">
	<h1 class="font-extrabold">Aleph Alpha Intelligence Layer</h1>
	{#if $activeTrace}
		<TraceViewer trace={data.trace} />
	{:else}
		<TracePicker />
	{/if}
</div>

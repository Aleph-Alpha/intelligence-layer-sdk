<script lang="ts">
	import { activeTrace } from '$lib/active';
	import TracePicker from '$lib/components/TracePicker.svelte';
	import TraceViewer from '$lib/components/TraceViewer.svelte';
</script>

<div class="flex h-screen flex-col gap-2 p-2">
	<h1 class="font-extrabold">Aleph Alpha Intelligence Layer</h1>
	{#if $activeTrace}
		<TraceViewer trace={$activeTrace} />
	{:else}
		<TracePicker />
	{/if}
</div>

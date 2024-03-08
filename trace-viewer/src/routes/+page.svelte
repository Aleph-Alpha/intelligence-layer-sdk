<script lang="ts">
	import { activeTrace } from '$lib/active';
	import TracePicker from '$lib/components/TracePicker.svelte';
	import TraceViewer from '$lib/components/TraceViewer.svelte';
	import { set, get, clear } from '$lib/db';
	import type { PageData } from './$types';
	console.log('active trace in page', $activeTrace);
	console.log('get in page', get());
	export let data: PageData;

	activeTrace.set(data.trace);
</script>

<div class="flex h-screen flex-col gap-2 p-2">
	<h1 class="font-extrabold">Aleph Alpha Intelligence Layer</h1>
	{#if $activeTrace != null}
		<TraceViewer trace={$activeTrace} />
		<form method="POST">
			<button
				class="rounded bg-blue-500 px-4 py-2 font-bold text-white hover:bg-blue-700"
				on:click={() => activeTrace.set(null)}>Upload New Trace</button
			>
		</form>
	{:else}
		<TracePicker />
	{/if}
</div>

<script lang="ts">
	import { enhance } from '$app/forms';
	import { activeTrace } from '$lib/active';
	import TracePicker from '$lib/components/TracePicker.svelte';
	import TraceViewer from '$lib/components/TraceViewer.svelte';
	import type { PageData } from './$types';

	export let data: PageData;

	activeTrace.set(data.trace);
</script>

<div class="flex h-screen flex-col gap-2 p-2">
	<h1 class="font-extrabold">Aleph Alpha Intelligence Layer</h1>
	{#if $activeTrace != null}
		<TraceViewer trace={$activeTrace} />
		<form
			method="POST"
			action="?/clearTrace"
			use:enhance={() => {
				return ({ result }) => {
					if (result.type === 'success') {
						activeTrace.set(null);
					}
				};
			}}
		>
			<button class="rounded bg-blue-500 px-4 py-2 font-bold text-white hover:bg-blue-700"
				>Upload New Trace</button
			>
		</form>
	{:else}
		<TracePicker submitAction="?/setTrace" />
	{/if}
</div>

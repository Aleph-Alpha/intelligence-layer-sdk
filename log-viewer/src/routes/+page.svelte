<script lang="ts">
	import { activeLog } from '$lib/active';
	import LogPicker from '$lib/components/LogPicker.svelte';
	import LogViewer from '$lib/components/LogViewer.svelte';
</script>

<div class="flex h-screen flex-col gap-2 p-2">
	<h1 class="font-extrabold">Aleph Alpha Intelligence Layer</h1>
	{#if $activeLog}
		<LogViewer log={$activeLog} />
	{:else}
		<LogPicker />
	{/if}
</div>

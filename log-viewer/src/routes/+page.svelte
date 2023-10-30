<script lang="ts">
	import { activeLog } from '$lib/active';
	import LogPicker from '$lib/components/LogPicker.svelte';
	import LogViewer from '$lib/components/LogViewer.svelte';
</script>

<h1>Aleph Alpha Intelligence Layer</h1>

{#if $activeLog}
	<LogViewer log={$activeLog} />
{:else}
	<LogPicker />
{/if}

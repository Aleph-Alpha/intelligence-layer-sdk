<script lang="ts">
	import { activeLog } from '$lib/active';

	// Reset on refresh
	let value = '';
</script>

<!--
    @component
    Allows for the user to choose a new DebugLog to preview
-->
<div class="grid h-screen place-content-center">
	<label for="debug-log" class="mb-2 font-extrabold text-gray-950"
		>Upload a debug log to render
	</label>
	<textarea
		class="appearance-none border-0 bg-white font-mono font-medium text-gray-950 shadow outline-none ring-1 ring-gray-950/20 placeholder:text-gray-400 focus:border-accent focus:ring-gray-950"
		id="debug-log"
		bind:value
		on:change={(e) => {
			activeLog.set(JSON.parse(e.currentTarget.value));
		}}
	/>
</div>

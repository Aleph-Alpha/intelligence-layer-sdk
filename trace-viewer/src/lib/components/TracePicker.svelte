<script lang="ts">
	import { activeTrace } from '$lib/active';
	import { tracer } from '$lib/trace';
	import { parseTraceFile } from '$lib/tracefile.parser';

	let files: FileList;
	// Reset on refresh
	let value = '';
</script>

<!--
    @component
    Allows for the user to choose a new Trace to preview
-->
<div class="grid grow place-content-center">
	<label for="trace" class="mb-2 font-extrabold text-gray-950">Upload a trace to render </label>
	<textarea
		class="appearance-none border-0 bg-white font-mono font-medium text-gray-950 shadow outline-none ring-1 ring-gray-950/20 placeholder:text-gray-400 focus:border-accent focus:ring-gray-950"
		id="trace"
		bind:value
		on:change={(e) => {
			activeTrace.set(tracer.parse(JSON.parse(e.currentTarget.value)));
		}}
	/>
	<input
		type="file"
		accept=".json, .jsonl, .log, .txt"
		bind:files
		on:change={async (file) => {
			const firstFile = file.currentTarget?.files?.item(0);
			firstFile && activeTrace.set(await parseTraceFile(firstFile));
		}}
	/>
</div>

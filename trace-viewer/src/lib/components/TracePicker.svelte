<script lang="ts">
	import { activeTrace } from '$lib/active';
	import { tracer, type Tracer } from '$lib/trace';
	import { parseTraceFile } from '$lib/tracefile.parser';
	import { enhance } from '$app/forms';

	let files: FileList;
	// Reset on refresh
	let value = '';
	let trace: Tracer;
	export let submitAction = '?/setTrace';
</script>

<!--
    @component
    Allows for the user to choose a new Trace to preview
-->
<div class="grid grow place-content-center">
	<div class="grid grid-cols-1 gap-3">
		<h2 class="text-xl font-extrabold">Upload a Tracer</h2>
		<label for="trace-json" class="font-extrabold text-gray-950"
			>JSON output from InMemoryTracer
		</label>
		<textarea
			class="appearance-none border-0 bg-white font-mono font-medium text-gray-950 shadow outline-none ring-1 ring-gray-950/20 placeholder:text-gray-400 focus:border-accent focus:ring-gray-950"
			id="trace-json"
			name="trace-file"
			bind:value
			on:change={(e) => {
				const newTrace = tracer.parse(JSON.parse(e.currentTarget.value));
				trace = newTrace ?? trace;
			}}
		/>

		<label for="trace-file" class="font-extrabold text-gray-950">Log output from FileTracer</label>
		<input
			id="trace-file"
			name="trace-file"
			type="file"
			accept=".json, .jsonl, .log, .txt"
			bind:files
			on:change={async (file) => {
				const firstFile = file.currentTarget?.files?.item(0);
				if (firstFile) {
					trace = await parseTraceFile(firstFile);
				}
			}}
		/>

		<form
			method="POST"
			action={submitAction}
			use:enhance={() => {
				return ({ result }) => {
					if (result.type === 'success') {
						activeTrace.set(trace);
					}
				};
			}}
		>
			<input type="hidden" name="trace" value={JSON.stringify(trace)} required />
			<button class="rounded bg-blue-500 px-4 py-2 font-bold text-white hover:bg-blue-700"
				>Submit</button
			>
			<!--on:click={() =>activeTrace.set(trace)}-->
		</form>
	</div>
</div>

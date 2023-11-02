<script lang="ts">
	import { activeLog } from '$lib/active';
	import { parseLogFile } from '$lib/logfile.parser';

	let files: FileList;
	// Reset on refresh
	let value = '';
</script>

<!--
    @component
    Allows for the user to choose a new DebugLog to preview
-->
<div class="grid grow place-content-center">
	<label for="debug-log" class="mb-2 font-extrabold text-gray-950"
		>Upload a debug log to render
	</label>
	<textarea
		class="focus:border-accent appearance-none border-0 bg-white font-mono font-medium text-gray-950 shadow outline-none ring-1 ring-gray-950/20 placeholder:text-gray-400 focus:ring-gray-950"
		id="debug-log"
		bind:value
		on:change={(e) => {
			// TODO: do zod validation
			// eslint-disable-next-line @typescript-eslint/no-unsafe-argument
			activeLog.set(JSON.parse(e.currentTarget.value));
		}}
	/>
	<input
		type="file"
		accept=".json, .jsonl, .log, .txt"
		bind:files
		on:change={async (file) => {
			const firstFile = file.currentTarget?.files?.item(0);
			activeLog.set(firstFile ? await parseLogFile(firstFile) : { name: '', logs: [] });
		}}
	/>
</div>

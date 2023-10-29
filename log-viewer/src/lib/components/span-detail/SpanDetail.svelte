<script lang="ts">
	import { createTabs, melt } from '@melt-ui/svelte';
	import { cubicInOut } from 'svelte/easing';
	import { crossfade } from 'svelte/transition';
	import type { SpanEntry } from '$lib/log';
	import JsonValue from './JsonValue.svelte';
	import LogEntries from './LogEntries.svelte';

	/**
	 * A Span or Task Span to be rendered
	 */
	export let span: SpanEntry | null;

	const {
		elements: { root, list, content, trigger },
		states: { value }
	} = createTabs({
		defaultValue: 'tab-1'
	});

	const triggers = [
		{ id: 'tab-1', title: 'Logs' },
		{ id: 'tab-2', title: 'Input' },
		{ id: 'tab-3', title: 'Output' }
	];

	const [send, receive] = crossfade({
		duration: 250,
		easing: cubicInOut
	});
</script>

<!--
    @component
    Renders the various parts of a Span, whether it is a specialized TaskSpan, or just a normal one.
-->
<div
	use:melt={$root}
	class="flex h-full flex-col overflow-hidden border data-[orientation=vertical]:flex-row"
>
	<div
		use:melt={$list}
		class="flex shrink-0 overflow-x-auto bg-gray-100
  data-[orientation=vertical]:flex-col data-[orientation=vertical]:border-r"
		aria-label="Span Detail"
	>
		{#each triggers as triggerItem}
			<button
				use:melt={$trigger(triggerItem.id)}
				class="relative flex h-10 flex-auto cursor-default select-none items-center justify-center rounded-none bg-gray-100 px-2 text-sm leading-none text-gray-900 focus-visible:z-10 focus-visible:ring-2 data-[state='active']:bg-white"
			>
				{triggerItem.title}
				{#if $value === triggerItem.id}
					<div
						in:send={{ key: 'trigger' }}
						out:receive={{ key: 'trigger' }}
						class="absolute bottom-1 left-1/2 h-1 w-14 -translate-x-1/2 rounded-full bg-accent-400"
					/>
				{/if}
			</button>
		{/each}
	</div>

	{#if span}
		<div use:melt={$content('tab-1')} class="grow bg-white p-5">
			<div class="grid grow grid-flow-row gap-2">
				<div>
					<p>
						<span class="inline-block w-12 font-extrabold">Start:</span><span
							>{span.start_timestamp}</span
						>
					</p>
					<p>
						<span class="inline-block w-12 font-extrabold">End:</span><span
							>{span.end_timestamp}</span
						>
					</p>
				</div>

				<LogEntries logs={span.logs} />
			</div>
		</div>

		<div use:melt={$content('tab-2')} class="grow bg-white p-5">
			{#if 'input' in span}
				<JsonValue value={span.input} />
			{:else}
				<p>No input for this span</p>
			{/if}
		</div>

		<div use:melt={$content('tab-3')} class="grow bg-white p-5">
			{#if 'output' in span}
				<JsonValue value={span.output} />
			{:else}
				<p>No output for this span</p>
			{/if}
		</div>
	{:else}
		<p class="p-4">Select a span to see further details.</p>
	{/if}
</div>

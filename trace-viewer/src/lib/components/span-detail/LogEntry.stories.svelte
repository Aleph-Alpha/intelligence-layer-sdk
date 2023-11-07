<script context="module" lang="ts">
	import type { MetaProps } from '@storybook/addon-svelte-csf';
	import { randomLogEntry } from '../../trace.test_utils';
	import LogEntry from './LogEntry.svelte';

	export const meta: MetaProps = {
		title: 'SpanDetail/Molecules/LogEntry',
		component: LogEntry,
		tags: ['autodocs'],
		args: {
			log: randomLogEntry()
		}
	};
</script>

<script lang="ts">
	import { Story, Template } from '@storybook/addon-svelte-csf';
</script>

<Template let:args>
	<LogEntry {...args} />
</Template>

<Story name="Base" />

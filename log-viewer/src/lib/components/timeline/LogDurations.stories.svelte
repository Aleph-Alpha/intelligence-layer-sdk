<script context="module" lang="ts">
	import type { MetaProps } from '@storybook/addon-svelte-csf';
	import { randomLogger } from '../../log.test_utils';
	import LogDurations from './LogDurations.svelte';

	const logger = randomLogger();

	export const meta: MetaProps = {
		title: 'Timeline/Molecules/LogDurations',
		component: LogDurations,
		tags: ['autodocs'],
		args: {
			logs: logger.logs
		}
	};
</script>

<script lang="ts">
	import { Story, Template } from '@storybook/addon-svelte-csf';
</script>

<Template let:args>
	<LogDurations {...args} />
</Template>

<Story name="Base" />

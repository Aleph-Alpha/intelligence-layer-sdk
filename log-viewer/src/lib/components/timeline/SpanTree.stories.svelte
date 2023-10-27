<script context="module" lang="ts">
	import type { MetaProps } from '@storybook/addon-svelte-csf';
	import { randomLogger } from '../../log.test_utils';
	import { logRange } from '../../log';
	import SpanTree from './SpanTree.svelte';

	const logger = randomLogger();
	const range = logRange(logger);

	export const meta: MetaProps = {
		title: 'Timeline/Molecules/SpanTree',
		component: SpanTree,
		tags: ['autodocs'],
		args: {
			logs: logger.logs,
			range
		}
	};
</script>

<script lang="ts">
	import { Story, Template } from '@storybook/addon-svelte-csf';
</script>

<Template let:args>
	<SpanTree {...args} />
</Template>

<Story name="Base" />

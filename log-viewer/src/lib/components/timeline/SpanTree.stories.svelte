<script context="module" lang="ts">
	import type { MetaProps } from '@storybook/addon-svelte-csf';
	import type { Entry } from '../../log';
	import { randomSpan } from '../../log.test_utils';
	import SpanTree from './SpanTree.svelte';

	const logs: Entry[] = [randomSpan()];

	export const meta: MetaProps = {
		title: 'Timeline/Molecules/SpanTree',
		component: SpanTree,
		tags: ['autodocs'],
		args: {
			logs
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

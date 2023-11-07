<script context="module" lang="ts">
	import type { MetaProps } from '@storybook/addon-svelte-csf';
	import { randomTracer } from '../../trace.test_utils';
	import Timeline from './Timeline.svelte';

	export const meta: MetaProps = {
		title: 'Timeline/Organisms/Timeline',
		component: Timeline,
		tags: ['autodocs'],
		args: {
			trace: randomTracer()
		}
	};
</script>

<script lang="ts">
	import { Story, Template } from '@storybook/addon-svelte-csf';
</script>

<Template let:args>
	<Timeline {...args} />
</Template>

<Story name="Base" />

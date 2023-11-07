<script context="module" lang="ts">
	import type { MetaProps } from '@storybook/addon-svelte-csf';
	import { randomTracer } from '../trace.test_utils';
	import TraceViewer from './TraceViewer.svelte';

	export const meta: MetaProps = {
		title: 'Pages/TraceViewer',
		component: TraceViewer,
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
	<TraceViewer {...args} />
</Template>

<Story name="Base" />

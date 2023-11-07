<script context="module" lang="ts">
	import type { MetaProps } from '@storybook/addon-svelte-csf';
	import { randomTracer } from '../../trace.test_utils';
	import { traceRange } from '../../trace';
	import SpanTree from './SpanTree.svelte';

	const tracer = randomTracer();
	const range = traceRange(tracer);

	export const meta: MetaProps = {
		title: 'Timeline/Molecules/SpanTree',
		component: SpanTree,
		tags: ['autodocs'],
		args: {
			entries: tracer.entries,
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

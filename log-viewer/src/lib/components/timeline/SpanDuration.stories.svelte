<script context="module" lang="ts">
	import type { MetaProps } from '@storybook/addon-svelte-csf';
	import { randomSpan, randomDateRange } from '../../log.test_utils';
	import SpanDuration from './SpanDuration.svelte';

	const range = randomDateRange();
	const span = randomSpan(range);

	export const meta: MetaProps = {
		title: 'Timeline/Atoms/SpanDuration',
		component: SpanDuration,
		tags: ['autodocs'],
		args: {
			span,
			range
		}
	};
</script>

<script lang="ts">
	import { Story, Template } from '@storybook/addon-svelte-csf';
</script>

<Template let:args>
	<SpanDuration {...args} />
</Template>

<Story name="Base" />

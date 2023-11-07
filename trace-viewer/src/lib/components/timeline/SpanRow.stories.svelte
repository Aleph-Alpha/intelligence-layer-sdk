<script context="module" lang="ts">
	import type { MetaProps } from '@storybook/addon-svelte-csf';
	import { randomDateRange, randomSpan } from '../../trace.test_utils';
	import SpanRow from './SpanRow.svelte';

	const range = randomDateRange();
	const span = randomSpan(range);

	export const meta: MetaProps = {
		title: 'Timeline/Molecules/SpanRow',
		component: SpanRow,
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
	<SpanRow {...args} />
</Template>

<Story name="Base" />

<script context="module" lang="ts">
	import type { MetaProps } from '@storybook/addon-svelte-csf';
	import { randomSpan, randomTaskSpan } from '../../trace.test_utils';
	import SpanDetail from './SpanDetail.svelte';

	export const meta: MetaProps = {
		title: 'SpanDetail/Templates/SpanDetail',
		component: SpanDetail,
		tags: ['autodocs'],
		args: {
			span: randomTaskSpan()
		}
	};
</script>

<script lang="ts">
	import { Story, Template } from '@storybook/addon-svelte-csf';
</script>

<Template let:args>
	<SpanDetail {...args} />
</Template>

<Story name="TaskSpan" />
<Story name="Span" args={{ span: randomSpan() }} />

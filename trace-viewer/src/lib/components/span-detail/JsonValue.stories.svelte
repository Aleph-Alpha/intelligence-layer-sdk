<script context="module" lang="ts">
	import type { MetaProps } from '@storybook/addon-svelte-csf';
	import { randomValue } from '../../trace.test_utils';
	import JsonValue from './JsonValue.svelte';

	export const meta: MetaProps = {
		title: 'SpanDetail/Atoms/JsonValue',
		component: JsonValue,
		tags: ['autodocs'],
		args: {
			value: randomValue()
		}
	};
</script>

<script lang="ts">
	import { Story, Template } from '@storybook/addon-svelte-csf';
</script>

<Template let:args>
	<JsonValue {...args} />
</Template>

<Story name="Base" />

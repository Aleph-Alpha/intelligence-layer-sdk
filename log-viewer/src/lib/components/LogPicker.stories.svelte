<script context="module" lang="ts">
	import type { MetaProps } from '@storybook/addon-svelte-csf';
	import LogPicker from './LogPicker.svelte';

	export const meta: MetaProps = {
		title: 'Pages/LogPicker',
		component: LogPicker,
		tags: ['autodocs']
	};
</script>

<script lang="ts">
	import { Story, Template } from '@storybook/addon-svelte-csf';
</script>

<Template let:args>
	<LogPicker {...args} />
</Template>

<Story name="Base" />

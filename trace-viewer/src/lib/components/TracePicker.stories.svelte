<script context="module" lang="ts">
	import type { MetaProps } from '@storybook/addon-svelte-csf';
	import TracePicker from './TracePicker.svelte';

	export const meta: MetaProps = {
		title: 'Pages/TracePicker',
		component: TracePicker,
		tags: ['autodocs']
	};
</script>

<script lang="ts">
	import { Story, Template } from '@storybook/addon-svelte-csf';
</script>

<Template let:args>
	<TracePicker {...args} />
</Template>

<Story name="Base" />

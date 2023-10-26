<script context="module">
	import Timeline from './Timeline.svelte';

	export const meta = {
		title: 'Timeline/Organisms/Timeline',
		component: Timeline,
		tags: ['autodocs']
	};
</script>

<script lang="ts">
	import { Story, Template } from '@storybook/addon-svelte-csf';
</script>

<Template let:args>
	<Timeline {...args} />
</Template>

<Story name="Base" />

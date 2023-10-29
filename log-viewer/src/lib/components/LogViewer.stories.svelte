<script context="module" lang="ts">
	import type { MetaProps } from '@storybook/addon-svelte-csf';
	import { randomLogger } from '../log.test_utils';
	import LogViewer from './LogViewer.svelte';
	import log from '../qa_example.json';

	export const meta: MetaProps = {
		title: 'Pages/LogViewer',
		component: LogViewer,
		tags: ['autodocs'],
		args: {
			log //: randomLogger()
		}
	};
</script>

<script lang="ts">
	import { Story, Template } from '@storybook/addon-svelte-csf';
</script>

<Template let:args>
	<LogViewer {...args} />
</Template>

<Story name="Base" />

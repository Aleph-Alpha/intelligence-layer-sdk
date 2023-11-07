<script context="module" lang="ts">
	import { faker } from '@faker-js/faker';
	import type { MetaProps } from '@storybook/addon-svelte-csf';
	import { randomLogEntry } from '../../trace.test_utils';
	import LogEntries from './LogEntries.svelte';

	export const meta: MetaProps = {
		title: 'SpanDetail/Organisms/LogEntries',
		component: LogEntries,
		tags: ['autodocs'],
		args: {
			entries: faker.helpers.multiple(() => randomLogEntry())
		}
	};
</script>

<script lang="ts">
	import { Story, Template } from '@storybook/addon-svelte-csf';
</script>

<Template let:args>
	<LogEntries {...args} />
</Template>

<Story name="Base" />

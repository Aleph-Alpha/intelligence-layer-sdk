<script lang="ts">
	import type { JSONValue } from '$lib/log';

	/**
	 * Any value that can be passed to JSON.stringify
	 */
	export let value: JSONValue;
</script>

<!--
    @component
    Renders any JSON Value for easier readability.
-->
<pre class="whitespace-pre-wrap whitespace-break-spaces border px-2 py-1.5 text-sm">{JSON.stringify(
		value,
		undefined,
		2
	)}</pre>

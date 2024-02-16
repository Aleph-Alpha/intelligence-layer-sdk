<script lang="ts">
	import { json } from '@codemirror/lang-json';
	import CodeMirror from 'svelte-codemirror-editor';
	import { EditorView } from 'codemirror';
	import type { JSONValue } from '$lib/trace';

	/**
	 * Any value that can be passed to JSON.stringify
	 */
	export let value: JSONValue;
</script>

<!--
    @component
    Renders any JSON Value for easier readability.
-->
<CodeMirror
	value={JSON.stringify(value, undefined, 2)}
	extensions={[EditorView.lineWrapping]}
	lang={json()}
	editable={false}
	readonly={true}
/>

<script lang="ts">
	import { env } from '$env/dynamic/public';
	import '../app.css';
</script>

<svelte:head>
	{#if !env.PUBLIC_DISABLE_ANALYTICS || env.PUBLIC_DISABLE_ANALYTICS !== 'false'}
		<script
			src="https://cdn-eu.usefathom.com/script.js"
			data-spa="auto"
			data-site="GLRCMWLG"
			defer
		></script>
	{/if}
</svelte:head>

<slot />

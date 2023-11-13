<script lang="ts">
	import { PUBLIC_DISABLE_ANALYTICS } from '$env/static/public';
	import '../app.css';
</script>

<svelte:head>
	{#if !PUBLIC_DISABLE_ANALYTICS || PUBLIC_DISABLE_ANALYTICS !== "false" }
		<!-- Fathom - beautiful, simple website analytics -->
		<script
			src="https://cdn-eu.usefathom.com/script.js"
			data-spa="auto"
			data-site="GLRCMWLG"
			defer
		></script>
		<!-- / Fathom -->
	{/if}
</svelte:head>

<slot />

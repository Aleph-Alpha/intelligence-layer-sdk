import { expect, test } from '@playwright/test';
import { randomTracer } from '../src/lib/trace.test_utils';
import { isLogEntry, isSpan } from '../src/lib/trace';

test('user can paste a new trace', async ({ page }) => {
	await page.goto('/');
	await expect(page.getByRole('heading', { name: 'Aleph Alpha Intelligence Layer' })).toBeVisible();

	const trace = randomTracer();

	const textbox = page.getByLabel('Upload a trace to render');
	await textbox.fill(JSON.stringify(trace));
	await textbox.blur();

	for (const span of trace.entries.filter(isSpan)) {
		await page.getByText(span.name).click();

		await page.getByText('Logs').click();
		await expect(page.getByText(span.start_timestamp)).toBeVisible();
		await expect(page.getByText(span.end_timestamp)).toBeVisible();

		for (const log of span.entries.filter(isLogEntry)) {
			await expect(page.getByText(log.message)).toBeVisible();
		}

		if ('input' in span && 'output' in span) {
			await page.getByText('Input').click();
			await expect(page.getByText('No input for this span')).not.toBeVisible();

			await page.getByText('Output').click();
			await expect(page.getByText('No output for this span')).not.toBeVisible();
		}
	}
});

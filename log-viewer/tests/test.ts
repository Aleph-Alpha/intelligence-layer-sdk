import { expect, test } from '@playwright/test';
import { randomLogger } from '../src/lib/log.test_utils';
import { isLogEntry, isSpan } from '../src/lib/log';

test('user can paste a new log', async ({ page }) => {
	await page.goto('/');
	await expect(page.getByRole('heading', { name: 'Aleph Alpha Intelligence Layer' })).toBeVisible();

	const log = randomLogger();

	const textbox = page.getByLabel('Upload a debug log to render');
	await textbox.fill(JSON.stringify(log));
	await textbox.blur();

	await expect(page.getByText(log.name)).toBeVisible();

	for (const span of log.logs.filter(isSpan)) {
		await page.getByText(span.name).click();

		await page.getByText('Logs').click();
		await expect(page.getByText(span.start_timestamp)).toBeVisible();
		await expect(page.getByText(span.end_timestamp)).toBeVisible();

		for (const log of span.logs.filter(isLogEntry)) {
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

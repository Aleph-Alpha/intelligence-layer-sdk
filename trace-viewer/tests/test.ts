import { expect, test, type Page } from '@playwright/test';
import { randomTracer } from '../src/lib/trace.test_utils';
import { isLogEntry, isSpan, type Tracer } from '../src/lib/trace';

// Extend base test by providing "todoPage" and "settingsPage".
// This new "test" can be used in multiple test files, and each of them will get the fixtures.
// Declare the types of your fixtures.
// type TraceFixture = {
// 	trace: Tracer;
// };

const TRACE_UPLOAD_URL = 'http://localhost:4173/trace';

// Declare the types of your fixtures.
interface TraceFixture {
	randomTrace: Tracer;
}

export const testWithCleanup = test.extend<TraceFixture>({
	randomTrace: async ({ page }, use) => {
		// Set up the fixture.
		const trace = randomTracer();

		// Use the fixture value in the test.
		await page.goto('/');

		await use(trace);

		await fetch(TRACE_UPLOAD_URL, { method: 'DELETE' });
	}
});

testWithCleanup('user can paste a new trace and it persists', async ({ page, randomTrace }) => {
	await submitTrace(page, randomTrace);
	await checkCorrectTraceDisplay(randomTrace, page);

	await page.reload();
	await checkCorrectTraceDisplay(randomTrace, page);
});

testWithCleanup(
	'user can submit and delete a trace and it remains deleted',
	async ({ page, randomTrace }) => {
		await submitTrace(page, randomTrace);
		await page.getByText('Upload New Trace').click();
		await expect(page.getByLabel('JSON output from InMemoryTracer')).toBeEmpty();

		await page.reload();
		await expect(page.getByLabel('JSON output from InMemoryTracer')).toBeEmpty();
	}
);

testWithCleanup('can load externally posted trace', async ({ page, randomTrace }) => {
	//load empty page, post trace, reload -> trace should be there
	const textbox = page.getByLabel('JSON output from InMemoryTracer');
	await expect(textbox).toHaveText('');

	await fetch(TRACE_UPLOAD_URL, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json'
		},
		body: JSON.stringify(randomTrace)
	});

	await page.reload();
	await checkCorrectTraceDisplay(randomTrace, page);
});

async function checkCorrectTraceDisplay(randomTrace: Tracer, page: Page) {
	for (const span of randomTrace.entries.filter(isSpan)) {
		await page.getByText(span.name).click();

		await page.getByText('Logs').click();
		await expect(page.getByText(span.start_timestamp)).toBeVisible();
		await expect(page.getByText(span.end_timestamp)).toBeVisible();

		for (const log of span.entries.filter(isLogEntry)) {
			await expect(page.getByText(log.message)).toBeVisible();
		}

		if ('input' in span && 'output' in span) {
			await page.getByRole('tab', { name: 'Input' }).click();
			await expect(page.getByText('No input for this span')).not.toBeVisible();

			await page.getByRole('tab', { name: 'Output' }).click();
			await expect(page.getByText('No output for this span')).not.toBeVisible();
		}
	}
}

async function submitTrace(page: Page, randomTrace: Tracer) {
	await expect(page.getByRole('heading', { name: 'Aleph Alpha Intelligence Layer' })).toBeVisible();
	const textbox = page.getByLabel('JSON output from InMemoryTracer');
	await textbox.fill(JSON.stringify(randomTrace));
	await textbox.blur();
	await page.getByText('Submit').click();
}

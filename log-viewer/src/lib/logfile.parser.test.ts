import { expect, test } from 'vitest';
import { parseLogLines } from './logfile.parser';
import { randomLogEntry, randomSpan } from './log.test_utils';
import { faker } from '@faker-js/faker';

test('parseLogLines can parse start task entries', () => {
	const rootId = faker.string.uuid();
	const span = randomSpan();
	const spanId = faker.string.uuid();
	const entry = randomLogEntry();
	const entries = [
		{
			entry_type: 'SpanStart' as const,
			entry: { parent: rootId, uuid: spanId, name: span.name, start: span.start_timestamp }
		},
		{
			entry_type: 'LogEntry' as const,
			entry: {
				parent: spanId,
				message: entry.message,
				value: entry.value,
				timestamp: entry.timestamp
			}
		},
		{ entry_type: 'SpanEnd' as const, entry: { uuid: spanId, end: span.end_timestamp } }
	];
	const debugLogger = parseLogLines(entries);

	expect(debugLogger).toStrictEqual({
		name: rootId,
		logs: [
			{
				name: span.name,
				start_timestamp: span.start_timestamp,
				end_timestamp: span.end_timestamp,
				logs: [entry]
			}
		]
	});
});

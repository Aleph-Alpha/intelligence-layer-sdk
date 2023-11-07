import { expect, test } from 'vitest';
import { type LogLine, parseTraceFile } from './tracefile.parser';
import { randomTracer } from './trace.test_utils';
import { faker } from '@faker-js/faker';
import {
	isLogEntry,
	type Entry,
	type TaskSpan,
	type Tracer,
	type LogEntry,
	type Span
} from './trace';

test('parseLogFile can parse JSON entries from a file', async () => {
	const expected = randomTracer();

	const actual = await parseTraceFile(toFile(expected));

	expect(actual).toStrictEqual(expected);
});

function toFile(expected: Tracer) {
	const entries = toFlatEntries('test', expected.entries);
	const file = new File(
		entries.map((entry) => JSON.stringify(entry) + '\n'),
		'ignored.txt'
	);
	return file;
}

function toFlatLogEntry(parent: string, entry: LogEntry): LogLine {
	return {
		entry_type: 'PlainEntry',
		entry: { parent, message: entry.message, timestamp: entry.timestamp, value: entry.value }
	};
}

function toFlatEntries(parent: string, entries: Entry[]): LogLine[] {
	return entries.flatMap((entry) => {
		if (isLogEntry(entry)) {
			return [toFlatLogEntry(parent, entry)];
		} else if (isTaskSpan(entry)) {
			return toFlatTaskSpanEntries(parent, entry);
		} else {
			return toFlatSpanEntries(parent, entry);
		}
	});
}

function toFlatSpanEntries(parent: string, entry: Span): LogLine[] {
	const uuid = faker.string.uuid();
	const entries: LogLine[] = [
		{
			entry_type: 'StartSpan',
			entry: { parent, name: entry.name, start: entry.start_timestamp, uuid }
		}
	];
	entries.push(...toFlatEntries(uuid, entry.entries));
	entries.push({ entry_type: 'EndSpan', entry: { end: entry.end_timestamp, uuid } });
	return entries;
}

function toFlatTaskSpanEntries(parent: string, entry: TaskSpan): LogLine[] {
	const uuid = faker.string.uuid();
	const entries: LogLine[] = [
		{
			entry_type: 'StartTask',
			entry: {
				parent,
				name: entry.name,
				start: entry.start_timestamp,
				uuid,
				input: entry.input
			}
		}
	];
	entries.push(...toFlatEntries(uuid, entry.entries));
	entries.push({
		entry_type: 'EndTask',
		entry: { end: entry.end_timestamp, uuid, output: entry.output }
	});
	return entries;
}

function isTaskSpan(entry: Entry): entry is TaskSpan {
	return 'input' in entry;
}

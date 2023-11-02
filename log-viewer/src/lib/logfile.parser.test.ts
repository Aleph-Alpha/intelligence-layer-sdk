import { expect, test } from 'vitest';
import { type LogLine, parseLogFile } from './logfile.parser';
import { randomLogger } from './log.test_utils';
import { faker } from '@faker-js/faker';
import {
	isLogEntry,
	type Entry,
	type TaskSpan,
	type DebugLog,
	type LogEntry,
	type Span
} from './log';

test('parseLogFile can parse JSON debug-log entries from a file', async () => {
	const expected = randomLogger();

	const actual = await parseLogFile(toFile(expected));

	expect(actual).toStrictEqual(expected);
});

function toFile(expected: DebugLog) {
	const entries = toFlatEntries(expected.name, expected.logs);
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

function toFlatEntries(parent: string, logs: Entry[]): LogLine[] {
	return logs.flatMap((entry) => {
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
	const logs: LogLine[] = [
		{
			entry_type: 'StartSpan',
			entry: { parent, name: entry.name, start: entry.start_timestamp, uuid }
		}
	];
	logs.push(...toFlatEntries(uuid, entry.logs));
	logs.push({ entry_type: 'EndSpan', entry: { end: entry.end_timestamp, uuid } });
	return logs;
}

function toFlatTaskSpanEntries(parent: string, entry: TaskSpan): LogLine[] {
	const uuid = faker.string.uuid();
	const logs: LogLine[] = [
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
	logs.push(...toFlatEntries(uuid, entry.logs));
	logs.push({
		entry_type: 'EndTask',
		entry: { end: entry.end_timestamp, uuid, output: entry.output }
	});
	return logs;
}

function isTaskSpan(entry: Entry): entry is TaskSpan {
	return 'input' in entry;
}

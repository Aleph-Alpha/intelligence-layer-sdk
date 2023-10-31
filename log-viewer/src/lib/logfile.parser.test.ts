import { expect, test } from 'vitest';
import { parseLogLines, type LogLine } from './logfile.parser';
import { randomLogger } from './log.test_utils';
import { faker } from '@faker-js/faker';
import { isLogEntry, type Entry, type TaskSpan } from './log';

test('parseLogLines can parse start task entries', () => {
	const rootId = faker.string.uuid();
	const expected = randomLogger();
	expected.name = rootId;
	const actual = parseLogLines(toFlatEntries(rootId, expected.logs));

	expect(actual).toStrictEqual(expected);
});

function toFlatEntries(parent: string, logs: Entry[]): LogLine[] {
	return logs.flatMap((entry) => {
		if (isLogEntry(entry)) {
			return [
				{
					entry_type: 'LogEntry',
					entry: { parent, message: entry.message, timestamp: entry.timestamp, value: entry.value }
				}
			];
		} else if (isTaskSpan(entry)) {
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
		} else {
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
	});
}

function isTaskSpan(entry: Entry): entry is TaskSpan {
	return 'input' in entry;
}

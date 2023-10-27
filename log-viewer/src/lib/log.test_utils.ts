import { faker } from '@faker-js/faker';
import { compareAsc } from 'date-fns';
import type { Entry, LogEntry, Span, TaskSpan, JSONValue, DebugLog, TimeRange } from './log';

/**
 * Produces a random time range, with optional time bounds
 */
export function randomDateRange(between?: TimeRange): TimeRange {
	if (between) {
		const dates = faker.date.betweens({ ...between, count: 2 });
		dates.sort(compareAsc);
		const [start, end] = dates;
		return { from: start, to: end };
	}

	const end = faker.date.recent();
	return { from: faker.date.recent({ refDate: end }), to: end };
}

export function randomValue(): JSONValue {
	return faker.helpers.arrayElement([
		() => faker.word.sample(),
		() => faker.number.int(),
		() => faker.datatype.boolean(),
		() => null,
		() => faker.helpers.multiple(randomValue, { count: { max: 2, min: 0 } }),
		() =>
			faker.helpers
				.multiple(() => faker.word.sample(), { count: { max: 2, min: 0 } })
				.reduce((acc, key) => ({ ...acc, [key]: randomValue() }), {})
	])();
}

/**
 * Produce a random log entry, within a given time range
 */
export function randomLogEntry(between?: TimeRange): LogEntry {
	return {
		message: faker.lorem.sentence(),
		value: randomValue(),
		timestamp: (between ? faker.date.between(between) : faker.date.recent()).toISOString()
	};
}

/**
 * Produces a random span, as well as potentially child spans and entries
 */
export function randomSpan(between?: TimeRange): Span {
	const range = randomDateRange(between);
	return {
		name: faker.word.sample(),
		start_timestamp: range.from.toISOString(),
		end_timestamp: range.to.toISOString(),
		logs: faker.helpers.multiple(() => randomEntry(range), { count: { max: 2, min: 0 } })
	};
}

export function randomTaskSpan(between?: TimeRange): TaskSpan {
	return {
		...randomSpan(between),
		input: randomValue(),
		output: randomValue()
	};
}

/**
 * Return a random LogEntry, Span, or TaskSpan
 */
export function randomEntry(between?: TimeRange): Entry {
	return faker.helpers.arrayElement([
		() => randomLogEntry(between),
		() => randomSpan(between),
		() => randomTaskSpan(between)
	])();
}

export function randomLogger(): DebugLog {
	const range = randomDateRange();
	return {
		name: faker.word.sample(),
		logs: faker.helpers.multiple(() => randomEntry(range), { count: { max: 2, min: 1 } })
	};
}

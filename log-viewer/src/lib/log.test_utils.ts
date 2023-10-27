import { faker } from '@faker-js/faker';
import { compareAsc } from 'date-fns';
import type { Span } from './log';

interface TimeRange {
	start: Date;
	end: Date;
}

/**
 * Produces a random time range, with optional time bounds
 */
function randomDateRange(between?: TimeRange): TimeRange {
	if (between) {
		const dates = faker.date.betweens({ from: between.start, to: between.end, count: 2 });
		dates.sort(compareAsc);
		const [start, end] = dates;
		return { start, end };
	}

	const end = faker.date.recent();
	return { start: faker.date.recent({ refDate: end }), end };
}

/**
 * Produces a random span, as well as potentially child spans
 */
export function randomSpan(between?: TimeRange): Span {
	const range = randomDateRange(between);
	return {
		name: faker.word.sample(),
		start_timestamp: range.start.toISOString(),
		end_timestamp: range.end.toISOString(),
		logs: faker.helpers.multiple(() => randomSpan(range), { count: { max: 2, min: 0 } })
	};
}

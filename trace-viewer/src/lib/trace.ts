import { compareAsc } from 'date-fns';

export type JSONValue =
	| string
	| number
	| boolean
	| null
	| undefined
	| JSONValue[]
	| { [key: string]: JSONValue };

export interface LogEntry {
	message: string;
	value: JSONValue;
	timestamp: string;
}

export type Entry = LogEntry | Span | TaskSpan;
export type SpanEntry = Span | TaskSpan;

export interface Tracer {
	entries: Entry[];
}

export interface Span extends Tracer {
	name: string;
	start_timestamp: string;
	end_timestamp: string;
}

export interface TaskSpan extends Span {
	input: JSONValue;
	output: JSONValue;
}

export function isLogEntry(entry: Entry): entry is LogEntry {
	return 'message' in entry;
}

export function isSpan(entry: Entry): entry is SpanEntry {
	return 'entries' in entry;
}

export interface TimeRange {
	from: Date;
	to: Date;
}

/**
 * Calculate the first and last timestamp of a tracer
 */
export function traceRange(trace: Tracer): TimeRange | null {
	const entryTimes = trace.entries.reduce<Date[]>((acc, i) => {
		if ('timestamp' in i) acc.push(new Date(i.timestamp));
		if ('start_timestamp' in i) acc.push(new Date(i.start_timestamp));
		if ('end_timestamp' in i) acc.push(new Date(i.end_timestamp));
		return acc;
	}, []);
	entryTimes.sort(compareAsc);

	const from = entryTimes.at(0);
	const to = entryTimes.at(-1);
	return from && to ? { from, to } : null;
}

export function renderDuration(spanLength: number): string {
	let unit = 'ms';
	let length = spanLength;
	if (length >= 1000) {
		length /= 1000;
		unit = 's';
		if (length >= 60) {
			length /= 60;
			unit = 'min';
			if (length >= 60) {
				length /= 60;
				unit = 'h';
			}
		}
	}

	return `${length.toLocaleString('en-US')}${unit}`;
}

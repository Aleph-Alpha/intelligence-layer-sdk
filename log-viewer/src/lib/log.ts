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

export interface DebugLog {
	name: string;
	logs: Entry[];
}

export interface Span extends DebugLog {
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
	return 'logs' in entry;
}

export interface TimeRange {
	from: Date;
	to: Date;
}

/**
 * Calculate the first and last timestamp of a logger
 */
export function logRange(log: DebugLog): TimeRange | null {
	const logTimes = log.logs.reduce<Date[]>((acc, i) => {
		if ('timestamp' in i) acc.push(new Date(i.timestamp));
		if ('start_timestamp' in i) acc.push(new Date(i.start_timestamp));
		if ('end_timestamp' in i) acc.push(new Date(i.end_timestamp));
		return acc;
	}, []);
	logTimes.sort(compareAsc);

	const from = logTimes.at(0);
	const to = logTimes.at(-1);
	return from && to ? { from, to } : null;
}

export function renderDuration(spanLength: number): string {
	let unit = 'ms';
	let length = spanLength;
	if (length > 1000) {
		length /= 1000;
		unit = 's';
		if (length > 60) {
			length /= 60;
			unit = 'min';
			if (length > 60) {
				length /= 60;
				unit = 'h';
			}
		}
	}

	return `${length.toLocaleString('en-US')}${unit}`;
}

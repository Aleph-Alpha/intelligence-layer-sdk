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

export type Entry = LogEntry | Span | TraceSpan;
export type SpanEntry = Span | TraceSpan;

export interface DebugLog {
	name: string;
	logs: Entry[];
}

export interface Span extends DebugLog {
	start_timestamp: string;
	end_timestamp: string;
}

export interface TraceSpan extends Span {
	input: JSONValue;
	output: JSONValue;
}

export function isLogEntry(entry: Entry): entry is LogEntry {
	return 'message' in entry;
}

export function isSpan(entry: Entry): entry is SpanEntry {
	return 'logs' in entry;
}

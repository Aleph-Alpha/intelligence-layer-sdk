import { compareAsc } from 'date-fns';
import { z } from 'zod';

const literalSchema = z.union([z.string(), z.number(), z.boolean(), z.null()]);
type Literal = z.infer<typeof literalSchema>;
export type JSONValue = Literal | { [key: string]: JSONValue } | JSONValue[];
const jsonSchema: z.ZodType<JSONValue> = z.lazy(() =>
	z.union([literalSchema, z.array(jsonSchema), z.record(jsonSchema)])
);

const logEntry = z.object({
	message: z.string(),
	value: jsonSchema,
	timestamp: z.string(),
	trace_id: z.string()
});
export type LogEntry = z.infer<typeof logEntry>;

export type Entry = TaskSpan | Span | LogEntry;
export type SpanEntry = Span | TaskSpan;

const entry: z.ZodType<Entry> = z.lazy(() => z.union([taskSpan, logEntry, span]));

// eslint-disable-next-line @typescript-eslint/consistent-type-definitions
export type Tracer = {
	entries: Entry[];
};
export const tracer: z.ZodType<Tracer> = z.object({
	entries: z.array(entry)
});

export type Span = Tracer & {
	name: string;
	start_timestamp: string;
	end_timestamp: string | null;
	trace_id: string;
};
const span: z.ZodType<Span> = tracer.and(z.object({
	name: z.string(),
	start_timestamp: z.string(),
	end_timestamp: z.string().nullable(),
	trace_id: z.string()
}));

export type TaskSpan = Span & {
	input: JSONValue;
	output: JSONValue;
};
const taskSpan: z.ZodType<TaskSpan> = span.and(z.object({
	input: jsonSchema,
	output: jsonSchema
}));

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

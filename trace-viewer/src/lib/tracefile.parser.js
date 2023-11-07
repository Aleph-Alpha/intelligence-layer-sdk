import { z } from 'zod';
const plainEntry = z.object({
	parent: z.string(),
	message: z.string(),
	value: z.any(),
	timestamp: z.string()
});
const spanStart = z.object({
	uuid: z.string(),
	parent: z.string(),
	name: z.string(),
	start: z.string()
});
const spanEnd = z.object({
	uuid: z.string(),
	end: z.string()
});
const taskStart = z.object({
	uuid: z.string(),
	parent: z.string(),
	name: z.string(),
	start: z.string(),
	input: z.any()
});
const taskEnd = z.object({
	uuid: z.string(),
	end: z.string(),
	output: z.any()
});
const logLine = z.discriminatedUnion('entry_type', [
	z.object({
		entry_type: z.literal('PlainEntry'),
		entry: plainEntry
	}),
	z.object({
		entry_type: z.literal('StartSpan'),
		entry: spanStart
	}),
	z.object({
		entry_type: z.literal('EndSpan'),
		entry: spanEnd
	}),
	z.object({
		entry_type: z.literal('StartTask'),
		entry: taskStart
	}),
	z.object({
		entry_type: z.literal('EndTask'),
		entry: taskEnd
	})
]);
export async function parseTraceFile(file) {
	return parseLogLines(
		(await file.text())
			.split(/\r?\n/)
			.filter(Boolean)
			.map((line) => logLine.parse(JSON.parse(line)))
	);
}
export function parseLogLines(lines) {
	const builder = new TraceBuilder();
	for (const line of lines) {
		switch (line.entry_type) {
			case 'PlainEntry':
				builder.addPlainEntry(line.entry);
				break;
			case 'StartSpan':
				builder.startSpan(line.entry);
				break;
			case 'EndSpan':
				builder.endSpan(line.entry);
				break;
			case 'StartTask':
				builder.startTask(line.entry);
				break;
			case 'EndTask':
				builder.endTask(line.entry);
				break;
		}
	}
	return builder.root();
}
class TraceBuilder {
	roots = [];
	tracers = new Map();
	spans = new Map();
	tasks = new Map();
	addPlainEntry(entry) {
		const parent = this.parentTrace(entry.parent);
		// entry.value is any, but value is "Json"
		// eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
		parent.entries.push({ message: entry.message, value: entry.value, timestamp: entry.timestamp });
	}
	startSpan(entry) {
		const parent = this.parentTrace(entry.parent);
		const span = {
			name: entry.name,
			start_timestamp: entry.start,
			end_timestamp: entry.start,
			entries: []
		};
		parent.entries.push(span);
		this.spans.set(entry.uuid, span);
		this.tracers.set(entry.uuid, span);
	}
	endSpan(entry) {
		const span = this.spans.get(entry.uuid);
		span.end_timestamp = entry.end;
	}
	startTask(entry) {
		const parent = this.parentTrace(entry.parent);
		const task = {
			name: entry.name,
			start_timestamp: entry.start,
			end_timestamp: entry.start,
			// eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
			input: entry.input,
			output: undefined,
			entries: []
		};
		parent.entries.push(task);
		this.tasks.set(entry.uuid, task);
		this.tracers.set(entry.uuid, task);
	}
	endTask(entry) {
		const task = this.tasks.get(entry.uuid);
		task.end_timestamp = entry.end;
		// eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
		task.output = entry.output;
	}
	parentTrace(uuid) {
		const parent = this.tracers.get(uuid);
		if (parent) {
			return parent;
		}
		const parentTracer = { entries: [] };
		this.roots.push(uuid);
		this.tracers.set(uuid, parentTracer);
		return parentTracer;
	}
	root() {
		return this.tracers.get(this.roots[0]);
	}
}
//# sourceMappingURL=tracefile.parser.js.map

import { z } from 'zod';
import type { DebugLog, Span } from './log';
import { en } from '@faker-js/faker';

const logEntry = z.object({
    parent: z.string(),
    message: z.string(),
    value: z.any(),
    timestamp: z.string()
});

type LogEntry = z.infer<typeof logEntry>;

const spanStart = z.object({
    uuid: z.string(),
    parent: z.string(),
    name: z.string(),
    start: z.string()
});

type SpanStart = z.infer<typeof spanStart>;

const spanEnd = z.object({
    uuid: z.string(),
    end: z.string()
});

type SpanEnd = z.infer<typeof spanEnd>;

const logLine = z.discriminatedUnion('entry_type', [
    z.object({
        entry_type: z.literal('LogEntry'),
        entry: logEntry
    }),
    z.object({
        entry_type: z.literal('SpanStart'),
        entry: spanStart
    }),
    z.object({
        entry_type: z.literal('SpanEnd'),
        entry: spanEnd
    })
]);

export type LogLine = z.infer<typeof logLine>;

export function parseLogLines(lines: LogLine[]): DebugLog {
    const builder = new LogBuilder();
    for (const line of lines) {
        switch (line.entry_type) {
            case 'LogEntry':
                builder.addLogEntry(line.entry);
                break;
            case 'SpanEnd':
                builder.endSpan(line.entry);
                break;
            case 'SpanStart':
                builder.startSpan(line.entry);
                break;
        }
    }
    return builder.root;
}

class LogBuilder {
    root: DebugLog = { name: '', logs: [] };
    private loggers: Map<string, DebugLog> = new Map<string, DebugLog>();
    private spans: Map<string, Span> = new Map<string, Span>();

    addLogEntry(entry: LogEntry) {
        const parent = this.parentLogger(entry.parent);
        // entry.value is any, but value is "Json"
        // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
        parent.logs.push({ message: entry.message, value: entry.value, timestamp: entry.timestamp });
    }

    startSpan(entry: SpanStart) {
        const parent = this.parentLogger(entry.parent);
        const span: Span = {
            name: entry.name,
            start_timestamp: entry.start,
            end_timestamp: entry.start,
            logs: []
        };
        parent.logs.push(span);
        this.spans.set(entry.uuid, span);
        this.loggers.set(entry.uuid, span);
    }

    endSpan(entry: SpanEnd) {
        const span = this.spans.get(entry.uuid);
        span!.end_timestamp = entry.end;
    }

    parentLogger(uuid: string): DebugLog {
        const parent = this.loggers.get(uuid);
        if (parent) {
            return parent;
        }
        this.root = { name: uuid, logs: [] };
        this.loggers.set(uuid, this.root);
        return this.root;
    }
}

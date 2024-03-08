import { describe, test, expect } from 'vitest';
import { traceRange, renderDuration, tracer } from './trace';
import { randomLogEntry, randomTracer, randomSpan } from './trace.test_utils';

describe('trace', () => {
	test('parser works for empty', () => {
		const basicTrace = '{"entries":[]}';
		const obj = tracer.parse(JSON.parse(basicTrace));
		expect(obj).toEqual({ entries: [] });
	});

	test('parser works for span', () => {
		const basicTrace =
			'{"entries":[{"entries":[],"name":"test","start_timestamp":"2024-03-07T15:08:11.069884Z","end_timestamp":"2024-03-07T15:08:11.069884Z","trace_id":"7771aee4-b305-4504-850d-ebabed761eea"}]}';
		// eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
		const parsedTrace = JSON.parse(basicTrace);
		const serializedTrace = tracer.parse(parsedTrace);
		expect(serializedTrace).toEqual(parsedTrace);
	});

	test('parser works for task span', () => {
		const basicTrace =
			'{"entries":[{"entries":[],"name":"test","start_timestamp":"2024-03-07T15:18:30.377538Z","end_timestamp":null,"trace_id":"89e151f1-2379-47fc-954b-56e1953edd03","input":"input","output":null}]}';
		// eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
		const parsedTrace = JSON.parse(basicTrace);
		const serializedTrace = tracer.parse(parsedTrace);
		expect(serializedTrace).toEqual(parsedTrace);
	});
	test('parser works for span with log', () => {
		const basicTrace =
			'{"entries":[{"entries":[{"message":"test","value":"value","timestamp":"2024-03-07T15:19:38.977970Z","trace_id":"697a9e44-47b4-4dfe-8c61-f65e676f1de7"}],"name":"test","start_timestamp":"2024-03-07T15:19:38.977896Z","end_timestamp":null,"trace_id":"697a9e44-47b4-4dfe-8c61-f65e676f1de7"}]}';
		// eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
		const parsedTrace = JSON.parse(basicTrace);
		const serializedTrace = tracer.parse(parsedTrace);
		expect(serializedTrace).toEqual(parsedTrace);
	});
	test('parser works for nested spans', () => {
		const basicTrace =
			'{"entries":[{"entries":[],"name":"test","start_timestamp":"2024-03-07T15:18:30.377538Z","end_timestamp":null,"trace_id":"89e151f1-2379-47fc-954b-56e1953edd03","input":"input","output":null}]}';
		// eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
		const parsedTrace = JSON.parse(basicTrace);
		const serializedTrace = tracer.parse(parsedTrace);
		expect(serializedTrace).toEqual(parsedTrace);
	});
});

describe('traceRange', () => {
	test('Single Span', () => {
		const span = randomSpan();
		const range = traceRange({ entries: [span] });
		expect(range?.from).toEqual(new Date(span.start_timestamp));
		expect(range?.to).toEqual(new Date(span.end_timestamp));
	});

	test('Empty Logs', () => {
		const range = traceRange({ entries: [] });
		expect(range).toBeNull();
	});

	test('Single Entry', () => {
		const entry = randomLogEntry();
		const range = traceRange({ entries: [entry] });
		expect(range?.from).toEqual(range?.to);
		expect(range?.from).toEqual(new Date(entry.timestamp));
	});

	test('Larger Trace', () => {
		const trace = randomTracer();
		const range = traceRange(trace);
		if (range) {
			expect(range.from <= range.to).toBe(true);
		}
	});
});

describe('renderDuration', () => {
	test('ms', () => {
		expect(renderDuration(500)).toBe('500ms');
	});
	test('s', () => {
		expect(renderDuration(1000)).toBe('1s');
	});
	test('partial s', () => {
		expect(renderDuration(1001)).toBe('1.001s');
	});
	test('min', () => {
		expect(renderDuration(60000)).toBe('1min');
	});
	test('partial min', () => {
		expect(renderDuration(60050)).toBe('1.001min');
	});
	test('hr', () => {
		expect(renderDuration(3600000)).toBe('1h');
	});
	test('partial hr', () => {
		expect(renderDuration(3605000)).toBe('1.001h');
	});
});

import { describe, test, expect } from 'vitest';
import { traceRange, renderDuration } from './trace';
import { randomLogEntry, randomTracer, randomSpan } from './trace.test_utils';

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

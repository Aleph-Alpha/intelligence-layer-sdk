import type { Tracer } from './trace';

let db: Tracer | null = null;

export function set(value: Tracer | null): void {
	db = value;
}

export function get(): Tracer | null {
	if (db === null) {
		throw new Error('Database is not set.');
	}
	return db;
}

export function clear(): void {
	db = null;
}

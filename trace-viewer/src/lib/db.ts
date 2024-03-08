import type { Tracer } from './trace';

let db: Tracer | null = null;

export function set(value: Tracer | null): void {
	db = value;
}

export function get(): Tracer | null {
	return db;
}

export function clear(): void {
	db = null;
}

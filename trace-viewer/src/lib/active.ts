import { type Tracer, type SpanEntry } from '$lib/trace';
import { writable } from 'svelte/store';

export const activeTrace = writable<Tracer | null>(null);
export const activeSpan = writable<SpanEntry | null>(null);

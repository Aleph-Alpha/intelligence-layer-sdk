import type { Tracer, SpanEntry } from '$lib/trace';
import { writable } from 'svelte/store';

export const activeSpan = writable<SpanEntry | null>(null);

export const activeTrace = writable<Tracer | null>(null);

import type { Tracer, SpanEntry } from '$lib/trace';
import { writable, get } from 'svelte/store';

export const activeTrace = writable(null);
export const activeSpan = writable<SpanEntry | null>(null);

import type { Tracer, SpanEntry } from '$lib/trace';
import { writable } from 'svelte/store';

// initialize ativeTrace from globalTrace
export const activeTrace = writable('1');
export const activeSpan = writable<SpanEntry | null>(null);

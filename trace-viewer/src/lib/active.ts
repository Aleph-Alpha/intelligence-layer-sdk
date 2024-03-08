import type { SpanEntry } from '$lib/trace';
import { writable } from 'svelte/store';

export const activeSpan = writable<SpanEntry | null>(null);

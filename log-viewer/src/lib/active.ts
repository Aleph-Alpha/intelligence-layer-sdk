import type { SpanEntry } from '$lib/log';
import { writable } from 'svelte/store';

export const activeSpan = writable<SpanEntry | null>();

import type { DebugLog, SpanEntry } from '$lib/log';
import { writable } from 'svelte/store';

export const activeSpan = writable<SpanEntry | null>(null);

export const activeLog = writable<DebugLog | null>(null);

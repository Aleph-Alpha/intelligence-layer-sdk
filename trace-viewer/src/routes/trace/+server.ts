import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { set } from '$lib/db';
import { tracer } from '$lib/trace';

export const POST: RequestHandler = async ({ request }) => {
	const trace = tracer.parse((await request.json()) as string);
	set(trace);
	return json({ status: 'success', message: 'Trace successfully parsed and set.' });
};

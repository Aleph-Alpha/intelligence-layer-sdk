import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { tracer } from '$lib/trace';
import { randomTracer } from '$lib/trace.test_utils';

export const POST: RequestHandler = async ({ request, locals }) => {
	const trace = await request.json();
	locals.globalTrace = randomTracer();
	console.log(locals.globalTrace);
	return json({ status: 'success', message: 'Trace successfully parsed and set.' });
};

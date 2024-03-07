import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { set } from '$lib/db';
import { get } from '$lib/db';

export const POST: RequestHandler = async ({ request, locals }) => {
	const trace = await request.json();
	set('trace', trace); // Store the trace in the database
	return json({ status: 'success', message: 'Trace successfully parsed and set.' });
};

// export const GET: RequestHandler = async () => {
// 	const trace = get('trace');
// 	return json({ trace });
// };

import { clear, get, set } from '$lib/db';
import type { PageServerLoad } from './$types';
import { tracer } from '$lib/trace';
import type { Actions } from './$types';

export const load: PageServerLoad = () => {
	return { trace: get() };
};

export const actions = {
	clearTrace: () => {
		clear();
	},
	setTrace: async ({ request }) => {
		const data = await request.formData();
		const trace = data.get('trace')?.valueOf();
		if (typeof trace === 'string') {
			const parsedTracer = tracer.parse(JSON.parse(trace));
			set(parsedTracer);
		}
	}
} satisfies Actions;

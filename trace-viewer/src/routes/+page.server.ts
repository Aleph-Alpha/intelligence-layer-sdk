import { clear, get } from '$lib/db';
import { activeTrace } from '$lib/active';
import type { PageServerLoad } from './$types';
export const load: PageServerLoad = () => {
	return { trace: get() };
};

export const actions = {
	default: async (event) => {
		clear();
	}
};

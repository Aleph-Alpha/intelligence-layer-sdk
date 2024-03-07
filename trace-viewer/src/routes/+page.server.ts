
import { get } from '$lib/db';
import type { PageServerLoad } from './$types';
export const load: PageServerLoad = ({ }) => {
	const trace = get('trace')
	return { trace: trace };
};

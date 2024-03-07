import type { PageServerLoad } from './$types';
export const load: PageServerLoad = ({ locals }) => {
	console.log(locals.globalTrace);
	return { trace: locals.globalTrace };
};

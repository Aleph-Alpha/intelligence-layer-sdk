let db = {};

export function set(key, value) {
	db[key] = value;
}

export function get(key) {
	return db[key];
}

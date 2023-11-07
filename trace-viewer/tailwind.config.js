const defaultTheme = require('tailwindcss/defaultTheme');

/** @type {import('tailwindcss').Config} */
export default {
	content: ['./src/**/*.{html,js,svelte,ts}'],
	theme: {
		extend: {
			colors: {
				accent: {
					DEFAULT: '#e3ff00',
					50: '#ffffe4',
					100: '#feffc4',
					200: '#fbff90',
					300: '#f2ff50',
					400: '#e3ff00',
					500: '#c7e600',
					600: '#9ab800',
					700: '#748b00',
					800: '#5b6d07',
					900: '#4d5c0b',
					950: '#283400'
				}
			},
			fontFamily: {
				sans: ['Raleway', ...defaultTheme.fontFamily.sans],
				mono: ['iA Writer Mono V', ...defaultTheme.fontFamily.mono]
			}
		}
	},
	plugins: [
		require('@tailwindcss/typography'),
		require('@tailwindcss/forms'),
		require('@tailwindcss/container-queries')
	]
};

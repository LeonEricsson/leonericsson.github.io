import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic':
          'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
      },
      colors: {
        'warm-cream': '#F5F1E8',
        'vintage-paper': '#FFFBF6',
        'deep-charcoal': '#1F1F1F',
        'medium-gray': '#6B6B6B',
        'light-gray': '#A8A8A8',
        'sepia-accent': '#8B7355',
        'subtle-line': '#D4CFC7',
      },
      fontFamily: {
        'garamond': ['Garamond', 'EB Garamond', 'serif'],
        'palatino': ['Palatino Linotype', 'Book Antiqua', 'Palatino', 'serif'],
        'georgia': ['Georgia', 'serif'],
        'charter': ['Charter', 'Bitstream Charter', 'Georgia', 'serif'],
      }
    },
  },
  plugins: [],
}
export default config

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
        'eb-garamond': ['"EB Garamond"', 'Garamond', 'serif'],
        'crimson': ['"Crimson Text"', 'Georgia', 'serif'],
        'cormorant': ['"Cormorant Garamond"', 'Garamond', 'serif'],
        'lora': ['"Lora"', 'Georgia', 'serif'],
        'georgia': ['Georgia', 'serif'],
      }
    },
  },
  plugins: [],
}
export default config

module.exports = {
  content: [
    "./pages/**/*.{js,jsx,ts,tsx}",
    "./components/**/*.{js,jsx,ts,tsx}",
    "./app/**/*.{js,jsx,ts,tsx}",
    "./src/**/*.{js,jsx,ts,tsx}", // Add this line to include all files in src
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
        display: ['Space Grotesk', 'sans-serif'],
      },
      fontSize: {
        hero: '48px',
        h1: '36px',
        h2: '28px',
        h3: '22px',
        body: '16px',
        caption: '14px',
        small: '12px',
      },
      colors: {
        'deep-space-navy': '#0B0B1A',
        'charcoal-elevated': '#1A1B2E',
        'electric-purple': '#8B5CF6',
        'emerald-success': '#10B981',
        'amber-warning': '#F59E0B',
        'rose-danger': '#F43F5E',
      },
      boxShadow: {
        'glow': '0 0 15px rgba(139, 92, 246, 0.7)', // electric-purple with some transparency
        'elevation-1': '0 2px 4px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.06)',
        'elevation-2': '0 4px 6px rgba(0, 0, 0, 0.1), 0 2px 4px rgba(0, 0, 0, 0.06)',
        'elevation-3': '0 8px 16px rgba(0, 0, 0, 0.1), 0 4px 8px rgba(0, 0, 0, 0.06)',
      },
      borderRadius: {
        'card': '12px',
        'button': '8px',
        'form': '6px',
      },
    },
    keyframes: {
      'shine': {
        '0%': { backgroundPosition: '-200% center' },
        '100%': { backgroundPosition: '200% center' },
      },
      'pulse-light': {
        '0%, 100%': { opacity: 1 },
        '50%': { opacity: 0.7 },
      },
    },
    animation: {
      'shine': 'shine 2s linear infinite',
      'pulse-light': 'pulse-light 1.5s ease-in-out infinite',
    },
   },
    darkMode: 'class',
    plugins: [],
    corePlugins: {
      preflight: false,
    }
  }

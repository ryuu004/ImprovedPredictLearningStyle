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
        'charcoal-elevated': '#151626',
        'charcoal-hover': '#1F2033',
        'charcoal-light': '#36454F',
        'charcoal-dark': '#2B333E',
        'electric-purple': '#8B5CF6',
        'emerald-success': '#10B981',
        'amber-warning': '#F59E0B',
        'rose-danger': '#F43F5E',
        'dark-charcoal': '#0F101A',
        'purple-accent': '#8B5CF6',
        'pink-accent': '#EC4899',
        'sidebar-start': '#0B0B1A',
        'sidebar-end': '#151626',
        'dark-navy-start': '#0B0B1A',
        'dark-navy-end': '#1A0F2D',
        'button-purple-start': '#8B5CF6',
        'button-purple-end': '#C084FC',
      },
      backgroundImage: {
        'radial-gradient-charcoal': 'radial-gradient(circle at center, #1F2033 0%, #151626 100%)',
        'linear-overlay-purple-teal': 'linear-gradient(to right, rgba(139, 92, 246, 0.2), rgba(16, 185, 129, 0.2))',
        'border-gradient-purple-teal': 'linear-gradient(to right, #8B5CF6, #10B981)',
        'gradient-navy-purple': 'linear-gradient(135deg, #0B0B1A 0%, #1A0F2D 100%)',
        'gradient-button-purple': 'linear-gradient(90deg, #8B5CF6 0%, #C084FC 100%)',
      },
      boxShadow: {
        'glow': '0 0 15px rgba(139, 92, 246, 0.7)', // electric-purple with some transparency
        'elevation-1': '0 2px 4px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.06)',
        'elevation-2': '0 4px 6px rgba(0, 0, 0, 0.1), 0 2px 4px rgba(0, 0, 0, 0.06)',
        'elevation-3': '0 8px 16px rgba(0, 0, 0, 0.1), 0 4px 8px rgba(0, 0, 0, 0.06)',
        'inset-highlight': 'inset 0 1px 3px 0 rgba(255, 255, 255, 0.1), inset 0 -1px 1px 0 rgba(0, 0, 0, 0.1)',
        'layered-shadow-1': '0px 1px 2px rgba(0, 0, 0, 0.05), 0px 2px 4px rgba(0, 0, 0, 0.05)',
        'layered-shadow-2': '0px 4px 8px rgba(0, 0, 0, 0.1), 0px 8px 16px rgba(0, 0, 0, 0.1)',
        'layered-shadow-3': '0px 12px 24px rgba(0, 0, 0, 0.15), 0px 24px 48px rgba(0, 0, 0, 0.15)',
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
      spin: {
        from: { transform: 'rotate(0deg)' },
        to: { transform: 'rotate(360deg)' },
      },
      float: {
        '0%': { transform: 'translateY(0px)' },
        '50%': { transform: 'translateY(-10px)' },
        '100%': { transform: 'translateY(0px)' },
      },
      'pop-in': {
        '0%': { transform: 'scale(0)', opacity: 0 },
        '70%': { transform: 'scale(1.1)', opacity: 1 },
        '100%': { transform: 'scale(1)' },
      },
    },
    animation: {
      'shine': 'shine 2s linear infinite',
      'pulse-light': 'pulse-light 1.5s ease-in-out infinite',
      'spin': 'spin 1s linear infinite',
      'float': 'float 3s ease-in-out infinite',
      'pop-in': 'pop-in 0.5s cubic-bezier(0.68, -0.55, 0.27, 1.55)',
    },
   },
     darkMode: 'class',
     plugins: [],
     corePlugins: {
       preflight: false,
     }
   }

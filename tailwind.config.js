module.exports = {
  content: [
    "./pages/**/*.{js,jsx,ts,tsx}",
    "./components/**/*.{js,jsx,ts,tsx}",
    "./app/**/*.{js,jsx,ts,tsx}",
    "./src/**/*.{js,jsx,ts,tsx}", // Add this line to include all files in src
  ],
  theme: {
    extend: {
      colors: {
        'dark-navy': '#0D1117',
        'charcoal': '#161B22',
        'accent-blue': '#007BFF',
        'accent-cyan': '#00FFFF',
        'subtle-gray-light': '#2D333B',
        'subtle-gray-dark': '#21262D',
      },
      boxShadow: {
        'glow': '0 0 15px rgba(0, 123, 255, 0.5)',
      },
    },
   },
   darkMode: 'class',
   plugins: [],
 }

"use client";

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { motion } from 'framer-motion';
import { FaHome, FaChartBar, FaChartLine, FaFlask, FaUserCircle } from 'react-icons/fa';
import { useState, useEffect } from 'react';

export default function Sidebar({ isCollapsed, toggleSidebar }) {
  const pathname = usePathname();
  const [isMdScreen, setIsMdScreen] = useState(false);

  useEffect(() => {
    const checkScreenSize = () => {
      setIsMdScreen(window.innerWidth >= 768); // Tailwind's 'md' breakpoint
    };
    checkScreenSize();
    window.addEventListener('resize', checkScreenSize);
    return () => window.removeEventListener('resize', checkScreenSize);
  }, []);

  const navItems = [
    { name: 'Home', href: '/', icon: FaHome },
    { name: 'Student Data', href: '/dashboard', icon: FaChartBar },
    { name: 'Model Stats', href: '/model-stats', icon: FaChartLine },
    { name: 'Simulate', href: '/simulate', icon: FaFlask },
  ];

  return (
    <motion.aside
      initial={false}
      animate={{
        width: isCollapsed ? 80 : 256,
        x: isMdScreen ? 0 : (isCollapsed ? -256 : 0), // Animate x based on screen size and collapse state
      }}
      transition={{ duration: 0.3, ease: "easeInOut" }}
      className={`fixed top-0 left-0 h-screen text-white p-6 z-40
      bg-gradient-to-b from-deep-space-navy to-charcoal-elevated shadow-2xl border-r border-gray-800 rounded-r-2xl
      md:flex-shrink-0 backdrop-filter backdrop-blur-lg bg-opacity-80
      overflow-y-auto z-50`}
    >
      <style jsx>{`
        .animated-gradient-text {
          background: linear-gradient(to right, #A755F7, #EC4899, #A755F7); /* Adjust hex codes if your Tailwind config defines custom colors for electric-purple and pink-accent */
          -webkit-background-clip: text;
          background-clip: text;
          color: transparent;
          background-size: 200% auto;
          animation: gradient-animation 3s linear infinite;
        }

        @keyframes gradient-animation {
          0% { background-position: 0% center; }
          100% { background-position: -200% center; }
        }
      `}</style>
      <div className="flex items-center justify-between mb-6 h-16">
        {!isCollapsed && (
          <div className="text-3xl font-extrabold tracking-widest drop-shadow-lg">
            <span className="animated-gradient-text">PredictLS</span>
          </div>
        )}
        <button onClick={toggleSidebar} className="p-2 rounded-md bg-charcoal-elevated hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-electric-purple transition-all duration-300 ease-out glow-on-hover">
          {/* Mobile close button, only visible on small screens when sidebar is open */}
          {!isCollapsed && (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-white md:hidden hover:text-charcoal-elevated" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          )}
          {/* Desktop toggle button */}
          <svg xmlns="http://www.w3.org/2000/svg" className={`h-6 w-6 text-white ${!isCollapsed ? 'md:block hidden' : ''} hover:text-charcoal-elevated`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            {isCollapsed ? (
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 12h14" />
            ) : (
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
            )}
          </svg>
        </button>
      </div>

      <div className={`flex items-center mb-8 p-2 rounded-lg bg-charcoal-elevated bg-opacity-30 shadow-inner h-16`}>
        {/* Icon container with fixed width */}
        <div className="flex-shrink-0 flex justify-center items-center" style={{ width: '40px' }}>
          <FaUserCircle className="text-4xl text-electric-purple" />
        </div>
        {/* Text content - animates max-width and opacity */}
        <motion.div
          initial={false}
          animate={{
            opacity: isCollapsed ? 0 : 1,
            maxWidth: isCollapsed ? 0 : '100%',
            marginLeft: isCollapsed ? 0 : '0.75rem', // Conditional margin-left for spacing
            overflow: 'hidden',
            whiteSpace: 'nowrap',
            pointerEvents: isCollapsed ? 'none' : 'auto',
          }}
          transition={{ duration: 0.3, ease: "easeInOut" }}
          className="flex flex-col flex-grow" // Allow text to take up remaining space
        >
          <span className="font-semibold text-lg text-white">Rhyio</span>
          <span className="text-sm text-gray-300">Admin</span>
        </motion.div>
      </div>

      <nav className="">
        <ul className="list-none p-0 m-0">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href;
            return (
              <motion.li
                key={item.name}
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.98 }}
                transition={{ duration: 0.2 }}
                className="relative mb-3"
              >
                <Link href={item.href} className={`flex items-center rounded-lg group transition-all duration-300 ease-in-out w-full
                  ${isActive
                    ? 'bg-gradient-to-r from-electric-purple to-pink-accent text-white shadow-lg border border-electric-purple transform scale-100'
                    : 'text-gray-300 hover:bg-charcoal-elevated hover:text-electric-purple'
                  }
                  focus:outline-none focus:ring-2 focus:ring-electric-purple focus:ring-opacity-75
                  glass-effect
                  `}
                >
                  {/* Inner container with fixed height and horizontal padding */}
                  <div className="flex items-center px-4 w-full h-12">
                    {/* Icon container with fixed width to prevent horizontal movement */}
                    <div className="flex-shrink-0 flex justify-center items-center" style={{ width: '28px' }}>
                      <motion.span
                        className="text-2xl"
                        transition={{ duration: 0.3, ease: "easeInOut" }}
                      >
                        <Icon />
                      </motion.span>
                    </div>
                    {/* Text content - animates max-width and opacity */}
                    <motion.span
                      initial={false}
                      animate={{
                        opacity: isCollapsed ? 0 : 1,
                        maxWidth: isCollapsed ? 0 : '100%', // Animate max-width
                        marginLeft: isCollapsed ? 0 : '0.75rem', // Conditional margin-left
                        overflow: 'hidden', // Hide overflow as width shrinks
                        whiteSpace: 'nowrap', // Prevent text wrapping
                        pointerEvents: isCollapsed ? 'none' : 'auto', // Disable interaction when hidden
                      }}
                      transition={{ duration: 0.3, ease: "easeInOut" }}
                      className="flex-grow" // Allow text to take up remaining space
                    >
                      {item.name}
                    </motion.span>
                  </div>
                </Link>
                {isActive && (
                  <motion.div
                    className="absolute inset-y-0 left-0 w-1 bg-gradient-to-t from-pink-accent to-electric-purple rounded-full"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.3 }}
                  />
                )}
                {/* Subtle shadow and glass effect for navigation items */}
                <motion.div
                  className="absolute inset-0 rounded-lg -z-10"
                  initial={{ opacity: 0 }}
                  whileHover={{ opacity: 1, boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08)" }}
                  transition={{ duration: 0.2 }}
                  style={{
                    background: 'rgba(255, 255, 255, 0.05)',
                    backdropFilter: 'blur(5px)',
                    WebkitBackdropFilter: 'blur(5px)',
                  }}
                />
              </motion.li>
            );
          })}
        </ul>
      </nav>
    </motion.aside>
  );
}
"use client";

import Link from 'next/link';
import { useState } from 'react';
import { usePathname } from 'next/navigation';

export default function Sidebar() {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const pathname = usePathname();

  const toggleSidebar = () => {
    setIsCollapsed(!isCollapsed);
  };

  const navItems = [
    { name: 'Home', href: '/' },
    { name: 'Student Data', href: '/dashboard' },
    { name: 'Model Stats', href: '/model-stats' },
    { name: 'Simulate', href: '/simulate' },
  ];

  return (
    <aside className={`min-h-screen bg-deep-space-navy text-white p-4 shadow-lg transition-all duration-300 ease-in-out ${isCollapsed ? 'w-20' : 'w-64'}`}>
      <div className="flex items-center justify-between mb-6">
        {!isCollapsed && <div className="text-2xl font-bold text-electric-purple">Navigation</div>}
        <button onClick={toggleSidebar} className="p-2 rounded-md hover:bg-charcoal-elevated focus:outline-none focus:ring-2 focus:ring-electric-purple transition-all duration-300 ease-out">
          {isCollapsed ? (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 12h14" />
            </svg>
          ) : (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
            </svg>
          )}
        </button>
      </div>
      <nav className="space-y-4">
        <ul>
          {navItems.map((item) => (
            <li key={item.name}>
              <Link href={item.href} className={`flex items-center p-2 text-base font-normal rounded-lg group transition-all duration-200 ease-in-out hover:bg-charcoal-elevated focus:outline-none focus:ring-2 focus:ring-electric-purple ${pathname === item.href ? 'bg-electric-purple text-white shadow-md' : 'text-white'}`}>
                {isCollapsed ? (
                  <span className="sr-only">{item.name}</span>
                ) : (
                  item.name
                )}
              </Link>
            </li>
          ))}
        </ul>
      </nav>
    </aside>
  );
}
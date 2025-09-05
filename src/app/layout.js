"use client";

import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import Sidebar from "./components/Sidebar";
import MotionMainWrapper from "./components/MotionMainWrapper";
import { useState } from "react";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});


export default function RootLayout({ children }) {
  const [isCollapsed, setIsCollapsed] = useState(false);

  const toggleSidebar = () => {
    setIsCollapsed(!isCollapsed);
  };

  return (
    <html lang="en" className="dark">
      <body className={`${geistSans.variable} ${geistMono.variable} bg-deep-space-navy flex h-screen`}>
        {/* Sidebar */}
        <Sidebar isCollapsed={isCollapsed} toggleSidebar={toggleSidebar} />

        {/* Overlay for mobile */}
        {!isCollapsed && (
          <div
            className="fixed inset-0 bg-black bg-opacity-50 z-30 transition-opacity duration-300 md:hidden"
            onClick={toggleSidebar}
          ></div>
        )}

        {/* Main Content */}
        <div className={`flex-grow transition-all duration-300 ease-in-out ${isCollapsed ? 'ml-[128px]' : 'ml-[304px]'}`}>
          <MotionMainWrapper isCollapsed={isCollapsed}>
            <div className={`p-4`}>
              {children}
            </div>
          </MotionMainWrapper>
        </div>
      </body>
    </html>
  );
}

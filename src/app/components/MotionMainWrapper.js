"use client";

import dynamic from "next/dynamic";
import { motion } from "framer-motion";

const MotionMain = dynamic(() => Promise.resolve(motion.main), {
  ssr: false,
});

export default function MotionMainWrapper({ children, isCollapsed }) {
  return (
    <MotionMain
      className={`flex-grow transition-all duration-300 ease-in-out`}
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
    >
      {children}
    </MotionMain>
  );
}
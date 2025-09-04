"use client";
import { useState, useEffect, useRef } from 'react';

const useCountUp = (end, duration = 2000) => {
  const [count, setCount] = useState(0);
  const ref = useRef(0);

  const increment = end / (duration / 10);

  useEffect(() => {
    if (ref.current < end) {
      const counter = setInterval(() => {
        ref.current += increment;
        if (ref.current >= end) {
          ref.current = end;
          clearInterval(counter);
        }
        setCount(Math.floor(ref.current));
      }, 10);
      return () => clearInterval(counter);
    }
  }, [end, duration, increment]);

  return count;
};

export default function LearningStyleMetric({ style, count, percentage }) {
  const animatedCount = useCountUp(count);
  const animatedPercentage = useCountUp(parseFloat(percentage));

  return (
    <div className="mb-2">
      <div className="flex justify-between items-center text-gray-300 text-xs mb-1">
        <span className="capitalize">{style}:</span>
        <span className="font-semibold">{animatedCount} students ({animatedPercentage}%)</span>
      </div>
      <div className="w-full bg-charcoal-elevated rounded-full h-1.5">
        <div
          className="bg-electric-purple h-1.5 rounded-full"
          style={{ width: `${animatedPercentage}%` }}
        ></div>
      </div>
    </div>
  );
}
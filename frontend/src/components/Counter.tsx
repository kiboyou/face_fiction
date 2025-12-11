"use client";

import { useEffect, useRef, useState } from "react";

type CounterProps = {
  value: number; // target number
  duration?: number; // ms
  format?: (v: number) => string;
};

export default function Counter({ value, duration = 1200, format = (v) => `${Math.round(v)}` }: CounterProps) {
  const [display, setDisplay] = useState(0);
  const ref = useRef<HTMLSpanElement | null>(null);
  const [active, setActive] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        setActive(true);
        io.disconnect();
      }
    }, { threshold: 0.2 });
    io.observe(el);
    return () => io.disconnect();
  }, []);

  useEffect(() => {
    if (!active) return;
    const start = performance.now();
    const from = 0;
    const to = value;
    const step = (t: number) => {
      const p = Math.min(1, (t - start) / duration);
      const eased = 1 - Math.pow(1 - p, 3); // easeOutCubic
      setDisplay(from + (to - from) * eased);
      if (p < 1) requestAnimationFrame(step);
    };
    const raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [active, value, duration]);

  return <span ref={ref}>{format(display)}</span>;
}

"use client";

import { useEffect, useRef, useState } from "react";

export default function ProgressBar({ percent, color }: { percent: number; color: string }) {
  const [active, setActive] = useState(false);
  const ref = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        setActive(true);
        io.disconnect();
      }
    }, { threshold: 0.1 });
    io.observe(el);
    return () => io.disconnect();
  }, []);
  return (
    <div ref={ref} className="h-2 rounded-full bg-slate-100">
      <div
        className="h-2 rounded-full"
        style={{
          width: active ? `${Math.round(percent)}%` : 0,
          background: color,
          transition: "width 800ms ease"
        }}
      />
    </div>
  );
}

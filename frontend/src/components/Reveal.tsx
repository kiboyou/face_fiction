"use client";

import React, { PropsWithChildren, useEffect, useRef, useState } from "react";

type RevealProps = PropsWithChildren<{
  delay?: number; // ms
  as?: React.ElementType;
  className?: string;
}>;

export default function Reveal({ children, delay = 0, as = "div", className = "" }: RevealProps) {
  const ref = useRef<HTMLElement | null>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setTimeout(() => setVisible(true), delay);
          io.disconnect();
        }
      },
      { threshold: 0.12 }
    );
    io.observe(el);
    return () => io.disconnect();
  }, [delay]);

  const Comp: any = as;
  return (
    <Comp ref={ref} className={`reveal ${visible ? "is-visible" : ""} ${className}`}>
      {children}
    </Comp>
  );
}

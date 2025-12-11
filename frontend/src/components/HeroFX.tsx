"use client";

import { useEffect, useRef } from "react";

export default function HeroFX() {
  const lastY = useRef(0);
  useEffect(() => {
    const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) return;
    const bg = document.querySelector<HTMLElement>(".hero-bg");
    if (!bg) return;
    let raf = 0;
    const onScroll = () => {
      const y = window.scrollY;
      if (Math.abs(y - lastY.current) < 1) return;
      lastY.current = y;
      if (!raf) {
        raf = requestAnimationFrame(() => {
          raf = 0;
          const translate = Math.min(40, y * 0.06);
          bg.style.transform = `translateY(${translate}px)`;
        });
      }
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      window.removeEventListener("scroll", onScroll);
      if (raf) cancelAnimationFrame(raf);
    };
  }, []);
  return null;
}

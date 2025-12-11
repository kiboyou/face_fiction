"use client";

import { useEffect } from "react";

export default function GlobalInteractions() {
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      const btn = target.closest(".btn-primary, .btn-secondary") as HTMLElement | null;
      if (!btn) return;
      const rect = btn.getBoundingClientRect();
      const span = document.createElement("span");
      const size = Math.max(rect.width, rect.height);
      span.className = "ripple";
      span.style.width = span.style.height = `${size}px`;
      span.style.left = `${e.clientX - rect.left - size / 2}px`;
      span.style.top = `${e.clientY - rect.top - size / 2}px`;
      btn.appendChild(span);
      span.addEventListener("animationend", () => span.remove());
    };
    document.addEventListener("click", handler);
    return () => document.removeEventListener("click", handler);
  }, []);
  return null;
}

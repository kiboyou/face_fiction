"use client";

import { useEffect, useState } from "react";

type Toast = { id: number; message: string; type?: "success" | "error" | "info" };

export default function Toaster() {
  const [toasts, setToasts] = useState<Toast[]>([]);

  useEffect(() => {
    const handler = (e: Event) => {
      const ce = e as CustomEvent<{ message: string; type?: Toast["type"] }>;
      const id = Date.now() + Math.random();
      setToasts((t) => [...t, { id, message: ce.detail.message, type: ce.detail.type }]);
      setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 3500);
    };
    window.addEventListener("toast", handler as EventListener);
    return () => window.removeEventListener("toast", handler as EventListener);
  }, []);

  return (
    <div className="fixed z-50 top-4 right-4 space-y-2">
      {toasts.map((t) => (
        <div
          key={t.id}
          className="px-4 py-3 rounded-lg shadow-lg"
          style={{
            background: t.type === "error" ? "#FEE2E2" : t.type === "success" ? "#ECFDF5" : "#EFF6FF",
            color: t.type === "error" ? "#7F1D1D" : t.type === "success" ? "#065F46" : "#1E3A8A",
            border: `1px solid ${t.type === "error" ? "#FCA5A5" : t.type === "success" ? "#A7F3D0" : "#BFDBFE"}`,
          }}
        >
          {t.message}
        </div>
      ))}
    </div>
  );
}

export function toast(message: string, type?: Toast["type"]) {
  window.dispatchEvent(new CustomEvent("toast", { detail: { message, type } }));
}

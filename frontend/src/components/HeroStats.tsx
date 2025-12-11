"use client";

import { fetchMetrics } from "@/components/api";
import Counter from "@/components/Counter";
import { useEffect, useState } from "react";

export default function HeroStats() {
  const [precision, setPrecision] = useState<number | null>(null);
  const [mediaCount, setMediaCount] = useState<number | null>(null);
  const [modelCount, setModelCount] = useState<number>(3);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let mounted = true;
    async function load() {
      try {
        const metrics = await fetchMetrics().catch(() => null);
        if (!mounted) return;
        // Précision : metrics.evaluation?.validation?.accuracy
        setPrecision(metrics && metrics.evaluation && typeof metrics.evaluation.validation?.accuracy === "number"
          ? metrics.evaluation.validation.accuracy * 100
          : null);
        // Médias analysés : metrics.evaluation?.dataset_info?.train_samples + val_samples
        setMediaCount(metrics && metrics.evaluation && metrics.evaluation.dataset_info && typeof metrics.evaluation.dataset_info.train_samples === "number" && typeof metrics.evaluation.dataset_info.val_samples === "number"
          ? metrics.evaluation.dataset_info.train_samples + metrics.evaluation.dataset_info.val_samples
          : null);
        // Nombre de modèles : 1 (fixe, ou metrics.model_type si besoin)
        setModelCount(3);
      } finally {
        if (mounted) setLoading(false);
      }
    }
    load();
    return () => { mounted = false; };
  }, []);

  return (
    <div className="mt-8 grid grid-cols-3 gap-4 max-w-md">
      <div className="card">
        <div className="text-2xl font-bold">
          {loading ? "..." : <Counter value={precision ?? 0} format={(v) => precision !== null ? `${v.toFixed(1)}%` : "-"} />}
        </div>
        <div className="text-sm text-slate-600">Précision</div>
      </div>
      <div className="card">
        <div className="text-2xl font-bold">
          {loading ? "..." : <Counter value={mediaCount ?? 0} format={(v) => mediaCount !== null ? `${Math.round(v).toLocaleString()}` : "-"} />}
        </div>
        <div className="text-sm text-slate-600">Médias analysés</div>
      </div>
      <div className="card">
        <div className="text-2xl font-bold">
          {/* {loading ? "..." : <Counter value={modelCount} format={(v) => v.toString()} />} */}
          3
        </div>
        <div className="text-sm text-slate-600">Modèles</div>
      </div>
    </div>
  );
}

"use client";

import { useEffect, useState } from "react";

export default function MetricsSection() {
  const [metrics, setMetrics] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

  useEffect(() => {
    fetch(`${apiBase}/metrics`)
      .then(async (r) => {
        if (!r.ok) throw new Error(await r.text());
        return r.json();
      })
      .then(setMetrics)
      .catch((e) => setError(e.message));
  }, [apiBase]);

  if (error) return (
    <section className="section"><div className="text-red-600">Erreur lors du chargement des métriques : {error}</div></section>
  );
  if (!metrics) return (
    <section className="section"><div className="text-slate-500">Chargement des métriques...</div></section>
  );

  return (
    <section className="section section-alt">
      <div className="mx-auto max-w-3xl px-4 sm:px-6 lg:px-8">
        <div className="section-title">
          <span className="tag-accent">Métriques du modèle</span>
          <h2 className="mt-3 text-3xl md:text-4xl font-extrabold tracking-tight">Performance</h2>
          <div className="title-underline" />
        </div>
        <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-6">
          {typeof metrics.best_val_accuracy === "number" && (
            <div className="card">
              <div className="font-semibold">Précision</div>
              <div className="text-2xl font-mono text-blue-700">{(metrics.best_val_accuracy * 100).toFixed(1)}%</div>
            </div>
          )}
          {typeof metrics.media_count === "number" && (
            <div className="card">
              <div className="font-semibold">Médias analysés</div>
              <div className="text-2xl font-mono text-green-700">{metrics.media_count.toLocaleString()}</div>
            </div>
          )}
          {typeof metrics.model_count === "number" && (
            <div className="card">
              <div className="font-semibold">Modèles</div>
              <div className="text-2xl font-mono text-rose-700">{metrics.model_count}</div>
            </div>
          )}
        </div>
        <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="card">
            <div className="font-semibold">Architecture</div>
            <div className="text-base text-slate-700">{metrics.model_architecture}</div>
          </div>
          <div className="card">
            <div className="font-semibold">Shapes d'entrée</div>
            <ul className="text-base text-slate-700">
              {metrics.input_shapes && Object.entries(metrics.input_shapes).map(([k, v]) => (
                <li key={k}><span className="font-mono text-slate-600">{k}</span> : <span className="font-mono">{String(v)}</span></li>
              ))}
            </ul>
          </div>
        </div>
        <div className="mt-8 card">
          <div className="font-semibold mb-2">Détails bruts</div>
          <pre className="text-xs overflow-auto bg-slate-50 rounded p-2 border border-slate-100">{JSON.stringify(metrics, null, 2)}</pre>
        </div>
      </div>
    </section>
  );
}

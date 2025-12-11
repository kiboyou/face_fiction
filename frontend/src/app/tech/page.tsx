"use client";

import { OrbitControls } from "@react-three/drei";
import { Canvas, useFrame } from "@react-three/fiber";
import Link from "next/link";
import { useCallback, useRef } from "react";

export default function TechInfoPage() {
  const copy = useCallback(async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
    } catch {}
  }, []);
  return (
    <section className="section section-soft-bg">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="section-title">
          <span className="tag-accent">Technologies</span>
          <h1 className="mt-3 text-3xl md:text-4xl lg:text-5xl font-extrabold tracking-tight">Stack & Architecture</h1>
          <div className="title-underline" />
          <p className="mt-3 text-slate-700 max-w-2xl mx-auto text-center">
            Un aperçu clair et visuel des technologies qui propulsent FaceFiction.
          </p>
        </div>

        {/* Hero: 3D accent + tech chips */}
  <div className="mt-6 grid md:grid-cols-3 gap-5 items-stretch">
              <div className="card md:col-span-2 card-soft">
            <h2 className="font-semibold text-lg">Aperçu</h2>
            <p className="mt-1 text-slate-700">Frontend moderne, API rapide, MLOps fiable et traçabilité complète pour l'analyse média.</p>
            <div className="mt-3 flex flex-wrap gap-8 items-center">
              <div className="flex flex-wrap gap-2">
                {[
                  "Next.js",
                  "TypeScript",
                  "FastAPI",
                  "Python",
                  "scikit-learn",
                  "MLflow",
                  "DVC",
                  "Docker",
                  "S3",
                ].map((t) => (
                  <span key={t} className="chip-tech">{t}</span>
                ))}
              </div>
            </div>
            <div className="mt-4">
              <Link href="/prediction" className="btn-primary">Tester l'inférence</Link>
            </div>
          </div>
              <div className="card p-0 overflow-hidden" style={{ height: 220 }}>
            <Canvas camera={{ position: [2.8, 2.8, 2.8], fov: 50 }}>
              <ambientLight intensity={0.7} />
              <pointLight position={[6, 6, 6]} intensity={0.9} />
              <SpinningGem />
              <OrbitControls enableZoom={false} enablePan={false} autoRotate autoRotateSpeed={1.2} />
            </Canvas>
          </div>
        </div>

        {/* Architecture + Endpoints (row 1) */}
        <div className="mt-6 grid lg:grid-cols-2 gap-5">
          {/* Architecture */}
          <div className="card card-soft p-5">
            <div className="flex items-start gap-4">
              <div className="feature-icon text-2xl">🏛️</div>
              <div className="flex-1 min-w-0">
                <h3 className="font-semibold text-lg">Architecture</h3>
                <ul className="mt-3 space-y-2 text-slate-700">
                  {[
                    "Frontend: Next.js (App Router, TS)",
                    "API: FastAPI + Uvicorn",
                    "Modélisation: scikit-learn",
                    "Suivi: MLflow",
                    "Orchestration: Docker Compose",
                  ].map((t) => (
                    <li key={t} className="flex items-start gap-3">
                      <span className="icon-valid mt-0.5" aria-hidden>
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><path d="M20 6L9 17l-5-5"/></svg>
                      </span>
                      <span>{t}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>

          {/* Endpoints & Config */}
          <div className="card card-soft p-5">
            <div className="flex items-start gap-4">
              <div className="feature-icon text-2xl">⚙️</div>
              <div className="flex-1 min-w-0">
                <h3 className="font-semibold text-lg">Endpoints & Config</h3>
                <div className="mt-2.5 space-y-2.5">
                  {/* Updated endpoints list to match FastAPI backend */}
                  {[
                    { method: 'GET', path: '/api/health', desc: 'Health check' },
                    { method: 'GET', path: '/api/models', desc: 'List models' },
                    { method: 'GET', path: '/api/metrics', desc: 'Model metrics' },
                    { method: 'POST', path: '/api/predict/upload', desc: 'Predict (upload video)' },
                    { method: 'POST', path: '/api/predict/batch', desc: 'Batch prediction' },
                    { method: 'GET', path: '/api/stats', desc: 'Server stats' },
                    { method: 'GET', path: '/api/reports', desc: 'List reports' },
                    { method: 'GET', path: '/api/reports/[report_name]', desc: 'Download report' },
                  ].map((ep) => (
                    <div key={ep.path} className="p-3 rounded-lg border border-black/5 bg-white flex items-center justify-between gap-3">
                      <div className="flex items-center gap-2 min-w-0">
                        <span className={ep.method === 'GET' ? 'chip-label' : 'chip-role'} aria-hidden>{ep.method}</span>
                        <code className="v-mono text-sm text-slate-700">{ep.path}</code>
                        <span className="hint">{ep.desc}</span>
                      </div>
                      <button className="icon-ghost" title="Copier" onClick={() => copy(`${ep.method} ${ep.path}`)}>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Pipeline (row 2 - full width) */}
        <div className="mt-5">
          <div className="card card-soft p-5">
            <div className="flex items-start gap-4">
              <div className="feature-icon text-2xl">🔗</div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between flex-wrap gap-2">
                  <h3 className="font-semibold text-lg">Pipeline</h3>
                  <span className="inline-flex items-center gap-1 rounded-full bg-indigo-50 text-indigo-700 ring-1 ring-indigo-200 px-2.5 py-1 text-xs font-medium">5 étapes</span>
                </div>
                <div className="mt-3 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                  {[
                    {
                      label: "Prétraitement (resize, normalisation)",
                      icon: (
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                          <rect x="3" y="3" width="7" height="7"/>
                          <rect x="14" y="3" width="7" height="7"/>
                          <path d="M3 14h7v7H3z"/>
                        </svg>
                      ),
                    },
                    {
                      label: "Entraînement (experiments traqués)",
                      icon: (
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                          <path d="M3 3v18"/>
                          <path d="M3 12h18"/>
                          <circle cx="12" cy="12" r="3"/>
                        </svg>
                      ),
                    },
                    {
                      label: "Évaluation (accuracy, recall, F1)",
                      icon: (
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                          <path d="M3 3v18h18"/>
                          <path d="M7 14l3 3 7-7"/>
                        </svg>
                      ),
                    },
                    {
                      label: "Versionning (DVC + S3)",
                      icon: (
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                          <path d="M6 3v18"/>
                          <circle cx="6" cy="7" r="2"/>
                          <circle cx="6" cy="17" r="2"/>
                          <path d="M6 7h8a4 4 0 0 1 4 4v6"/>
                        </svg>
                      ),
                    },
                    {
                      label: "Déploiement (API + UI)",
                      icon: (
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                          <rect x="2" y="4" width="20" height="14" rx="2"/>
                          <path d="M8 20h8"/>
                        </svg>
                      ),
                    },
                  ].map(({ label, icon }, i) => (
                    <div key={label} className="flex items-start gap-3 rounded-xl border border-black/5 bg-white p-3 shadow-sm">
                      <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-white ring-2 ring-indigo-200 text-indigo-700 text-xs font-bold shadow-sm">
                        {i + 1}
                      </div>
                      <div className="flex items-start gap-2 text-slate-700 leading-snug">
                        <span className="text-indigo-600" aria-hidden>{icon}</span>
                        <span>{label}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* System map */}
        <div className="mt-6 grid md:grid-cols-2 gap-5">
          <div className="card p-5">
            <h3 className="font-semibold flex items-center gap-2">
              <span className="text-indigo-600" aria-hidden>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="2" y="4" width="20" height="14" rx="2"/><path d="M8 20h8"/></svg>
              </span>
              Frontend
            </h3>
            <p className="mt-0.5 text-slate-700">Next.js + TypeScript, animations fluides, design soigné, upload et résultats en temps réel.</p>
          </div>
          <div className="card p-5">
            <h3 className="font-semibold flex items-center gap-2">
              <span className="text-indigo-600" aria-hidden>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="7" width="18" height="13" rx="2"/><path d="M8 7V4h8v3"/></svg>
              </span>
              API
            </h3>
            <p className="mt-0.5 text-slate-700">FastAPI servie par Uvicorn, endpoints simples et documentés, orchestrés via Docker.</p>
          </div>
          <div className="card p-5">
            <h3 className="font-semibold flex items-center gap-2">
              <span className="text-indigo-600" aria-hidden>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 8 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 3.6 15a1.65 1.65 0 0 0-1.51-1H2a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 3.6 8a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 8 3.6a1.65 1.65 0 0 0 1-1.51V2a2 2 0 1 1 4 0v.09A1.65 1.65 0 0 0 15 3.6a1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 20.4 8c0 .59.23 1.17.6 1.6"/></svg>
              </span>
              Modèles
            </h3>
            <p className="mt-0.5 text-slate-700">Training scikit-learn, suivi des runs avec MLflow, métriques claires.</p>
          </div>
          <div className="card p-5">
            <h3 className="font-semibold flex items-center gap-2">
              <span className="text-indigo-600" aria-hidden>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 8 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 3.6 15a1.65 1.65 0 0 0-1.51-1H2a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 3.6 8c0 .59.23 1.17.6 1.6"/></svg>
              </span>
              MLOps
            </h3>
            <p className="mt-0.5 text-slate-700">DVC pour versionner les données/modèles, remote S3, reproductibilité.</p>
          </div>
        </div>
      </div>
    </section>
  );
}

function SpinningGem() {
  const ref = useRef<any>(null);
  useFrame((_, delta) => {
    if (ref.current) {
      ref.current.rotation.x += delta * 0.4;
      ref.current.rotation.y += delta * 0.6;
    }
  });
  return (
    <mesh ref={ref} castShadow receiveShadow>
      <icosahedronGeometry args={[1, 0]} />
      <meshStandardMaterial color="#6D28D9" metalness={0.3} roughness={0.25} />
    </mesh>
  );
}

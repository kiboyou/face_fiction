"use client";
import Reveal from "@/components/Reveal";
import { toast } from "@/components/Toaster";
import { useCallback, useEffect, useState } from "react";
import { fetchModels } from "./api";



interface PredictionResult {
  prediction_class: string;
  confidence: number;
  prediction_score?: number;
  model?: string;
  inference_ms?: number;
  timestamp?: string;
  topK?: Array<{ label: string; confidence?: number; prob?: number }>;
  filename?: string;
  warning?: string;
}

export default function PredictionPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [previewType, setPreviewType] = useState<string | null>(null); // 'image' | 'video' | null
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);

  const [imgQuality, setImgQuality] = useState<null | {
    sharp: boolean;
    contrast: boolean;
    bright: boolean;
    centered: boolean;
  }>(null);

  const [models, setModels] = useState<Array<{ name: string; type: string }>>([]);
  const [selectedModel, setSelectedModel] = useState<string>("default");
  const apiBase = "/api";

  useEffect(() => {
    fetchModels().then((data) => {
      if (data && Array.isArray(data.models)) {
        setModels(data.models);
      }
    }).catch(() => {
      setModels([]);
    });
  }, []);

  const onFile = useCallback((f: File | null) => {
    setFile(f);
    setResult(null);
    setError(null);
    setImgQuality(null);
    if (f) {
      const url = URL.createObjectURL(f);
      setPreview(url);
      if (f.type.startsWith("image/")) {
        setPreviewType("image");
      } else if (f.type.startsWith("video/")) {
        setPreviewType("video");
      } else {
        setPreviewType(null);
      }
    } else {
      setPreview(null);
      setPreviewType(null);
    }
  }, []);

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(true);
  };
  const onDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
  };
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFile(e.dataTransfer.files[0]);
    }
  };
  const onChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onFile(e.target.files[0]);
    }
  };

  const submit = async () => {
    if (!file) return;
    setLoading(true);
    setUploadProgress(0);
    setError(null);
    setResult(null);
    try {
      const form = new FormData();
      form.append("video", file); // Use 'video' as field name
      form.append("model_name", selectedModel);
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${apiBase}/predict/upload`, true); // Use correct endpoint
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
          setUploadProgress(Math.round((e.loaded / e.total) * 100));
        }
      };
      xhr.onload = () => {
        setLoading(false);
        if (xhr.status === 200) {
          try {
            const data = JSON.parse(xhr.responseText);
            setResult(data);
            toast("Analyse terminée", "success");
          } catch (err) {
            setError("Erreur de décodage de la réponse.");
          }
        } else {
          // Try to parse error message from backend if possible
          try {
            const errData = JSON.parse(xhr.responseText);
            setError(errData.detail || xhr.responseText || "Erreur lors de l'analyse.");
          } catch {
            setError(xhr.responseText || "Erreur lors de l'analyse.");
          }
        }
      };
      xhr.onerror = () => {
        setLoading(false);
        setError("Erreur réseau ou serveur inaccessible.");
      };
      xhr.send(form);
    } catch (err: any) {
      setLoading(false);
      setError(err.message || "Erreur inconnue");
    }
  };

  const disabled = !file || loading;
  const formatUTC = (iso: string) => {
    const d = new Date(iso);
    const pad = (n: number) => String(n).padStart(2, "0");
    return `${pad(d.getUTCDate())}/${pad(d.getUTCMonth() + 1)}/${pad(d.getUTCFullYear())} ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}`;
  };

  return (
    <>
      <section className="section section-soft-bg">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="section-title">
            <span className="tag-accent">Détection</span>
            <h1 className="mt-3 text-3xl md:text-4xl lg:text-5xl font-extrabold tracking-tight">Détecter un deepfake</h1>
            <div className="title-underline" />
            <p className="mt-3 text-slate-600 max-w-2xl mx-auto text-center">
              Importez une image ou une vidéo pour vérifier l’authenticité du média.
            </p>
          </div>

          {/* Étapes */}
          <div className="mt-6 grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="p-4 rounded-xl border border-black/5 bg-white">
              <div className="flex items-start gap-3">
                <span className="icon-badge" aria-hidden>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                </span>
                <div>
                  <div className="text-sm font-semibold">1. Importer un média</div>
                  <div className="hint">Glissez-déposez ou choisissez une image/vidéo.</div>
                </div>
              </div>
            </div>
            <div className="p-4 rounded-xl border border-black/5 bg-white">
              <div className="flex items-start gap-3">
                <span className="icon-badge badge-secondary" aria-hidden>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
                </span>
                <div>
                  <div className="text-sm font-semibold">2. Lancer l’analyse</div>
                  <div className="hint">Cliquez « Lancer l’analyse » et suivez la progression.</div>
                </div>
              </div>
            </div>
            <div className="p-4 rounded-xl border border-black/5 bg-white">
              <div className="flex items-start gap-3">
                <span className="icon-badge badge-neutral" aria-hidden>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                </span>
                <div>
                  <div className="text-sm font-semibold">3. Consulter le résultat</div>
                  <div className="hint">Le verdict et les métriques s’affichent dans la section « Résultats ».</div>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-8 grid lg:grid-cols-2 gap-8 items-start">
            {/* Upload card */}
            <Reveal>
              <div className="card">
                <div
                  className={`dropzone ${dragging ? "dragover" : ""}`}
                  onDragOver={onDragOver}
                  onDragEnter={onDragOver}
                  onDragLeave={onDragLeave}
                  onDrop={onDrop}
                >
                  <div className="flex items-center justify-between gap-4 flex-wrap">
                    <div>
                      <div className="font-semibold">Téléversez votre média</div>
                      <div className="hint">Glissez-déposez ou sélectionnez une image ou vidéo</div>
                    </div>
                    <label className="btn-secondary cursor-pointer">
                      Choisir un fichier
                      <input type="file" accept="image/*,video/*" onChange={onChange} />
                    </label>
                  </div>
                  {/* Modèle utilisé (sélection) */}
                  <div className="mt-4">
                    <span className="block text-sm font-medium mb-1">Choisir le modèle :</span>
                    <select
                      className="input input-bordered w-full max-w-xs"
                      value={selectedModel}
                      onChange={e => setSelectedModel(e.target.value)}
                      disabled={loading}
                    >
                      <option value="default">Auto (par défaut)</option>
                      {models.map((m) => (
                        <option key={m.name} value={m.name}>{m.name} ({m.type})</option>
                      ))}
                    </select>
                  </div>

                  {preview ? (
                    <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4 items-start">
                      <div className="preview-pane">
                        {previewType === "image" && (
                          // eslint-disable-next-line @next/next/no-img-element
                          <img src={preview} alt="Prévisualisation" className="w-full h-full object-cover" />
                        )}
                        {previewType === "video" && (
                          <video src={preview} controls className="w-full h-full object-cover" />
                        )}
                        {!previewType && (
                          <div className="text-sm text-slate-500">Format non supporté pour la prévisualisation.</div>
                        )}
                      </div>
                      <div>
                        {loading && (
                          <div className="mt-3">
                            <div className="progress">
                              <div className="bar" style={{ width: `${uploadProgress}%` }} />
                            </div>
                            <div className="mt-1 text-xs text-slate-600">Téléversement: {uploadProgress}%</div>
                          </div>
                        )}
                        {error && (
                          <p className="mt-3 text-sm" style={{ color: "var(--color-accent)" }}>
                            {error}
                          </p>
                        )}
                      </div>
                    </div>
                  ) : (
                    <div className="mt-4 hint">
                      Formats acceptés: .jpg, .jpeg, .png, .mp4 — Taille conseillée: 224×224 ou supérieure.
                    </div>
                  )}

                  {preview && (
                    <div className="card-actions">
                      <button className="btn-primary" onClick={submit} disabled={disabled}>
                        {loading ? "Analyse en cours..." : "Lancer l'analyse"}
                      </button>
                      <button
                        className="btn-secondary"
                        onClick={() => {
                          onFile(null);
                          setResult(null);
                          setError(null);
                        }}
                      >
                        Réinitialiser
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </Reveal>

            {/* Conseils & Exemples */}
            <Reveal delay={120}>
              <div className="card">
                <div className="font-semibold text-lg">Conseils & Informations</div>
                <ul className="mt-3 space-y-2 text-slate-700">
                  <li>• Utilisez un média net et bien éclairé.</li>
                  <li>• Centrez le visage/personnage et évitez les artefacts (logos, cadres, textes).</li>
                  <li>• Ce modèle détecte les manipulations — il ne remplace pas une vérification humaine.</li>
                </ul>
                {/* sample-grid supprimé à la demande de l'utilisateur */}
              </div>
            </Reveal>
          </div>

          {/* Résultats */}
          <section id="results" className="section">
            <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
              <div className="section-title">
                <span className="tag-accent">Résultats</span>
                <h2 className="mt-3 text-3xl md:text-4xl font-extrabold tracking-tight">Analyse</h2>
                <div className="title-underline" />
              </div>
              <div className="mt-6">
                {result ? (
                  <Reveal>
                    <div className="result-card typography-bump">
                      <div className="flex items-center justify-between">
                        <div className="font-semibold">Sortie du modèle</div>
                        {result?.prediction_class && <span className="chip-label">{String(result.prediction_class)}</span>}
                      </div>
                      {typeof result?.prediction_score === "number" && (
                        <div className="mt-2">
                          <div className="progress" style={{ background: "#e2e8f0" }}>
                            <div
                              className="bar"
                              style={{ width: `${(result.prediction_score * 100).toFixed(2)}%`, background: "var(--color-primary)" }}
                            />
                          </div>
                          <div className="mt-1 text-xs text-slate-600">Score: {result.prediction_score.toFixed(4)}</div>
                        </div>
                      )}
                      {result?.warning && (
                        <div className="mt-3 p-3 rounded bg-yellow-100 text-yellow-800 font-semibold">
                          ⚠️ {result.warning}
                        </div>
                      )}
                      {/* Top-K dynamique */}
                      {Array.isArray(result?.topK) && result.topK.length > 0 && (
                        <div className="mt-4 grid sm:grid-cols-2 gap-4">
                          <div>
                            <div className="text-sm font-medium">Top‑K</div>
                            <div className="mt-2 space-y-2">
                              {result.topK.map((k: any) => {
                                // Couleur dynamique selon le label
                                let color = "#64748b"; // slate par défaut
                                if (typeof k.label === "string") {
                                  if (k.label.toLowerCase() === "fake") color = "#ef4444"; // rouge
                                  if (k.label.toLowerCase() === "real") color = "#22c55e"; // vert
                                }
                                // Compatibilité : accepte .prob ou .confidence
                                const value = typeof k.prob === "number" ? k.prob : (typeof k.confidence === "number" ? k.confidence : 0);
                                return (
                                  <div key={k.label}>
                                    <div className="flex items-center justify-between text-sm">
                                      <span style={{ color }}>{k.label}</span>
                                      <span style={{ color }}>{Math.round(value * 100)}%</span>
                                    </div>
                                    <div className="progress" style={{ background: "#eef2f7" }}>
                                      <div className="bar" style={{ width: `${Math.round(value * 100)}%`, background: color }} />
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                          <div>
                            <div className="text-sm font-medium">Métadonnées</div>
                            <div className="mt-2 meta-grid meta-col">
                              {result?.model && (
                                <div className="meta-item meta-item--pro">
                                  <span className="icon-badge" aria-hidden>
                                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 12h18"/><path d="M3 6h18"/><path d="M3 18h18"/></svg>
                                  </span>
                                  <div className="flex-1 min-w-0">
                                    <div className="k">Modèle</div>
                                    <div className="v">{String(result.model)}</div>
                                  </div>
                                  <button className="icon-ghost" title="Copier" onClick={() => navigator.clipboard.writeText(String(result.model))}>
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                                  </button>
                                </div>
                              )}
                              {typeof result?.inference_ms === "number" && (
                                <div className="meta-item meta-item--pro">
                                  <span className="icon-badge badge-secondary" aria-hidden>
                                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                                  </span>
                                  <div className="flex-1 min-w-0">
                                    <div className="k">Inférence</div>
                                    <div className="v v-mono">{result.inference_ms} ms</div>
                                  </div>
                                  <button className="icon-ghost" title="Copier" onClick={() => navigator.clipboard.writeText(`${result.inference_ms} ms`)}>
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                                  </button>
                                </div>
                              )}
                              {result?.timestamp && (
                                <div className="meta-item meta-item--pro">
                                  <span className="icon-badge badge-neutral" aria-hidden>
                                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M8 2v4"/><path d="M16 2v4"/><rect x="3" y="4" width="18" height="18" rx="2"/><path d="M3 10h18"/></svg>
                                  </span>
                                  <div className="flex-1 min-w-0">
                                    <div className="k">Date</div>
                                    <div className="v v-mono">{formatUTC(result.timestamp)}</div>
                                  </div>
                                  <button className="icon-ghost" title="Copier" onClick={() => result.timestamp && navigator.clipboard.writeText(formatUTC(result.timestamp))}>
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                                  </button>
                                </div>
                              )}
                              {result?.filename && (
                                <div className="meta-item meta-item--pro">
                                  <span className="icon-badge badge-accent" aria-hidden>
                                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M8 6h.01"/><path d="M16 6h.01"/><path d="M8 18h.01"/><path d="M16 18h.01"/></svg>
                                  </span>
                                  <div className="flex-1 min-w-0">
                                    <div className="k">Fichier</div>
                                    <div className="v v-mono">{result?.filename}</div>
                                  </div>
                                  <button className="icon-ghost" title="Copier" onClick={() => navigator.clipboard.writeText(result.filename || "")}> 
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                                  </button>
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      )}
                      <details className="mt-3">
                        <summary className="cursor-pointer text-sm text-slate-600">Détails bruts</summary>
                        <pre className="mt-2 text-xs overflow-auto">{JSON.stringify(result, null, 2)}</pre>
                      </details>
                      <div className="mt-4 card">
                        <div className="font-semibold">Conseils & Explications</div>
                        <ul className="note-list text-sm">
                          <li>Ce résultat est un indicateur, pas une preuve légale.</li>
                          <li>Si la confiance est faible (&lt; 70%), essayez un média plus net ou une autre séquence.</li>
                          <li>En cas de doute, faites vérifier le média par un expert.</li>
                        </ul>
                      </div>
                    </div>
                  </Reveal>
                ) : (
                  <div className="result-card typography-bump text-center text-slate-400">
                    Aucun résultat à afficher. Veuillez importer un média et lancer l'analyse.
                  </div>
                )}
              </div>
            </div>
          </section>
        </div>
      </section>
    </>
  );
}

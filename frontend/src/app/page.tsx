import ContactSection from "@/components/ContactSection";
import HeroFX from "@/components/HeroFX";
import HeroStats from "@/components/HeroStats";
import Reveal from "@/components/Reveal";
import RotatingImage3D from "@/components/RotatingImage3D";
import Link from "next/link";

export default function Home() {
  return (
    <>
      {/* Hero / Accueil immersive */}
      <section className="hero-wrap">
        <div className="hero-bg" />
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 grid md:grid-cols-2 gap-10 lg:gap-14 items-center py-16 lg:py-20">
          <div>
            <span className="tag-accent">DeepFakes & Vision</span>
            <h1 className="mt-4 text-4xl md:text-5xl lg:text-6xl font-extrabold leading-tight tracking-tight">
              FaceFiction — détection de <span style={{ color: "var(--color-primary)" }}>deepfakes</span>
            </h1>
            <p className="mt-4 text-lg text-slate-400 max-w-xl">
              Plateforme end-to-end pour détecter et analyser les deepfakes : entraînement, tracking d'expériences, API et visualisations d'interprétabilité (Grad‑CAM, SHAP).
            </p>
            <div className="mt-6 flex flex-wrap gap-3">
              <Link href="/prediction" className="btn-primary">Tester un média</Link>
              <Link href="#how" className="btn-secondary">Comment ça marche ?</Link>
            </div>
            {/* Stats strip */}
            <HeroStats />
          </div>
          <div>
            <RotatingImage3D url="https://res.cloudinary.com/dqybzf7bu/image/upload/v1761959624/775ea58ec933153c58ae6b2b04964b08_tgzcj6.jpg" height={660} />
          </div>
        </div>
        <div className="scroll-cue" aria-hidden>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M6 9l6 6 6-6"/></svg>
          <span>Défiler</span>
        </div>
        <HeroFX />
      </section>

      {/* Objectifs */}
        <section className="section section-alt">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <div className="section-title">
              <span className="tag-accent">Pourquoi FaceFiction</span>
              <h2 className="mt-3 text-3xl md:text-4xl lg:text-5xl font-extrabold tracking-tight">Objectifs clés du projet</h2>
              <div className="title-underline" />
            </div>
            <div className="mt-8 mx-auto max-w-5xl lg:max-w-6xl grid gap-8 lg:gap-10 md:grid-cols-3">
              {[ 
                { title: "Collecte & Préprocessing", desc: "Pipeline pour vidéos/frames, extraction de visages et annotation (DFDC, autres jeux)", icon: (
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 7l9-4 9 4-9 4-9-4z"/><path d="M3 17l9 4 9-4"/><path d="M3 12l9 4 9-4"/></svg>
                ) },
                { title: "Modèles avancés", desc: "EfficientNet, ResNet, ViT — architectures pour robustesse spatiale et temporelle", icon: (
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 8 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 3.6 15a1.65 1.65 0 0 0-1.51-1H2a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 3.6 8a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 8 3.6a1.65 1.65 0 0 0 1-1.51V2a2 2 0 1 1 4 0v.09A1.65 1.65 0 0 0 15 3.6a1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 20.4 8c0 .59.23 1.17.6 1.6"/></svg>
                ) },
                { title: "Tracking & MLOps", desc: "Suivi d'expériences avec MLflow, containerisation Docker et orchestration", icon: (
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>
                ) },
                
                { title: "Triage assisté", desc: "Prioriser les contenus à vérifier pour accélérer la modération et la vérification", icon: (
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20 6L9 17l-5-5"/><path d="M12 2v6"/><path d="M9 5h6"/></svg>
                ) },
                { title: "Support à la vérification à distance", desc: "Analyse de médias à distance pour assistance à la vérification d'authenticité", icon: (
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="2" y="4" width="20" height="14" rx="2"/><path d="M8 20h8"/></svg>
                ) },
                { title: "Traçabilité & conformité", desc: "Journal des prédictions, anonymisation et conformité RGPD", icon: (
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2l7 4v6c0 5-3.5 9-7 10-3.5-1-7-5-7-10V6l7-4z"/><path d="M9 12l2 2 4-4"/></svg>
                ) },
              ].map((c, i) => (
                <Reveal key={c.title} delay={i * 80}>
                  <div className="feature-card media">
                    <div className="feature-icon">{c.icon}</div>
                    <div>
                      <h3 className="font-semibold text-lg">{c.title}</h3>
                      <p className="mt-1.5 text-slate-600">{c.desc}</p>
                    </div>
                  </div>
                </Reveal>
              ))}
            </div>
        </div>
      </section>

      {/* Vision */}
      <section className="section section-tight-bottom section-soft-bg">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="section-title">
            <span className="tag-accent">Vision & Impact</span>
            <h2 className="mt-3 text-3xl md:text-4xl lg:text-5xl font-extrabold tracking-tight">Notre ambition</h2>
            <div className="title-underline" />
              <p className="mt-3 text-slate-700 max-w-2xl mx-auto">
                Une IA fiable et transparente pour la détection d'altérations visuelles (deepfakes), conçue pour la recherche, la modération et la vérification d'authenticité.
              </p>
          </div>
          <div className="mt-8 grid md:grid-cols-2 gap-6 items-stretch">
            <Reveal>
              <div className="image-frame">
                <img
                  src="https://res.cloudinary.com/dqybzf7bu/image/upload/v1761959822/51a6820b910bf2eaf2a8404b8bdae205_ke8d1q.jpg"
                  alt="Professionnels de santé utilisant une solution numérique pour assister la prise en charge"
                  loading="lazy"
                  decoding="async"
                  referrerPolicy="no-referrer"
                  className="rounded-xl"
                />
              </div>
            </Reveal>
            <Reveal delay={100}>
                <div className="card card-soft">
                  <h3 className="text-xl md:text-2xl font-bold">Cas d’usage & bénéfices</h3>
                  <ul className="mt-4 space-y-2 benefit-list text-[1.05rem] md:text-[1.1rem] leading-relaxed">
                    {[
                      "Aide à la détection d'altérations visuelles et à la modération de contenu",
                      "Outils d'audit et d'interprétabilité pour expliquer les décisions du modèle",
                      "Évaluations cross-dataset pour garantir robustesse et généralisabilité",
                      "Recommandations pour atténuer les biais et usages à risque",
                      "Intégration API pour vérification média dans des pipelines existants",
                    ].map((item) => (
                      <li key={item} className="flex items-start gap-3">
                        <span className="icon-valid mt-0.5" aria-hidden>
                          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M20 6L9 17l-5-5" />
                          </svg>
                        </span>
                        <span className="leading-relaxed benefit-text">{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>
            </Reveal>
          </div>
        </div>
      </section>

      {/* Comment faire une prédiction */}
  <section id="how" className="section section-alt section-tight-top">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <div className="section-title">
            <span className="tag-accent">Guide</span>
            <h2 className="mt-3 text-3xl md:text-4xl lg:text-5xl font-extrabold tracking-tight">Comment faire une prédiction ?</h2>
            <div className="title-underline" />
            <p className="mt-3 mb-6 text-slate-400 max-w-2xl mx-auto">
              Trois étapes simples pour obtenir une prédiction d'authenticité à partir d'un média (image/vidéo). Le système retourne score, label et visualisations d'interprétabilité.
            </p>
          </div>
            <div className="mt-8 steps-grid">
            {[
              {
                step: 1,
                title: "Télécharger",
                desc: "Chargez un média : image ou courte vidéo. Pour vidéo, des frames sont extraites automatiquement.",
                tips: ["Fournir un extrait représentatif (10-30s)", "Résolution conseillée ≥ 224×224 pour les visages"],
                icon: (
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                    <path d="M16 16.58A5 5 0 0 0 18 7h-1.26A8 8 0 1 0 3 15.25" />
                    <path d="M12 12v9" />
                    <path d="m8 16 4 4 4-4" />
                  </svg>
                ),
              },
              {
                step: 2,
                title: "Analyser",
                desc: "Préprocessing, extraction de faces/frames puis prédiction par le modèle (score + label).",
                tips: ["Formats pris en charge: .mp4, .mov, .jpg, .png", "La confidentialité: les médias peuvent être anonymisés avant stockage"],
                icon: (
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                    <path d="M12 2v20" />
                    <path d="M2 12h20" />
                    <circle cx="12" cy="12" r="3" />
                  </svg>
                ),
              },
              {
                step: 3,
                title: "Résultat",
                desc: "Score d'authenticité, label (Real / Fake) et heatmap d'interprétabilité (Grad‑CAM).",
                tips: ["Interprétations à combiner avec audit humain", "Stockage et traçabilité configurable via MLflow"],
                icon: (
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                    <path d="M22 4 12 14.01l-3-3" />
                  </svg>
                ),
              },
            ].map((s, i, arr) => (
              <Reveal key={s.step} delay={i * 80}>
                <div className="step-card relative">
                  <div className="step-badge">{s.step}</div>
                  <div className="step-title">
                    <span className="step-icon">{s.icon}</span>
                    <h3 className="font-semibold text-xl">{s.title}</h3>
                  </div>
                  <p className="mt-2 text-slate-700 text-[0.975rem] md:text-base">{s.desc}</p>
                  {Array.isArray((s as any).tips) && (
                    <ul className="mt-3 note-list text-sm">
                      {(s as any).tips.map((t: string) => (
                        <li key={t}>{t}</li>
                      ))}
                    </ul>
                  )}
                  {i < arr.length - 1 && (
                    <>
                      {/* Desktop/tablet: horizontal connector to next card */}
                      <div className="hidden md:block absolute top-1/2 -translate-y-1/2 right-[-22px] w-10 h-[2px] bg-slate-200" aria-hidden />
                      {/* Mobile: vertical connector to next card */}
                      <div className="md:hidden absolute left-1/2 -translate-x-1/2 bottom-[-18px] h-6 w-[2px] bg-slate-200" aria-hidden />
                    </>
                  )}
                </div>
              </Reveal>
            ))}
          </div>
          <div className="mt-8 text-center">
            <Link href="/prediction" className="btn-primary">Essayer maintenant</Link>
          </div>
        </div>
      </section>

      {/* Équipe */}
      <section id="team" className="section">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="section-title">
            <span className="tag-accent">Équipe</span>
            <h2 className="mt-3 text-3xl md:text-4xl lg:text-5xl font-extrabold tracking-tight">Rencontrez l’équipe</h2>
            <div className="title-underline" />
            <p className="mt-3 text-slate-600 max-w-2xl mx-auto">Pluridisciplinaire, rigoureuse et soucieuse de l'impact social et éthique.</p>
          </div>
          <div className="mt-8 grid sm:grid-cols-2 md:grid-cols-3 gap-6">
            {[ 
              { id: 1, name: "Kiboyou Mohamed OUATTARA", role: "Etudiant en Machine Learning", spec: "AI / Front & API ", img: "https://res.cloudinary.com/dqybzf7bu/image/upload/v1761610805/WhatsApp_Image_2025-04-09_at_00.48.51_aqkq9a.jpg", linkedin: "https://www.linkedin.com/in/kiboyou-mohamed-ouattara-4131bb220/", github: "https://github.com/kiboyou" },
              { id: 2, name: "Jean Christian AHIKPA", role: "Etudiant en IA Engineer", spec: "Vision par ordinateur ", img: "https://res.cloudinary.com/dqybzf7bu/image/upload/v1761768905/Jean_Christian_AHIKPA_SummerCamp2025_Talan_rw26is.jpg", linkedin: "https://www.linkedin.com/in/jean-christian-ahikpa/", github: "https://github.com/ahikpa" },
              { id: 3, name: "Acobe ange BONI", role: "Etudiant en Data Engineer", spec: "Pipelines & Big Data", img: "https://res.cloudinary.com/dqybzf7bu/image/upload/v1761769425/A_252_dbbpuh.jpg", linkedin: "https://www.linkedin.com/in/acobe-ange-ulrich-boni/", github: "https://github.com/membre3" },
            ].map((m, idx) => (
              <Reveal key={m.id} delay={idx * 80}>
                <article className="group rounded-3xl border border-black/5 bg-white/70 backdrop-blur-sm shadow-sm hover:shadow-md transition-all">
                  <div className="p-6 flex flex-col items-center text-center relative">
                    <div className="relative">
                      <span className="absolute inset-0 -z-0 rounded-full bg-gradient-to-tr from-sky-500 to-indigo-500 blur opacity-40 group-hover:opacity-60 transition-opacity" aria-hidden />
                      <div className="relative p-[3px] rounded-full bg-gradient-to-tr from-sky-500 to-indigo-500">
                        <img
                          src={m.img}
                          alt={m.name}
                          loading="lazy"
                          decoding="async"
                          referrerPolicy="no-referrer"
                          className="h-40 w-40 rounded-full object-cover bg-white shadow-sm"
                        />
                      </div>
                    </div>
                    <h3 className="mt-4 font-semibold text-lg">{m.name}</h3>
                    <span className="mt-1 inline-flex items-center gap-1 rounded-full bg-sky-50 text-sky-700 ring-1 ring-sky-200 px-3 py-1 text-xs font-medium">
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
                        <path d="M4 4h16v4H4zM4 10h10v4H4zM4 16h7v4H4z" />
                      </svg>
                      {m.role}
                    </span>
                    <div className="mt-1.5 text-sm text-slate-600">{m.spec}</div>
                    <div className="mt-4 flex items-center gap-2">
                      <a className="inline-flex h-9 w-9 items-center justify-center rounded-full bg-white text-slate-700 shadow-sm ring-1 ring-slate-200 hover:bg-slate-50 transition" href={m.linkedin || "#"} target="_blank" rel="noreferrer" aria-label="LinkedIn" title="LinkedIn">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
                          <path d="M4.98 3.5A2.5 2.5 0 1 1 0 3.5a2.5 2.5 0 0 1 4.98 0zM.5 8.5h4.9V24H.5V8.5zM9 8.5h4.7v2.1h.1c.7-1.2 2.5-2.5 5.1-2.5 5.5 0 6.5 3.6 6.5 8.2V24h-4.9v-6.8c0-1.6 0-3.7-2.3-3.7-2.3 0-2.7 1.8-2.7 3.6V24H9V8.5z"/>
                        </svg>
                      </a>
                      <a className="inline-flex h-9 w-9 items-center justify-center rounded-full bg-white text-slate-700 shadow-sm ring-1 ring-slate-200 hover:bg-slate-50 transition" href={m.github || "#"} target="_blank" rel="noreferrer" aria-label="GitHub" title="GitHub">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
                          <path d="M12 2C6.48 2 2 6.58 2 12.26c0 4.5 2.87 8.31 6.84 9.66.5.1.68-.22.68-.49 0-.24-.01-.87-.01-1.71-2.78.62-3.37-1.37-3.37-1.37-.45-1.18-1.11-1.49-1.11-1.49-.91-.64.07-.63.07-.63 1 .07 1.52 1.05 1.52 1.05 .9 1.57 2.36 1.12 2.94.86.09-.67.35-1.12.63-1.38-2.22-.26-4.56-1.14-4.56-5.07 0-1.12.39-2.03 1.03-2.75-.1-.26-.45-1.3.1-2.71 0 0 .84-.27 2.75 1.05.8-.23 1.66-.35 2.52-.35s1.72.12 2.52.35c1.91-1.32 2.75-1.05 2.75-1.05.55 1.41.2 2.45.1 2.71.64.72 1.03 1.63 1.03 2.75 0 3.94-2.34 4.81-4.57 5.07.36.32.68.95.68 1.92 0 1.39-.01 2.51-.01 2.85 0 .27.18.6.69.49A10.04 10.04 0 0 0 22 12.26C22 6.58 17.52 2 12 2z"/>
                        </svg>
                      </a>
                    </div>
                  </div>
                </article>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      {/* Contact */}
      <ContactSection />
    </>
  );
}

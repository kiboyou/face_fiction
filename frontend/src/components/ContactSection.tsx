"use client";

import Reveal from "@/components/Reveal";
import { useRef, useState } from "react";

function SimpleModal({ open, title, onClose, children }: { open: boolean; title: string; onClose: () => void; children: React.ReactNode }) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/40" onClick={onClose} aria-hidden />
      <div className="relative z-10 w-full max-w-md rounded-2xl bg-white p-6 shadow-lg">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold">{title}</h3>
          <button onClick={onClose} className="icon-ghost" aria-label="Fermer">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
          </button>
        </div>
        <div className="mt-4">{children}</div>
      </div>
    </div>
  );
}

export default function ContactSection() {
  const [sent, setSent] = useState(false);
  const [successOpen, setSuccessOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const formRef = useRef<HTMLFormElement>(null);

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    const fd = new FormData(e.currentTarget);
    const body = {
      name: fd.get("name"),
      email: fd.get("email"),
      phone: fd.get("phone"),
      message: fd.get("message"),
    } as Record<string, any>;
    try {
      const res = await fetch("/api/contact", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(await res.text());
      setSent(true);
      setSuccessOpen(true);
      if (formRef.current) formRef.current.reset();
    } catch (err: any) {
      setError("Erreur lors de l'envoi. Merci de réessayer.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section id="contact" className="section section-alt">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="section-title">
          <span className="tag-accent">Contact</span>
          <h2 className="mt-3 text-3xl md:text-4xl lg:text-5xl font-extrabold tracking-tight">Nous contacter</h2>
          <div className="title-underline" />
          <p className="mt-3 text-slate-600 max-w-2xl mx-auto">Écrivez-nous pour toute question ou idée. Nous répondons sous 24–48h.</p>
        </div>
        <Reveal>
          {sent ? (
            <div className="mt-8 rounded-2xl border border-green-200 bg-green-50 text-green-800 p-6">
              Merci pour votre message ! Nous vous répondrons sous 24–48h.
            </div>
          ) : (
            <div className="mt-8 grid gap-6 md:grid-cols-5">
              {/* Info card */}
              <div className="md:col-span-2 rounded-2xl border border-slate-200 bg-white/70 p-6 backdrop-blur-sm">
                <h3 className="font-semibold text-lg">Coordonnées</h3>
                <p className="mt-2 text-sm text-slate-600">Écrivez-nous pour toute question ou idée. Nous répondons sous 24–48h.</p>
                <div className="mt-4 space-y-3 text-sm">
                  <div className="flex items-center gap-3"><span className="inline-flex h-9 w-9 items-center justify-center rounded-xl bg-blue-100 text-blue-700">📧</span> okm_baau_ajc@gmail.com</div>
                  <div className="flex items-center gap-3"><span className="inline-flex h-9 w-9 items-center justify-center rounded-xl bg-blue-100 text-blue-700">📞</span> +225 00 00 00 00 / 00 00 00 00</div>
                </div>
                <div className="mt-6 h-px bg-gradient-to-r from-transparent via-slate-200 to-transparent" />
                <p className="mt-6 text-xs text-slate-500">Vos informations ne sont utilisées que pour vous répondre.</p>
              </div>

              {/* Form card */}
              <form ref={formRef} onSubmit={handleSubmit} className="md:col-span-3 grid gap-4 rounded-2xl border border-slate-200 bg-white/70 p-6 shadow backdrop-blur-sm">
                <div className="grid gap-4 sm:grid-cols-2">
                  <input required name="name" className="bg-white/60 border border-slate-200 rounded-xl px-4 py-3 outline-none focus:ring-2 focus:ring-blue-300" placeholder="Nom & Prénom" />
                  <input required name="email" type="email" className="bg-white/60 border border-slate-200 rounded-xl px-4 py-3 outline-none focus:ring-2 focus:ring-blue-300" placeholder="Email" />
                </div>
                <div>
                  <input name="phone" className="w-full bg-white/60 border border-slate-200 rounded-xl px-4 py-3 outline-none focus:ring-2 focus:ring-blue-300" placeholder="Téléphone (optionnel)" />
                </div>
                <textarea name="message" rows={6} className="bg-white/60 border border-slate-200 rounded-xl px-4 py-3 outline-none focus:ring-2 focus:ring-blue-300" placeholder="Votre message" />
                {error && <div className="text-red-600 text-sm">{error}</div>}
                <div className="flex items-center gap-3">
                  <button type="submit" disabled={loading} className="px-5 py-3 rounded-2xl btn-primary text-white font-semibold transition w-fit disabled:opacity-60">{loading ? "Envoi..." : "Envoyer"}</button>
                  <span className="text-xs text-slate-500">Temps de réponse moyen: 24h</span>
                </div>
              </form>
            </div>
          )}
        </Reveal>

        <SimpleModal open={successOpen} title="Succès" onClose={() => setSuccessOpen(false)}>
          <div className="flex flex-col items-start gap-3">
            <p className="text-green-700 font-medium">Votre message a été envoyé avec succès. Merci !</p>
            <button onClick={() => setSuccessOpen(false)} className="px-4 py-2 rounded-xl bg-blue-500 text-white font-semibold hover:bg-blue-600 transition">Fermer</button>
          </div>
        </SimpleModal>
      </div>
    </section>
  );
}

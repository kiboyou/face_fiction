import Footer from "@/components/Footer";
import GlobalInteractions from "@/components/GlobalInteractions";
import Navbar from "@/components/Navbar";
import Toaster from "@/components/Toaster";
import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Script from "next/script";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "FaceFiction — Détection de deepfakes",
  description:
    "Plateforme end-to-end pour détecter, tracer et expliquer les deepfakes — entraînement, tracking MLflow, API et visualisations d'interprétabilité.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="fr">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        {/* Noscript fallback to avoid a blank screen if JS is disabled */}
        <noscript>
          <div style={{
            position: "fixed",
            top: 0,
            left: 0,
            right: 0,
            padding: "10px 12px",
            background: "#111827",
            color: "#fff",
            textAlign: "center",
            zIndex: 10000
          }}>
            JavaScript est désactivé dans votre navigateur. Activez‑le pour afficher l’interface.
          </div>
        </noscript>
        <a href="#content" className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-50 bg-white text-black px-3 py-2 rounded-md shadow">Aller au contenu</a>
        <Navbar />
        <main id="content">{children}</main>
        <Footer />
        <GlobalInteractions />
        <Toaster />
        {process.env.NODE_ENV === "development" && (
          <Script
            id="silence-react-devtools-semver"
            strategy="beforeInteractive"
            dangerouslySetInnerHTML={{
              __html:
                "window.addEventListener('error',function(e){try{var f=(e&&e.filename)||'';var s=(e&&e.error&&e.error.stack)||'';if((typeof f==='string'&&f.includes('react_devtools_backend'))||(typeof s==='string'&&s.includes('react_devtools_backend'))){e.preventDefault();}}catch(_){}},true);",
            }}
          />
        )}
        {process.env.NODE_ENV === "development" && (
          <Script
            id="disable-react-devtools-if-broken"
            strategy="beforeInteractive"
            dangerouslySetInnerHTML={{
              __html:
                "try{var h=window.__REACT_DEVTOOLS_GLOBAL_HOOK__;if(h){h.isDisabled=true;var rr=h.registerRenderer; if(typeof rr==='function'){h.registerRenderer=function(){try{return rr.apply(this,arguments)}catch(e){return undefined}}}}else{Object.defineProperty(window,'__REACT_DEVTOOLS_GLOBAL_HOOK__',{value:{isDisabled:true,supportsFiber:true,renderers:new Map(),on:function(){},off:function(){},emit:function(){},inject:function(){},getFiberRoots:function(){return new Map();}},configurable:true});}}catch(_){}",
            }}
          />
        )}
      </body>
    </html>
  );
}

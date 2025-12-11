"use client";

import dynamic from "next/dynamic";

// Client-only dynamic import of the rich 3D scene
const HeroScene = dynamic(() => import("@/components/HeroScene"), { ssr: false });

export default function HeroVisual() {
  return <HeroScene />;
}

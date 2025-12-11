"use client";

import { ContactShadows, Environment, Float, OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { Suspense, useEffect, useMemo, useState } from "react";
import { Texture, TextureLoader } from "three";

function Shape() {
  return (
    <Float speed={1.2} rotationIntensity={0.6} floatIntensity={0.8}>
      <mesh>
        <icosahedronGeometry args={[1.4, 1]} />
        <meshStandardMaterial
          color="#88C9FF"
          roughness={0.25}
          metalness={0.35}
        />
      </mesh>
    </Float>
  );
}

function AccentRing() {
  return (
    <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, -0.1, 0]}>
      <torusGeometry args={[2.2, 0.04, 16, 128]} />
      <meshStandardMaterial color="#34d399" roughness={0.4} metalness={0.2} />
    </mesh>
  );
}

function SkinPatch() {
  const fragmentShader = useMemo(
    () => `
      precision highp float;
      varying vec2 vUv;

      float hash(vec2 p){
        p = fract(p*vec2(123.34, 345.45));
        p += dot(p, p+34.345);
        return fract(p.x*p.y);
      }

      void main(){
        vec2 uv = vUv;
        vec3 skin = vec3(1.0, 0.86, 0.80);
        float r = distance(uv, vec2(0.5));
        float lesionMask = smoothstep(0.24, 0.18, r);
        vec3 lesion = mix(vec3(0.45,0.20,0.12), vec3(0.65,0.35,0.25), smoothstep(0.0,1.0,r*4.0));
        vec3 col = mix(skin, lesion, lesionMask*0.85);
        // sparse pores / speckles
        vec2 g = floor(uv*vec2(140.0,140.0));
        float rnd = hash(g);
        float s = step(0.995, rnd) * 0.2;
        col *= 1.0 - s;
        // subtle vignette
        float v = smoothstep(0.9, 0.5, r);
        col *= 1.0 - v*0.05;
        gl_FragColor = vec4(col, 1.0);
      }
    `,
    []
  );

  const vertexShader = useMemo(
    () => `
      varying vec2 vUv;
      void main(){
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    []
  );

  return (
    <Float speed={0.6} rotationIntensity={0.2} floatIntensity={0.2}>
      <mesh rotation={[-0.5, 0.25, 0]} position={[0, -0.35, 0]}>
        <planeGeometry args={[3.6, 3.6, 1, 1]} />
        {/* @ts-ignore - three-stdlib intrinsic */}
        <shaderMaterial vertexShader={vertexShader} fragmentShader={fragmentShader} />
      </mesh>
    </Float>
  );
}

function TextureSkin() {
  const [tex, setTex] = useState<Texture | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let alive = true;
    const loader = new TextureLoader();
    loader.load(
      "/textures/skin.jpg",
      (t) => {
        if (!alive) return;
        t.anisotropy = 4;
        setTex(t);
      },
      undefined,
      () => {
        if (!alive) return;
        setFailed(true);
      }
    );
    return () => {
      alive = false;
    };
  }, []);
  return (
    <Float speed={0.5} rotationIntensity={0.15} floatIntensity={0.15}>
      <mesh rotation={[-0.5, 0.25, 0]} position={[0, -0.35, 0]}>
        <planeGeometry args={[3.6, 3.6, 1, 1]} />
        {tex && <meshStandardMaterial map={tex} roughness={0.6} metalness={0.05} />}
        {!tex && (
          // Graceful fallback: flat skin-like color if texture missing
          <meshStandardMaterial color="#F3D0C8" roughness={0.7} metalness={0.03} />
        )}
      </mesh>
    </Float>
  );
}

// Model mode removed as per requirement

export default function HeroScene() {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    try {
      const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
      setReduced(mq.matches);
      const handler = (e: MediaQueryListEvent) => setReduced(e.matches);
      mq.addEventListener?.("change", handler);
      return () => mq.removeEventListener?.("change", handler);
    } catch {}
  }, []);
  return (
    <div style={{ width: "100%", height: 420 }}>
      <Canvas camera={{ position: [2.2, 1.8, 3.2], fov: 48 }}>
        <ambientLight intensity={0.7} />
        <directionalLight position={[5, 5, 5]} intensity={1.1} />
        <Suspense fallback={null}>
          <Shape />
          <AccentRing />
          <TextureSkin />
          <Environment preset="city" />
        </Suspense>
        <ContactShadows position={[0, -1.2, 0]} opacity={0.3} scale={6} blur={2.4} far={4.2} />
        <OrbitControls enableZoom={false} enablePan={false} autoRotate={!reduced} autoRotateSpeed={0.5} />
      </Canvas>
    </div>
  );
}

"use client";

import { Canvas, useFrame } from "@react-three/fiber";
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";

function RotatingImageMesh({ url, fallbackUrl }: { url: string; fallbackUrl?: string }) {
  const ref = useRef<any>(null);
  const [texture, setTexture] = useState<THREE.Texture | null>(null);

  useEffect(() => {
    let cancelled = false;
    const loader = new THREE.TextureLoader();
    loader.setCrossOrigin("anonymous");
    const load = (u: string, onFail?: () => void) => {
      loader.load(
        u,
        (tex) => {
          if (!cancelled) {
            (tex as any).colorSpace = (THREE as any).SRGBColorSpace || (THREE as any).sRGBEncoding;
            setTexture(tex);
          }
        },
        undefined,
        () => {
          if (onFail) onFail();
        }
      );
    };
    load(url, () => (fallbackUrl ? load(fallbackUrl) : setTexture(null)));
    return () => {
      cancelled = true;
    };
  }, [url, fallbackUrl]);

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    if (ref.current) {
      ref.current.rotation.y = Math.sin(t * 0.6) * 0.2;
      ref.current.position.y = Math.sin(t * 0.8) * 0.03;
    }
  });

  if (!texture) return null;
  const aspect = (texture.image as any)?.width
    ? (texture.image as any).height / (texture.image as any).width
    : 2 / 3;

  return (
    <mesh ref={ref} position={[0, 0, 0]}>
      <planeGeometry args={[2.6, 2.6 * aspect]} />
      <meshBasicMaterial map={texture} transparent />
    </mesh>
  );
}

export default function RotatingImage3D({
  url,
  fallbackUrl = "https://res.cloudinary.com/dqybzf7bu/image/upload/v1761609580/1f3b44c84f9b1d7e2568647fb80327be_fdendm.jpg",
  height = 280,
  className,
}: {
  url: string;
  fallbackUrl?: string;
  height?: number;
  className?: string;
}) {
  return (
    <div
      className={`card p-0 overflow-hidden rounded-2xl shadow-lg ${className ?? ""}`}
      style={{ height }}
    >
      <Canvas camera={{ position: [0, 0, 3], fov: 50 }}>
        <RotatingImageMesh url={url} fallbackUrl={fallbackUrl} />
      </Canvas>
    </div>
  );
}

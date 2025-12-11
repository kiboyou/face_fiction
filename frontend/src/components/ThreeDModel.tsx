"use client";

import { useEffect, useRef } from "react";

/**
 * Lightweight 3D-like animated hero element (CSS-only blob + parallax)
 * Serves as a tasteful placeholder without adding three.js.
 */
export default function ThreeDModel() {
	const containerRef = useRef<HTMLDivElement | null>(null);

	useEffect(() => {
		const el = containerRef.current;
		if (!el) return;
		const onMove = (e: MouseEvent) => {
			const rect = el.getBoundingClientRect();
			const x = (e.clientX - rect.left) / rect.width - 0.5;
			const y = (e.clientY - rect.top) / rect.height - 0.5;
			el.style.setProperty("--rx", `${y * -6}deg`);
			el.style.setProperty("--ry", `${x * 6}deg`);
		};
		el.addEventListener("mousemove", onMove);
		return () => el.removeEventListener("mousemove", onMove);
	}, []);

	return (
		<div
			ref={containerRef}
			className="relative w-full max-w-xl aspect-square mx-auto"
			style={{ perspective: "1000px" }}
		>
			<div
				className="tilt relative w-full h-full rounded-[28px] overflow-hidden"
				style={{ transform: "rotateX(var(--rx, 0)) rotateY(var(--ry, 0))" }}
			>
				<div className="absolute inset-0" style={{ background: "radial-gradient(140% 100% at 10% 10%, #ffffff 0%, #eaf3ff 45%, #d8f7ea 100%)" }} />
				<div className="blob -top-12 -left-12" />
				<div className="blob -bottom-16 -right-10" />
				<div className="absolute inset-0 grid place-items-center">
					<div className="rounded-2xl px-5 py-3 tag-accent">
						Visualisation dermatologique 3D
					</div>
				</div>
			</div>
		</div>
	);
}


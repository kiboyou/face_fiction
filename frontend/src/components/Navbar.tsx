"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

function NavItem({ href, children, onClick }: { href: string; children: React.ReactNode; onClick?: () => void }) {
	const pathname = usePathname();
	const active = pathname === href || (href !== "/" && pathname.startsWith(href));
	return (
		<Link href={href} className={`nav-link ${active ? "active" : ""}`} onClick={onClick}>{children}</Link>
	);
}

export default function Navbar() {
	const [open, setOpen] = useState(false);
	const [scrolled, setScrolled] = useState(false);
  const pathname = usePathname();

	useEffect(() => {
		const onScroll = () => {
			setScrolled(window.scrollY > 8);
		};
		onScroll();
		window.addEventListener("scroll", onScroll, { passive: true });
		return () => window.removeEventListener("scroll", onScroll);
	}, []);

  // Close mobile menu on route change
  useEffect(() => {
    if (open) setOpen(false);
  }, [pathname]);

	return (
		<header className={`glass-nav ${scrolled ? "scrolled" : ""}`} role="banner">
			<nav className="nav-shell mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
				<div className="flex items-center gap-2">
					<Link href="/" className="brand-link tracking-tight text-xl md:text-2xl" aria-label="FaceFiction - Accueil">
						FaceFiction
					</Link>
				</div>

				<button
					aria-label="Ouvrir le menu"
					className="sm:hidden inline-flex items-center justify-center w-10 h-10 rounded-md border border-black/10 bg-white/60"
					aria-expanded={open}
					aria-controls="mobile-menu"
					onClick={() => setOpen((v) => !v)}
				>
					<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
						<path d="M3 12h18M3 6h18M3 18h18" />
					</svg>
				</button>

				<div className="hidden sm:flex items-center gap-7">
					<NavItem href="/">Accueil</NavItem>
					<NavItem href="/prediction">Prédiction</NavItem>
					<NavItem href="/performance">Performances</NavItem>
					<NavItem href="/tech">Tech</NavItem>
					<Link href="/prediction" className="btn-primary">Essayer maintenant</Link>
				</div>
			</nav>

			{/* Mobile overlay & sheet */}
			<div className="sm:hidden">
				{open && <div className="mobile-overlay" onClick={() => setOpen(false)} />}
				<div id="mobile-menu" className={`mobile-sheet ${open ? "open" : ""}`}>
					<div className="px-4 py-4 flex flex-col gap-3">
						<NavItem href="/" onClick={() => setOpen(false)}>Accueil</NavItem>
						<NavItem href="/prediction" onClick={() => setOpen(false)}>Prédiction</NavItem>
						<NavItem href="/performance" onClick={() => setOpen(false)}>Performances</NavItem>
						<NavItem href="/tech" onClick={() => setOpen(false)}>Tech</NavItem>
						<Link href="/prediction" className="btn-primary text-center" onClick={() => setOpen(false)}>
							Essayer maintenant
						</Link>
					</div>
				</div>
			</div>
		</header>
	);
}


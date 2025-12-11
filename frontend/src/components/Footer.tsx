export default function Footer() {
  return (
    <footer className="footer-sticky" style={{ background: 'var(--color-primary)' }}>
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8 flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-6 h-6 rounded-md bg-white/90" />
          <p className="text-white/90">© {new Date().getFullYear()} FaceFiction. Tous droits réservés.</p>
        </div>
        <div className="flex items-center gap-4">
          <a href="#" className="text-white/90 hover:text-white transition">Mentions légales</a>
          <a href="#" className="text-white/90 hover:text-white transition">Confidentialité</a>
          <a href="#" className="text-white/90 hover:text-white transition">Contact</a>
        </div>
      </div>
    </footer>
  );
}

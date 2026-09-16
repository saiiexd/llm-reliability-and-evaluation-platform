import { useState } from "react";
import { NavLink, Outlet } from "react-router-dom";

const NAV_ITEMS = [
  { to: "/", label: "Overview", end: true },
  { to: "/experiments", label: "Experiments" },
  { to: "/reliability", label: "Reliability" },
  { to: "/rag", label: "RAG Diagnostics" },
  { to: "/evaluators", label: "Evaluators" },
  { to: "/robustness", label: "Robustness" },
];

function NavLinks({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <>
      {NAV_ITEMS.map((item) => (
        <NavLink
          key={item.to}
          to={item.to}
          end={item.end}
          onClick={onNavigate}
          className={({ isActive }) =>
            `block border-b px-1 py-2 text-sm tracking-tight sm:inline-block sm:border-b-0 sm:py-0 ${
              isActive
                ? "border-ink font-medium text-ink sm:border-b-2"
                : "border-transparent text-ink-soft hover:text-ink"
            }`
          }
        >
          {item.label}
        </NavLink>
      ))}
    </>
  );
}

export function Layout() {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <div className="min-h-screen bg-paper text-ink">
      <header className="border-b border-line bg-paper">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-4 sm:px-6">
          <NavLink to="/" className="flex items-baseline gap-2">
            <span className="font-serif text-lg font-medium">LLM Reliability Console</span>
            <span className="hidden font-mono text-xs text-ink-soft sm:inline">research instrument</span>
          </NavLink>
          <nav className="hidden items-center gap-6 sm:flex">
            <NavLinks />
          </nav>
          <button
            type="button"
            className="border border-line px-3 py-1 text-sm sm:hidden"
            aria-expanded={menuOpen}
            aria-controls="mobile-nav"
            onClick={() => setMenuOpen((open) => !open)}
          >
            Menu
          </button>
        </div>
        {menuOpen && (
          <nav id="mobile-nav" className="border-t border-line px-4 py-2 sm:hidden">
            <NavLinks onNavigate={() => setMenuOpen(false)} />
          </nav>
        )}
      </header>
      <main className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
        <Outlet />
      </main>
      <footer className="border-t border-line px-4 py-6 text-xs text-ink-soft sm:px-6">
        <div className="mx-auto max-w-6xl">
          A research and engineering instrument for measuring, comparing, and diagnosing LLM
          behavior. Not a chatbot, not a generic dashboard.
        </div>
      </footer>
    </div>
  );
}

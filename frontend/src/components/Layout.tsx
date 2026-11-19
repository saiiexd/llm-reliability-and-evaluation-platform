import { useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { ConnectionBanner, ConnectionIndicator } from "./ConnectionBanner";

// The four clear primary sections specified by the academic research design
const NAV_ITEMS = [
  { to: "/", label: "Overview", end: true },
  { to: "/experiments", label: "Experiments" },
  { to: "/results", label: "Results" },
  { to: "/analysis", label: "Analysis" },
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
            `block border-b px-2 py-2 text-sm tracking-tight transition-colors sm:inline-block sm:border-b-0 sm:py-0 ${
              isActive
                ? "border-ink font-semibold text-ink sm:border-b-2"
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
          <NavLink to="/" className="flex items-baseline gap-2.5">
            <span className="font-serif text-lg font-semibold tracking-tight">LLM Reliability</span>
            <span className="hidden font-mono text-[11px] uppercase tracking-wider text-ink-soft sm:inline">
              Research Platform
            </span>
          </NavLink>
          <div className="hidden items-center gap-8 sm:flex">
            <nav className="flex items-center gap-6">
              <NavLinks />
            </nav>
            <div className="border-l border-line pl-6">
              <ConnectionIndicator />
            </div>
          </div>
          <div className="flex items-center gap-2 sm:hidden">
            <ConnectionIndicator />
            <button
              type="button"
              className="border border-line px-3 py-1 text-sm font-mono"
              aria-expanded={menuOpen}
              aria-controls="mobile-nav"
              onClick={() => setMenuOpen((open) => !open)}
            >
              Menu
            </button>
          </div>
        </div>
        {menuOpen && (
          <nav id="mobile-nav" className="border-t border-line px-4 py-3 sm:hidden">
            <NavLinks onNavigate={() => setMenuOpen(false)} />
          </nav>
        )}
      </header>
      <main className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
        <ConnectionBanner />
        <Outlet />
      </main>
      <footer className="border-t border-line px-4 py-6 text-xs text-ink-soft sm:px-6">
        <div className="mx-auto flex max-w-6xl flex-col justify-between gap-3 sm:flex-row sm:items-center">
          <div>
            A scientific research instrument for evaluating, comparing, and diagnosing language model
            reliability.
          </div>
          <div className="font-mono text-[11px]">University Capstone Research Project</div>
        </div>
      </footer>
    </div>
  );
}

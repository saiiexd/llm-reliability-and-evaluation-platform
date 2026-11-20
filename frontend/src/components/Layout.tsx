import { useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { ConnectionBanner, ConnectionIndicator } from "./ConnectionBanner";

// Four sections, in the order people use them:
// see the state -> define a test -> read one run -> compare across runs.
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
            `block rounded-md px-3 py-1.5 text-sm transition-colors ${
              isActive ? "bg-paper-raised font-medium text-ink" : "text-ink-soft hover:text-ink"
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
    <div className="flex min-h-screen flex-col bg-paper text-ink">
      <header className="sticky top-0 z-20 border-b border-line bg-paper/85 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-6xl items-center justify-between gap-4 px-4 sm:px-6">
          <NavLink to="/" className="flex items-baseline gap-2">
            <span className="font-semibold tracking-tight">LLM Reliability</span>
            <span className="hidden text-xs text-ink-soft md:inline">Research Platform</span>
          </NavLink>
          <div className="hidden items-center gap-4 sm:flex">
            <nav className="flex items-center gap-1">
              <NavLinks />
            </nav>
            <div className="h-5 w-px bg-line" />
            <ConnectionIndicator />
          </div>
          <div className="flex items-center gap-2 sm:hidden">
            <button
              type="button"
              className="rounded-md border border-line px-3 py-1 text-sm"
              aria-expanded={menuOpen}
              aria-controls="mobile-nav"
              onClick={() => setMenuOpen((open) => !open)}
            >
              Menu
            </button>
          </div>
        </div>
        {menuOpen && (
          <nav id="mobile-nav" className="space-y-1 border-t border-line px-4 py-3 sm:hidden">
            <NavLinks onNavigate={() => setMenuOpen(false)} />
            <div className="border-t border-line pt-3">
              <ConnectionIndicator />
            </div>
          </nav>
        )}
      </header>
      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-10 sm:px-6">
        <ConnectionBanner />
        <Outlet />
      </main>
      <footer className="border-t border-line">
        <div className="mx-auto max-w-6xl px-4 py-5 text-xs text-ink-soft sm:px-6">
          University capstone project · measures how often a language model answers correctly, and why
          it fails when it doesn't.
        </div>
      </footer>
    </div>
  );
}

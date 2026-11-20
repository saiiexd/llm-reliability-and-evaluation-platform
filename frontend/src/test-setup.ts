// Node >= 25 defines an experimental global `localStorage` (unusable unless
// Node runs with --localstorage-file). Because the global already exists,
// vitest's jsdom environment does not copy jsdom's storage over it, so tests
// see a broken `localStorage`. Re-point the globals at the jsdom window's
// storage so `npm test` works on every Node version without extra flags.
const dom = (globalThis as { jsdom?: { window: Window } }).jsdom;
if (dom) {
  for (const key of ["localStorage", "sessionStorage"] as const) {
    Object.defineProperty(globalThis, key, {
      value: dom.window[key],
      configurable: true,
      writable: true,
    });
  }
}

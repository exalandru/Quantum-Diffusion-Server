// The dashboard reaches outside itself in exactly one way — an HTTP request to
// the origin that served it — so `fetch` is the single seam the tests replace
// (see `test-server.ts`). Everything else — availability rules, quantization
// choices, provenance — is exercised as real backend-shaped data, never
// re-invented in the frontend.
//
// `sessionStorage` is the one piece of browser storage the dashboard uses: it
// holds the playground's per-tab unlock tokens. jsdom provides one, but it is
// shared across tests in a file, so it is cleared between them — a token stored
// by one test must not silently unlock the next.
//
// Nothing clears `localStorage`, because nothing writes it: the API key was
// removed from browser storage in the Tauri->web migration (see the
// "Authentication" note in `api.ts`). Touching it here would also break on
// Node >=22, whose own `localStorage` global shadows jsdom's and reads back
// `undefined` unless the process was started with `--localstorage-file`.
import { afterEach } from "vitest";

// jsdom implements no `ResizeObserver`, and the gallery needs one: its rows are
// justified to the wall's width, so the wall has to be measured, and the wall
// also changes width when the project rail collapses — which no window event
// reports. Without this the view throws on mount and every gallery test fails
// with `ReferenceError` rather than with anything about galleries.
//
// A stub that observes nothing, deliberately. jsdom has no layout engine, so a
// real implementation could only ever report zero — the honest thing is to let
// the component fall back to its initial width and to prove the *solver*
// separately, as a pure function, in `justify.test.ts`. Tests that need a real
// width set one on the element and read it back through `getBoundingClientRect`,
// which jsdom does support.
if (!("ResizeObserver" in globalThis)) {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  } as unknown as typeof ResizeObserver;
}

afterEach(() => {
  window.sessionStorage.clear();
});

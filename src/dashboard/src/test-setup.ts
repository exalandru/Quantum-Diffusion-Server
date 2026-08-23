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

afterEach(() => {
  window.sessionStorage.clear();
});

// The dashboard reaches outside itself in exactly one way — an HTTP request to
// the origin that served it — so `fetch` is the single seam the tests replace
// (see `test-server.ts`). Everything else — availability rules, quantization
// choices, provenance — is exercised as real backend-shaped data, never
// re-invented in the frontend.
//
// `localStorage` is where the API key lives. jsdom provides one, but it is
// shared across tests in a file, so it is cleared between them: a key stored by
// one test must not silently authenticate the next.
import { afterEach } from "vitest";

afterEach(() => {
  window.localStorage.clear();
});

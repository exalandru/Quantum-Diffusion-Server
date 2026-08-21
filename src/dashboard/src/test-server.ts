/**
 * A programmable stand-in for the server, installed over `fetch`.
 *
 * The seam moved. While this was a Tauri app the single boundary was `invoke`,
 * so the tests replaced that; now every call is an HTTP request to the origin
 * that served the page, so `fetch` is the boundary and this replaces it.
 *
 * Replacing `fetch` rather than the `api` module is deliberate, and it is the
 * stronger of the two: it exercises the real paths, the real bodies, the real
 * status handling and the real error unwrapping in `api.ts`. Mocking `api`
 * would assert that the components call functions that exist — which the type
 * checker already says — while leaving `PUT /admin/config` free to become
 * `POST /admin/configuration` with every test still green.
 *
 * An unrouted request is a failure, not an empty answer. A component quietly
 * calling something nobody declared is exactly the drift these tests exist to
 * catch.
 */
import { vi } from "vitest";

export type Seen = { method: string; path: string; body: unknown; headers: Record<string, string> };
export type Route = (request: Seen) => unknown;

export type FakeServer = {
  /** `on("GET /admin/overview", () => …)`. Later registrations win. */
  on: (route: string, handler: Route) => void;
  /** Answer with an OpenAI-shaped error, the way the real server does. */
  fail: (route: string, status: number, message: string, code?: string) => void;
  /** Every request seen, in order. */
  requests: Seen[];
  restore: () => void;
};

export function fakeServer(): FakeServer {
  const routes = new Map<string, Route>();
  const requests: Seen[] = [];

  const handler = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString();
    const path = url.split("?")[0]!;
    const method = (init?.method ?? "GET").toUpperCase();
    const body =
      typeof init?.body === "string" ? JSON.parse(init.body) : (init?.body ?? undefined);
    const headers = Object.fromEntries(
      Object.entries((init?.headers as Record<string, string>) ?? {}).map(([key, value]) => [
        key.toLowerCase(),
        value,
      ]),
    );
    requests.push({ method, path, body, headers });

    // Exact match first, then the query-stripped path, so `/admin/logs?after=3`
    // can be routed as `/admin/logs` without every test spelling the query.
    const route = routes.get(`${method} ${url}`) ?? routes.get(`${method} ${path}`);
    if (!route) {
      throw new Error(`no route for ${method} ${url}`);
    }

    const result = route({ method, path, body, headers });
    if (result instanceof Response) return result;
    return new Response(JSON.stringify(result ?? null), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  });

  const original = globalThis.fetch;
  globalThis.fetch = handler as unknown as typeof fetch;

  return {
    on: (route, fn) => routes.set(route, fn),
    fail: (route, status, message, code) =>
      routes.set(
        route,
        () =>
          new Response(
            JSON.stringify({ error: { message, type: "invalid_request_error", param: null, code } }),
            { status, headers: { "Content-Type": "application/json" } },
          ),
      ),
    requests,
    restore: () => {
      globalThis.fetch = original;
    },
  };
}

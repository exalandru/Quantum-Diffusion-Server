/**
 * One channel: HTTP to the server that served this page.
 *
 * The desktop app had two — Tauri IPC for what a browser cannot do, HTTP for
 * everything else — and the split is gone because the first half moved. Process
 * supervision and installation belong to the menubar app, which does not render
 * anything; the configuration, the catalogue, jobs and logs became `/admin`
 * endpoints on the server itself.
 *
 * **Same-origin, so no address is configured anywhere.** The page is served from
 * `/dashboard` by the very server it talks to, so every request here is a
 * relative path. That is not only convenience: the control plane refuses
 * cross-origin requests, which is what protects a keyless loopback install from
 * any page the user happens to have open, and a hard-coded `http://127.0.0.1:8765`
 * here would be exactly the cross-origin request it refuses.
 */
import type {
  Capabilities,
  CatalogueStatus,
  ConfigDocument,
  Health,
  ImportVerdict,
  JobStatus,
  LocateVerdict,
  LogPage,
  ModelEntry,
  ModelList,
  Overview,
  PlaygroundGeneration,
  PlaygroundSession,
  PlaygroundSessionList,
  Progress,
  Upscaler,
} from "./types";

// ── Authentication ─────────────────────────────────────────────────────────

/**
 * Nothing is stored in this browser.
 *
 * The password is typed once and exchanged for a session the server keeps; the
 * cookie carrying it is `HttpOnly`, so no script on this page — including a
 * hostile one — can read it. The API key is gone from here entirely: an admin
 * session also opens `/v1`, and an admin can read `server.api_key` out of
 * `GET /admin/config` anyway, so holding a second secret in `localStorage`
 * bought nothing and risked something.
 */

/** Thrown for a 401, so the shell can ask for the password instead of erroring. */
export class Unauthorized extends Error {
  constructor(message = "This server requires the admin password.") {
    super(message);
    this.name = "Unauthorized";
  }
}

/** Thrown for a 429, so the shell can say "wait" rather than "wrong". */
export class TooManyAttempts extends Error {
  constructor(message: string) {
    super(message);
    this.name = "TooManyAttempts";
  }
}

/**
 * Thrown for a 404, so a caller can tell "this is gone" from "I could not ask".
 *
 * The distinction is load-bearing on the playground: a deleted session must drop
 * out of the view, while a failed poll must leave the view alone and say so.
 */
export class NotFound extends Error {
  constructor(message: string) {
    super(message);
    this.name = "NotFound";
  }
}

/**
 * Thrown for a 403 whose code is `session_locked`: a playground session with a
 * password, and no live unlock token for it.
 *
 * Its own class, and never a 401: a 401 is what sends the shell to the admin
 * login, and a locked session is not a missing admin credential. Matched on the
 * code rather than the status, because `cross_site_denied` is a 403 too and
 * must stay an ordinary error.
 */
export class Locked extends Error {
  constructor(message: string) {
    super(message);
    this.name = "Locked";
  }
}

// ── Unlock tokens ──────────────────────────────────────────────────────────
//
// A session password is redeemed once for a token the server keeps in memory.
// The browser's copy lives in `sessionStorage`: it dies with the tab, which is
// the lifetime the user asked for when they set a password, and it is not
// shared with other tabs, each of which unlocks on its own.

const UNLOCK_HEADER = "X-QDS-Session-Token";
const unlockKey = (sessionId: string) => `qds.playground.unlock.${sessionId}`;

export function unlockToken(sessionId: string): string | null {
  try {
    return window.sessionStorage.getItem(unlockKey(sessionId));
  } catch {
    return null;
  }
}

export function rememberUnlock(sessionId: string, token: string): void {
  try {
    window.sessionStorage.setItem(unlockKey(sessionId), token);
  } catch {
    // Storage disabled: the unlock lasts as long as the page does.
  }
}

export function forgetUnlock(sessionId: string): void {
  try {
    window.sessionStorage.removeItem(unlockKey(sessionId));
  } catch {
    // Nothing to forget.
  }
}

/** The token's header, when one is held for this session. */
function unlockHeaders(sessionId: string | undefined): Record<string, string> {
  const token = sessionId ? unlockToken(sessionId) : null;
  return token ? { [UNLOCK_HEADER]: token } : {};
}

/**
 * An image URL the browser can load directly.
 *
 * An `<img>` sends no headers, so the image route accepts the token as a query
 * parameter — and only the image route does. Applied at render time: the URL
 * in state stays the server's, so a token change or a lock never has to rewrite
 * history, and the filename the delete button parses out of it stays clean.
 */
export function imageUrl(url: string, sessionId: string): string {
  const token = unlockToken(sessionId);
  return token ? `${url}?t=${encodeURIComponent(token)}` : url;
}

/**
 * The same image, as the small derived tile the gallery scrolls.
 *
 * `/thumb` is a sibling of the file's own route and is enforced identically —
 * same session lock, per request, and the same `?t=` for the same reason, since
 * an `<img>` sends no headers. Built here rather than in the view so the URL in
 * state stays the server's, exactly as `imageUrl` leaves it: the filename the
 * delete button parses back out of it must not acquire a suffix.
 *
 * A thumbnail the server cannot derive is answered with the full image, so a
 * caller of this never has to handle a missing tile.
 */
export function thumbnailUrl(url: string, sessionId: string): string {
  return imageUrl(`${url}/thumb`, sessionId);
}

/** Extra per-request options on top of `fetch`'s. */
type Options = RequestInit & {
  /** Attach the unlock token held for this playground session, if any. */
  sessionId?: string;
};

async function request<T>(path: string, { sessionId, ...init }: Options = {}): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { ...unlockHeaders(sessionId), ...(init.headers as Record<string, string>) },
    // The session cookie is same-origin, and this is what sends it.
    credentials: "same-origin",
  });
  if (response.status === 401) throw new Unauthorized((await describe(response)).message);
  if (response.status === 429) throw new TooManyAttempts((await describe(response)).message);
  if (response.status === 404) throw new NotFound((await describe(response)).message);
  if (response.status === 403) {
    const { message, code } = await describe(response);
    if (code === "session_locked") throw new Locked(message);
    throw new Error(message);
  }
  if (!response.ok) throw new Error((await describe(response)).message);
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

const get = <T,>(path: string, options: Options = {}) => request<T>(path, options);

const send = <T,>(
  path: string,
  method: "POST" | "PUT" | "PATCH" | "DELETE",
  body?: unknown,
  options: Options = {},
) =>
  request<T>(path, {
    ...options,
    method,
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });

// ── Control plane ──────────────────────────────────────────────────────────

export const overview = () => get<Overview>("/admin/overview");

/**
 * The configuration document, exactly as written.
 *
 * `unparsed` when the file will not parse at all — the state that starts the
 * server in recovery mode, and therefore the one where this screen matters
 * most. The raw text comes back with it, because a JSON error names a line and
 * a column and neither means anything without the line.
 */
export const configRead = () => get<ConfigDocument>("/admin/config");

export const configWrite = (value: unknown) =>
  send<{ ok: boolean; restartRequired: boolean; issues: RuntimeIssue[] }>(
    "/admin/config",
    "PUT",
    value,
  );

export type RuntimeIssue = { code: string; field: string; message: string };

export const hfTokenWrite = (token: string) =>
  send<{ ok: boolean }>("/admin/hf-token", "POST", { token });

/**
 * The catalogue and whatever is wrong with the configuration it was read from.
 *
 * Two outputs rather than one list: a runtime invariant the *generation server*
 * needs — a default model that has been switched off, say — used to make this
 * call fail outright, taking the Models view with it and removing the only
 * screen that could repair it.
 */
export const modelsStatus = () => get<CatalogueStatus>("/admin/models");

// ── Jobs ───────────────────────────────────────────────────────────────────

/** Starts the download and returns once the child is running, not once it is done. */
export const modelFetch = (key: string) =>
  send<JobStatus>("/admin/jobs/fetch", "POST", { key });

export const prequantizeRun = (
  model: string,
  bits: number,
  components: string[] = [],
  dest?: string,
) =>
  send<JobStatus>("/admin/jobs/prequantize", "POST", {
    model,
    bits,
    components: components.length ? components : null,
    dest: dest ?? null,
  });

export const jobStatus = () => get<JobStatus>("/admin/jobs");
export const jobCancel = () => send<JobStatus>("/admin/jobs/cancel", "POST");

// ── Local models ───────────────────────────────────────────────────────────

export const localModelInspect = (path: string) =>
  send<ImportVerdict>("/admin/import/inspect", "POST", { path });

export const localModelLocate = (path: string, model: string) =>
  send<LocateVerdict>("/admin/import/locate", "POST", { path, model });

export const localModelRegister = (
  path: string,
  baseProfile: string,
  name?: string,
  apiName?: string,
) =>
  send<{ ok: boolean; reason?: string; already_imported?: boolean }>(
    "/admin/import/register",
    "POST",
    { path, base_profile: baseProfile, name: name ?? null, api_name: apiName ?? null },
  );

export const localModelForget = (id: string) =>
  send<{ ok: boolean; reason?: string }>("/admin/import/forget", "POST", { id });

// ── Logs ───────────────────────────────────────────────────────────────────

/**
 * Everything logged after `after`, and where the tail now ends.
 *
 * A cursor rather than a stream: a poll is idempotent, a missed poll is
 * harmless, and a tab left in the background resumes where it stopped instead
 * of replaying or losing the interval. `dropped` says how much fell off the end
 * of the server's ring buffer in between, which is the difference between
 * "nothing happened" and "you missed it".
 */
export const logsAfter = (after: number, limit = 500) =>
  get<LogPage>(`/admin/logs?after=${after}&limit=${limit}`);

// ── Logging in ─────────────────────────────────────────────────────────────

export type SessionStatus = {
  passwordSet: boolean;
  authenticated: boolean;
  loopback: boolean;
  recoveryMode: boolean;
};

/** Unauthenticated on purpose: the login screen has to be able to ask. */
export const sessionStatus = () => get<SessionStatus>("/admin/session");

export const logIn = (password: string) =>
  send<void>("/admin/session", "POST", { password });

export const logOut = () => request<void>("/admin/session", { method: "DELETE" });

/** Set the first password, or change one. `current` is required once one exists. */
export const setPassword = (next: string, current?: string) =>
  send<{ ok: boolean; loggedOut: boolean }>("/admin/password", "POST", {
    new: next,
    current: current ?? null,
  });

// ── Logging in to the playground ───────────────────────────────────────────
//
// A second plane with a second credential, so a second set of calls. Sharing
// `SessionStatus` would be wrong in the one field that matters: this plane says
// whether its gate is armed at all, which is what `playground_auth_scope`
// decides and what the admin plane has no equivalent of.

export type PlaygroundSessionStatus = {
  passwordSet: boolean;
  authenticated: boolean;
  loopback: boolean;
  /** Whether the playground demands its password here and now. */
  gated: boolean;
};

/** Unauthenticated on purpose, and outside the gate it describes. */
export const playgroundSessionStatus = () =>
  get<PlaygroundSessionStatus>("/playground/api/session");

export const playgroundLogIn = (password: string) =>
  send<void>("/playground/api/session", "POST", { password });

export const playgroundLogOut = () =>
  request<void>("/playground/api/session", { method: "DELETE" });

/**
 * Set or change the playground password. Admin authority, not the playground's
 * own: the credential being replaced is not the one that authorises the call,
 * which is what makes this the way back from a forgotten one.
 */
export const playgroundPasswordSet = (next: string) =>
  send<{ ok: boolean }>("/admin/playground/password", "POST", { new: next });

/** Refused with a 409 while `playground_auth_scope` is `always`. */
export const playgroundPasswordRemove = () =>
  send<{ ok: boolean }>("/admin/playground/password", "DELETE");

// ── Lifetime ───────────────────────────────────────────────────────────────

/**
 * Ask the server to re-exec itself.
 *
 * There is no start and no stop here, and that is the architecture rather than
 * an omission: this page is served by the server, so it cannot exist to start
 * one, and stopping one from inside it would leave nothing to show the result.
 * The menubar app owns the lifetime; `qds serve` owns it otherwise.
 */
export const serverRestart = () => send<{ ok: boolean }>("/admin/restart", "POST");

// ── Generation ─────────────────────────────────────────────────────────────

/** Unauthenticated: a probe available even with a key configured. */
export const health = () => get<Health>("/health");
export const capabilities = () => get<Capabilities>("/v1/capabilities");
export const cancelGeneration = () =>
  send<{ cancelled: boolean; state: string }>("/v1/cancel", "POST");
export const unload = () =>
  send<{ loaded_model: string | null; memory: Health["memory"] }>("/v1/unload", "POST");

/** Public model names, the identifiers `/v1` accepts and records store. */
export const models = () => get<ModelList>("/v1/models");

/** One model's capabilities, by public name. */
export const modelInfo = (id: string) => get<ModelEntry>(`/v1/models/${encodeURIComponent(id)}`);

/** The OpenAPI schema. The server serves no rendered documentation page. */
export const schemaUrl = () => "/openapi.json";

// ── Playground ─────────────────────────────────────────────────────────────
//
// Server-owned history: nothing here is cached in the browser, so a reopened
// tab and a fresh one see the same thing.

export const playgroundSessions = () =>
  get<PlaygroundSessionList>("/playground/api/sessions");

/**
 * Hold or release the playground queue, for every session at once.
 *
 * Deliberately not a claim about the engine: the hold takes effect at the
 * runner's next boundary, so the image already being denoised still finishes.
 */
export const playgroundSetPaused = (paused: boolean) =>
  send<{ paused: boolean }>("/playground/api/queue", "POST", { paused });

export const playgroundSessionCreate = () =>
  send<PlaygroundSession>("/playground/api/sessions", "POST");

export const playgroundSession = (id: string) =>
  get<{ session: PlaygroundSession; generations: PlaygroundGeneration[] }>(
    `/playground/api/sessions/${encodeURIComponent(id)}`,
    { sessionId: id },
  );

export const playgroundSessionDelete = (id: string) =>
  send<void>(`/playground/api/sessions/${encodeURIComponent(id)}`, "DELETE", undefined, {
    sessionId: id,
  });

/** `null` or blank hands the title back to the first prompt. */
export const playgroundSessionRename = (id: string, title: string | null) =>
  send<PlaygroundSession>(
    `/playground/api/sessions/${encodeURIComponent(id)}`,
    "PATCH",
    { title },
    { sessionId: id },
  );

/**
 * Redeem the password for a token, and keep it. The rejection for a wrong
 * password is a plain `Error` (403, not 401); five of them are a `TooManyAttempts`.
 */
export const playgroundSessionUnlock = async (id: string, password: string) => {
  const payload = await send<{ token: string; session: PlaygroundSession }>(
    `/playground/api/sessions/${encodeURIComponent(id)}/unlock`,
    "POST",
    { password },
  );
  rememberUnlock(id, payload.token);
  return payload.session;
};

/** Give the token back. Forgotten here even if the server no longer has it. */
export const playgroundSessionLock = async (id: string) => {
  try {
    await send<void>(`/playground/api/sessions/${encodeURIComponent(id)}/lock`, "POST", undefined, {
      sessionId: id,
    });
  } finally {
    forgetUnlock(id);
  }
};

/**
 * Set or change the password. Every other token dies with the old password;
 * the one that comes back is this tab's, kept so it stays where it is.
 */
export const playgroundSessionPasswordSet = async (id: string, password: string) => {
  const payload = await send<{ token: string }>(
    `/playground/api/sessions/${encodeURIComponent(id)}/password`,
    "POST",
    { password },
    { sessionId: id },
  );
  rememberUnlock(id, payload.token);
};

export const playgroundSessionPasswordRemove = async (id: string) => {
  await send<void>(
    `/playground/api/sessions/${encodeURIComponent(id)}/password`,
    "DELETE",
    undefined,
    { sessionId: id },
  );
  forgetUnlock(id);
};

/** The admin's way past a forgotten password: needs the admin session, not the token. */
export const playgroundAdminStripPassword = (id: string) =>
  send<void>(`/admin/playground/sessions/${encodeURIComponent(id)}/password`, "DELETE");

/**
 * Accepted, not finished: the record comes back `queued`.
 *
 * Not `send`, which JSON-encodes: `fetch` must write the multipart
 * `Content-Type` itself, or the boundary the server parses by is missing.
 */
export const playgroundGenerate = (sessionId: string, form: FormData) =>
  request<PlaygroundGeneration>(
    `/playground/api/sessions/${encodeURIComponent(sessionId)}/generations`,
    { method: "POST", body: form, sessionId },
  );

/** What the image toolbar can offer, and whether a click costs a download. */
export const playgroundUpscalers = () =>
  get<{ upscalers: Upscaler[] }>("/playground/api/upscalers");

/**
 * Enlarge an image the session already owns. Accepted, not finished.
 *
 * JSON rather than multipart, unlike `playgroundGenerate`: the source is a file
 * the server already holds and can attribute, so there are no bytes to send.
 */
export const playgroundUpscale = (
  sessionId: string,
  body: { image: string; model: string; scale: number; group?: string | null },
) =>
  send<PlaygroundGeneration>(
    `/playground/api/sessions/${encodeURIComponent(sessionId)}/upscales`,
    "POST",
    body,
    { sessionId },
  );

export const playgroundCancel = (sessionId: string, generationId: string) =>
  send<PlaygroundGeneration>(
    `/playground/api/generations/${encodeURIComponent(generationId)}/cancel`,
    "POST",
    undefined,
    { sessionId },
  );

/** A whole feed entry: every generation of the lineage, and the files only it owned. */
export const playgroundGroupDelete = (sessionId: string, groupId: string) =>
  send<void>(`/playground/api/groups/${encodeURIComponent(groupId)}`, "DELETE", undefined, {
    sessionId,
  });

export const playgroundImageDelete = (sessionId: string, filename: string) =>
  send<void>(`/playground/api/images/${encodeURIComponent(filename)}`, "DELETE", undefined, {
    sessionId,
  });

/**
 * Subscribe to `/v1/progress`, reconnecting until told to stop.
 *
 * `fetch` rather than `EventSource`: the latter allows no `Authorization`
 * header, and the route is protected like the rest of `/v1`. SSE framing is
 * simple enough to parse by hand — and doing so is what makes the retry below
 * possible, since `EventSource`'s own reconnect is neither bounded nor
 * cancellable on our terms.
 *
 * Retrying here rather than in the panel is deliberate. One loop owns the
 * connection, so "only one stream at a time" is structural rather than a rule
 * someone has to remember: each attempt is awaited before the next begins, and
 * the caller's single unsubscribe cancels both the in-flight request and any
 * pending wait.
 *
 * Two different endings are both drops. A thrown error is the obvious one; a
 * *clean* end — the server closing the response — used to fall out of the read
 * loop and return silently, leaving a panel subscribed to nothing at all with
 * no indication. Both now schedule a retry.
 */
export function subscribeProgress(
  onProgress: (progress: Progress) => void,
  onError: (message: string) => void,
): () => void {
  let stopped = false;
  let controller: AbortController | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;
  /** Resolves the pending backoff early, so unsubscribing is immediate. */
  let wake: (() => void) | null = null;
  let attempt = 0;

  const sleep = (ms: number) =>
    new Promise<void>((resolve) => {
      wake = resolve;
      timer = setTimeout(resolve, ms);
    });

  const consume = async (signal: AbortSignal) => {
    const response = await fetch("/v1/progress", {
      headers: { Accept: "text/event-stream" },
      signal,
    });
    if (!response.ok || !response.body) throw new Error((await describe(response)).message);
    // The connection opened, so whatever went wrong before is over: the next
    // failure starts its backoff from the beginning rather than inheriting a
    // ten-second delay from an outage that has since been fixed.
    attempt = 0;

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) return;
      buffer += decoder.decode(value, { stream: true });
      // SSE frames are separated by a blank line.
      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const payload = frame
          .split("\n")
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trim())
          .join("");
        // Keep-alive `: ping` frames have no `data:`: nothing to do.
        if (payload) onProgress(JSON.parse(payload) as Progress);
        boundary = buffer.indexOf("\n\n");
      }
    }
  };

  void (async () => {
    while (!stopped) {
      const attemptController = new AbortController();
      controller = attemptController;
      try {
        await consume(attemptController.signal);
        if (stopped) return;
        onError("the server closed the progress stream");
      } catch (error) {
        // An abort is us, not a fault: unsubscribing must not report an error
        // to a panel that is on its way out.
        if (stopped || attemptController.signal.aborted) return;
        onError(messageOf(error));
      } finally {
        // Whatever ended this attempt, the request that carried it is finished
        // with. Without this a failure that is *not* the connection closing —
        // a malformed frame, say — would leave the previous body draining
        // while the retry opens a second one, which is the duplicate-stream
        // case this design exists to make impossible.
        attemptController.abort();
      }
      if (stopped) return;
      await sleep(backoffMs(attempt++));
    }
  })();

  return () => {
    stopped = true;
    controller?.abort();
    if (timer) clearTimeout(timer);
    // Let the loop observe `stopped` and finish, rather than leaving its
    // promise suspended for the lifetime of the page.
    wake?.();
  };
}

/** First retry, and the ceiling. Doubling in between. */
export const RECONNECT_BASE_MS = 500;
export const RECONNECT_MAX_MS = 10_000;

/**
 * How long to wait before retry number `attempt` (0-based).
 *
 * Bounded on purpose: a local server that is genuinely down should be retried
 * every ten seconds forever — often enough that restarting it recovers the view
 * without touching anything, rarely enough that a stopped server does not
 * produce a request storm in the log for as long as the page is open.
 */
export function backoffMs(attempt: number): number {
  return Math.min(RECONNECT_MAX_MS, RECONNECT_BASE_MS * 2 ** Math.max(0, attempt));
}

async function describe(response: Response): Promise<{ message: string; code: string | null }> {
  try {
    const body = (await response.json()) as { error?: { message?: string; code?: string | null } };
    if (body.error?.message) return { message: body.error.message, code: body.error.code ?? null };
  } catch {
    // Non-JSON body: fall back to the status.
  }
  return { message: `HTTP ${response.status}`, code: null };
}

export function messageOf(error: unknown): string {
  if (error instanceof Error) return error.message;
  if (typeof error === "string") return error;
  return JSON.stringify(error);
}

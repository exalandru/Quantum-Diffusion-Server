import { useCallback, useEffect, useRef, useState } from "react";

import * as api from "../api";
import { NotFound, Unauthorized, messageOf } from "../api";
import { LoginPrompt } from "../LoginPrompt";
import { Unreachable } from "../Unreachable";
import type { PlaygroundGeneration, PlaygroundSession, Progress } from "../types";
import { Composer, type Draft } from "./Composer";
import { GenerationFeed } from "./GenerationFeed";
import { SessionList } from "./SessionList";

const IDLE: Progress = {
  state: "idle",
  model: null,
  kind: null,
  seed: null,
  step: 0,
  total: 0,
  preview_seq: 0,
  elapsed_s: null,
  loaded_model: null,
  memory: {},
};

/** While something is in flight; the list moves more slowly than the feed. */
const DETAIL_POLL_MS = 2000;
const LIST_POLL_MS = 5000;

/**
 * One more image for a group, from the request that opened it.
 *
 * The size is the one that actually ran (`WxH`, never a model default), and the
 * `group` field is what makes the result land in the same feed entry instead of
 * a new one. No seed: the server picks a fresh one unless the caller sets it.
 */
function settingsOf(root: PlaygroundGeneration) {
  const form = new FormData();
  form.set("prompt", root.prompt);
  form.set("model", root.model);
  form.set("n", "1");
  form.set("size", root.size);
  form.set("steps", String(root.steps));
  form.set("group", root.groupId);
  return form;
}

/**
 * The studio.
 *
 * It holds no history of its own: sessions, generations, images and live status
 * all come from the server, so closing this tab mid-generation loses nothing and
 * reopening it shows the finished image. The selected session is kept in the URL
 * (`?session=…`) so a reload — or a bookmark — lands back where it was.
 */
export function PlaygroundApp() {
  const [sessions, setSessions] = useState<PlaygroundSession[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [generations, setGenerations] = useState<PlaygroundGeneration[]>([]);
  const [progress, setProgress] = useState<Progress>(IDLE);
  const [models, setModels] = useState<{ id: string; name: string }[]>([]);
  const [maxN, setMaxN] = useState(1);
  const [defaultModel, setDefaultModel] = useState("");
  const [session, setSession] = useState<api.SessionStatus | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const [cancelling, setCancelling] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);

  const onUnauthorized = useCallback(() => {
    void api
      .sessionStatus()
      .then(setSession)
      .catch((cause) => setConnectionError(messageOf(cause)));
  }, []);

  /** Any call that can 401 goes through here, so the login gate is one path. */
  const guarded = useCallback(
    async <T,>(call: () => Promise<T>): Promise<T | undefined> => {
      try {
        const value = await call();
        setConnectionError(null);
        return value;
      } catch (cause) {
        if (cause instanceof Unauthorized) onUnauthorized();
        else setConnectionError(messageOf(cause));
        return undefined;
      }
    },
    [onUnauthorized],
  );

  const refreshSessions = useCallback(async () => {
    const payload = await guarded(api.playgroundSessions);
    if (payload) setSessions(payload.sessions);
    setLoaded(true);
    return payload?.sessions;
  }, [guarded]);

  // Kept in a ref as well: the pollers below must read the current selection
  // without re-arming their timers on every change.
  const current = useRef<string | null>(null);
  current.current = selected;

  const refreshDetail = useCallback(async () => {
    const id = current.current;
    if (!id) {
      setGenerations([]);
      return;
    }
    try {
      const payload = await api.playgroundSession(id);
      setGenerations(payload.generations);
      setConnectionError(null);
    } catch (cause) {
      if (cause instanceof Unauthorized) {
        onUnauthorized();
      } else if (cause instanceof NotFound) {
        // Gone — deleted here or elsewhere. Drop the selection rather than
        // reporting a failure the user cannot act on.
        setSelected(null);
        setGenerations([]);
      } else {
        // A failed poll is not a deleted session: the feed stays, and the page
        // says it could not reach the server. Wiping the view here was the way
        // a proxy hiccup mid-generation looked exactly like a deletion.
        setConnectionError(messageOf(cause));
      }
    }
  }, [onUnauthorized]);

  // The URL is the only client-side state, and it holds an id, not a history.
  useEffect(() => {
    const wanted = new URLSearchParams(window.location.search).get("session");
    void refreshSessions().then((listed) => {
      if (wanted && listed?.some((entry) => entry.id === wanted)) setSelected(wanted);
    });
  }, [refreshSessions]);

  useEffect(() => {
    const url = new URL(window.location.href);
    if (selected) url.searchParams.set("session", selected);
    else url.searchParams.delete("session");
    window.history.replaceState(null, "", url);
    void refreshDetail();
  }, [selected, refreshDetail]);

  // What is generated, and how much of it: the model list and `max_n` are the
  // two things the composer cannot invent.
  useEffect(() => {
    void guarded(api.models).then((listed) => {
      if (listed)
        setModels(listed.data.map((entry) => ({ id: entry.id, name: entry.display_name || entry.id })));
    });
    void guarded(api.capabilities).then((caps) => {
      if (caps) {
        setMaxN(caps.max_n);
        setDefaultModel(caps.default_model);
      }
    });
  }, [guarded]);

  useEffect(() => {
    const stop = api.subscribeProgress(setProgress, () => setProgress(IDLE));
    return stop;
  }, []);

  // Polling, and only while it buys something: a session with nothing in flight
  // changes only when this page changes it, and every mutation refetches.
  const active =
    generations.some((entry) => entry.status === "queued" || entry.status === "running") ||
    sessions.some((entry) => entry.generating);

  // Id → readable name. A `Record`, not a `Map`: a string-keyed lookup table
  // with no insertion, deletion or iteration order to preserve is exactly what
  // a `Record` is for.
  const modelNames: Record<string, string> = Object.fromEntries(
    models.map(({ id, name }) => [id, name]),
  );
  // `?? id` is the honest fallback: a record can name a model that has since
  // been disabled or removed from the catalogue.
  const nameOf = (id: string) => modelNames[id] ?? id;

  useEffect(() => {
    if (!active) return;
    const detail = setInterval(() => void refreshDetail(), DETAIL_POLL_MS);
    const list = setInterval(() => void refreshSessions(), LIST_POLL_MS);
    return () => {
      clearInterval(detail);
      clearInterval(list);
    };
  }, [active, refreshDetail, refreshSessions]);

  /**
   * The one submission path: create the session on demand, POST, refresh.
   *
   * Shared by the composer and by the per-image toolbar, so a refine or a
   * variation gets the same session-creation, refresh and error handling as a
   * typed prompt rather than a second, drifting copy of it.
   */
  async function post(sessionId: string | null, form: FormData) {
    setSending(true);
    setSubmitError(null);
    try {
      // The session row is created by the first submission, so "New session"
      // cannot litter the sidebar with empty conversations.
      const id = sessionId ?? (await api.playgroundSessionCreate()).id;
      await api.playgroundGenerate(id, form);
      if (id !== selected) setSelected(id);
      else await refreshDetail();
      await refreshSessions();
    } catch (cause) {
      if (cause instanceof Unauthorized) onUnauthorized();
      else setSubmitError(messageOf(cause));
    } finally {
      setSending(false);
    }
  }

  async function submit(draft: Draft) {
    const form = new FormData();
    form.set("prompt", draft.prompt);
    form.set("model", draft.model);
    form.set("n", String(draft.n));
    if (draft.size) form.set("size", draft.size);
    if (draft.steps !== null) form.set("steps", String(draft.steps));
    if (draft.seed !== null) form.set("seed", String(draft.seed));
    if (draft.image) form.set("image", draft.image);
    await post(selected, form);
  }

  /**
   * The selected image back through the model, joining its own group.
   *
   * `root` is the generation that opened the group, and it owns the settings:
   * prompt, model, size and step count. Only the image and its seed come from
   * the thumbnail that was clicked, which is what "refine this one" means.
   *
   * `sending` is raised before the image is fetched, not inside `post`: the
   * toolbar's disabled state is what stops a second click, and the fetch is a
   * window in which it would otherwise still be enabled.
   */
  async function refine(root: PlaygroundGeneration, image: { url: string; seed: number }) {
    setSending(true);
    setSubmitError(null);
    let form: FormData;
    try {
      const blob = await fetch(image.url).then((response) => {
        if (!response.ok) throw new Error(`Could not fetch the image (${response.status}).`);
        return response.blob();
      });
      form = settingsOf(root);
      form.set("seed", String(image.seed));
      form.set("image", new File([blob], "refine.png", { type: "image/png" }));
    } catch (cause) {
      setSubmitError(messageOf(cause));
      setSending(false);
      return;
    }
    await post(root.sessionId, form);
  }

  /**
   * Another take on the group's original request: its prompt and settings, a
   * fresh random seed, and none of the images the group has produced since —
   * a variation of the idea, not of the picture it was asked from. Refining is
   * the action that starts from a generated image.
   *
   * The one image it does re-send is the reference the *original* request was
   * made with, when there was one: "make the hat red" without the photo of the
   * hat is not the same request, it is a different one.
   */
  async function variation(root: PlaygroundGeneration) {
    setSending(true);
    setSubmitError(null);
    const form = settingsOf(root);
    if (root.contextImage) {
      try {
        const blob = await fetch(root.contextImage).then((response) => {
          if (!response.ok)
            throw new Error(`Could not fetch the reference image (${response.status}).`);
          return response.blob();
        });
        form.set("image", new File([blob], "context.png", { type: "image/png" }));
      } catch (cause) {
        setSubmitError(messageOf(cause));
        setSending(false);
        return;
      }
    }
    await post(root.sessionId, form);
  }

  async function deleteImage(url: string) {
    const filename = url.split("/").pop() ?? "";
    await guarded(() => api.playgroundImageDelete(filename));
    await refreshDetail();
    // The server bumps the session's `updated_at`, which is both the sidebar's
    // sort key and its timestamp: an idle session polls nothing, so without this
    // the list would keep showing the pre-deletion order for ever.
    await refreshSessions();
  }

  async function cancel(id: string) {
    setCancelling(id);
    await guarded(() => api.playgroundCancel(id));
    await refreshDetail();
    await refreshSessions();
    setCancelling(null);
  }

  async function remove(id: string) {
    await guarded(() => api.playgroundSessionDelete(id));
    if (id === selected) setSelected(null);
    await refreshSessions();
  }

  if (session) {
    return (
      <LoginPrompt
        status={session}
        onAuthenticated={() => {
          setSession(null);
          void refreshSessions();
          void refreshDetail();
        }}
      />
    );
  }

  if (!loaded && connectionError) return <Unreachable reason={connectionError} />;

  return (
    <main className="playground">
      <header>
        <div className="identity">
          <h1>Quantum Diffusion Server</h1>
          <span className="pill pill-live">Running</span>
        </div>
        {/* No tab strip here: this page has one view and one way out. The arrow
            says the button leaves rather than switches, and `?view=config` lands
            on the screen the label names. */}
        <a className="shell-link" href="/dashboard/?view=config" target="_blank">
          Server Config
          <svg
            viewBox="0 0 24 24"
            width="14"
            height="14"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <path d="M4 12h13M12.5 6.5 18 12l-5.5 5.5" />
          </svg>
        </a>
      </header>

      {connectionError && (
        <div className="notice notice-error" role="status">
          <strong>Could not reach the server.</strong> {connectionError}
        </div>
      )}

      <div className="pg-layout">
        <SessionList
          sessions={sessions}
          selected={selected}
          onSelect={setSelected}
          onNew={() => setSelected(null)}
          onDelete={(id) => void remove(id)}
        />
        <section className="pg-studio">
          {generations.length === 0 ? (
            <div className="pg-hero">
              <span className="pg-hero-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" strokeWidth="1.6">
                  <rect x="3" y="4" width="18" height="16" rx="2.5" />
                  <circle cx="8.5" cy="9.5" r="1.8" />
                  <path d="M4 17.5 9.5 12l4 4 2.5-2.5 4 4" strokeLinecap="round" />
                </svg>
              </span>
              <h2 className="pg-hero-title">Imagine</h2>
              <p className="pg-hero-tagline">
                Bring your vision to life and craft the extraordinary
              </p>
            </div>
          ) : (
            <GenerationFeed
              generations={generations}
              progress={progress}
              onCancel={(id) => void cancel(id)}
              cancelling={cancelling}
              busy={sending}
              onRefine={(entry, image) => void refine(entry, image)}
              onVariation={(entry) => void variation(entry)}
              onDeleteImage={(url) => void deleteImage(url)}
              nameOf={nameOf}
            />
          )}
          <Composer
            models={models}
            defaultModel={defaultModel}
            maxN={maxN}
            busy={sending}
            error={submitError}
            onSubmit={(draft) => void submit(draft)}
          />
        </section>
      </div>
    </main>
  );
}

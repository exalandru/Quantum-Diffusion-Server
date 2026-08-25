import { useCallback, useEffect, useRef, useState, type CSSProperties } from "react";

import * as api from "../api";
import { Locked, NotFound, Unauthorized, messageOf } from "../api";
import { LoginPrompt } from "../LoginPrompt";
import { Unreachable } from "../Unreachable";
import type {
  PlaygroundGeneration,
  PlaygroundSession,
  Progress,
  RewriteCapabilities,
  Upscaler,
} from "../types";
import { Composer, type Draft } from "./Composer";
import { GalleryView } from "./GalleryView";
import { GenerationFeed } from "./GenerationFeed";
import { LightTableView } from "./LightTableView";
import {
  NewProjectDialog,
  PasswordDialog,
  RenameDialog,
  UnlockDialog,
} from "./SessionDialogs";
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

/** A form open over the studio, and the project it is about. */
type Dialog =
  /** Not about a project yet: this one creates it. */
  | { kind: "new" }
  | { kind: "rename"; id: string }
  | { kind: "password"; id: string }
  | { kind: "unlock"; id: string; then: "open" | "delete" };

/**
 * Whether the rail is collapsed, remembered per browser.
 *
 * Every read and write of it is wrapped, because which state the rail is in is a
 * preference: losing it must degrade to the default, never throw. There are
 * three ways the store can be unavailable — a browser with storage disabled
 * (Safari's private mode throws on write), a quota refusal, and Node's own
 * `localStorage` global, which shadows jsdom's under Node >= 22 and reads back
 * `undefined` unless the process was started with `--localstorage-file` (see the
 * note in `test-setup.ts`).
 */
const RAIL_KEY = "qds.playground.rail-collapsed";

/**
 * Which presentation of the project is on screen.
 *
 * Three: the prompt feed, the gallery wall, and the light table — one picture at
 * a time with its facts beside it.
 */
type ViewMode = "prompts" | "gallery" | "table";

const DEFAULT_MODE: ViewMode = "prompts";

/**
 * A stored or linked string, read as a view — or as "no preference".
 *
 * One function rather than a comparison at each of the three sites that need it
 * (the URL at mount, the stored preference, and the guard below), because those
 * three drifting apart is how a third view becomes reachable from a link but not
 * from the store. Anything unrecognised is `null`: an unknown value must read as
 * "no preference" and never as "some fourth view".
 */
const asMode = (value: string | null): ViewMode | null =>
  value === "prompts" || value === "gallery" || value === "table" ? value : null;

/**
 * The URL parameter that carries it — `?mode=`, deliberately not `?view=`.
 *
 * `?view=` already names *which surface renders*: `?view=plugin` is this page
 * without its controls, for Hermes' pane, and `?view=config` is the dashboard's
 * own screen selector. That parameter is read once at mount and never rewritten,
 * because the surface does not change under the user. This one is rewritten
 * every time the user switches, and it means something only *within* the studio
 * surface — a different question, so a different name. Overloading `view` would
 * also make `?view=plugin&view=gallery` expressible, which is nonsense the URL
 * should not be able to say.
 */
const MODE_PARAM = "mode";

/**
 * The remembered view of one project, per browser.
 *
 * Keyed per project, the shape `api.ts` uses for unlock tokens
 * (`qds.playground.unlock.<id>`), because the question is the same one: a fact
 * about *this* browser and *that* project. `localStorage` rather than the
 * session store the tokens use, since a preference is meant to outlive the tab;
 * the consequence, accepted explicitly, is that it does not follow the user to
 * another machine.
 *
 * Every read is guarded and an unknown value reads as "no preference": which
 * view a project opens in must never be able to stop it opening. Same three
 * failure modes as `RAIL_KEY` above.
 */
const modeKey = (sessionId: string) => `qds.playground.view.${sessionId}`;

function rememberedMode(sessionId: string): ViewMode | null {
  try {
    return asMode(window.localStorage?.getItem(modeKey(sessionId)) ?? null);
  } catch {
    return null;
  }
}

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
  // Carried, not dropped: "make it not blurry" without the negative prompt is a
  // different request, and a refine that quietly changed it would be lying about
  // what the entry is a variation *of*.
  if (root.negativePrompt) form.set("negative_prompt", root.negativePrompt);
  // The exact text the ancestor was generated from, replayed rather than
  // regenerated. A rewrite is sampled: asking for a fresh one would produce
  // different words, and a "variation" of a different prompt is not a variation.
  // Note what is *not* set: `rewrite`. The two are refused together.
  if (root.rewrittenPrompt) form.set("rewritten_prompt", root.rewrittenPrompt);
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
  // The queue is held. Server-owned like everything else here, and read from the
  // same session-list payload the sidebar already polls.
  const [paused, setPaused] = useState(false);
  const [pausing, setPausing] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const [generations, setGenerations] = useState<PlaygroundGeneration[]>([]);
  const [progress, setProgress] = useState<Progress>(IDLE);
  const [models, setModels] = useState<{ id: string; name: string }[]>([]);
  const [maxN, setMaxN] = useState(1);
  const [rewrite, setRewrite] = useState<RewriteCapabilities | null>(null);
  const [presetPrompt, setPresetPrompt] = useState<{ text: string; nonce: number } | null>(null);
  const [defaultModel, setDefaultModel] = useState("");
  const [session, setSession] = useState<api.SessionStatus | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  //: The upscaler catalogue. Empty until it lands, and empty if it fails —
  //: which leaves the toolbar's Upscale button disabled rather than
  //: offering something the server would refuse.
  const [upscalers, setUpscalers] = useState<Upscaler[]>([]);
  const [cancelling, setCancelling] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [dialog, setDialog] = useState<Dialog | null>(null);
  // The selected session answered "locked": the studio shows that instead of an
  // empty feed, and polling it again would only be told the same thing.
  const [lockedOut, setLockedOut] = useState<string | null>(null);
  // Re-rendered from when a token is stored or forgotten: `sessionStorage` is
  // not React state, and the sidebar's lock glyphs read it.
  const [, setTokenEpoch] = useState(0);
  const tokensChanged = () => setTokenEpoch((epoch) => epoch + 1);

  // Read once, on the way in: the rail's state belongs to this browser, not to
  // the URL, and a project opened from a bookmark should find the rail as it was
  // left. `"1"` rather than `JSON.parse`, because a corrupt value must read as
  // "not collapsed" instead of throwing on the way to the first render.
  const [railCollapsed, setRailCollapsed] = useState(() => {
    try {
      return window.localStorage?.getItem(RAIL_KEY) === "1";
    } catch {
      return false;
    }
  });

  /**
   * Which view the studio is showing, and the link that asked for it.
   *
   * Two sources, in this order of authority: an explicit `?mode=` in the URL a
   * link was followed to, then the project's remembered preference. The URL wins
   * once and only for the project it arrived with — a shared link that says
   * "gallery" must open in the gallery even if this browser last left that
   * project in the feed, but it must not then re-impose itself on every other
   * project selected afterwards. `wantedMode` is that "once": read at mount,
   * spent on the first project that opens.
   */
  const [mode, setMode] = useState<ViewMode>(DEFAULT_MODE);
  const wantedMode = useRef<ViewMode | null>(
    asMode(new URLSearchParams(window.location.search).get(MODE_PARAM)),
  );

  /**
   * The floating composer's height, published to the cascade.
   *
   * The composer is `position: absolute` over the studio now, so it reserves no
   * room of its own and whatever scrolls under it has to reserve it instead —
   * but its height is not a constant: an attachment, a notice or a wrapped
   * control row all change it. Measured rather than guessed, and handed to the
   * cascade as `--pg-dock-h` on the studio, which `.pg-feed` and `.pg-hero`
   * each spend as bottom padding. A fixed reservation would either hide the
   * newest image behind the glass or leave a gap under it.
   *
   * A callback ref rather than an effect: the dock exists only on the surfaces
   * that render a composer, and it mounts and unmounts with the login gate as
   * well as with `?view=plugin`. A ref callback fires exactly when the element
   * appears and its cleanup when it goes, which an effect keyed on state would
   * have to reconstruct.
   */
  const [dockHeight, setDockHeight] = useState(0);
  const measureDock = useCallback((box: HTMLDivElement | null) => {
    if (!box) {
      setDockHeight(0);
      return;
    }
    // jsdom has no `ResizeObserver`. The consequence is a `0px` reservation in
    // tests, which is the same value the embedded surface uses, so nothing
    // asserted here depends on a measurement the environment cannot make.
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(() => setDockHeight(box.offsetHeight));
    observer.observe(box);
    return () => observer.disconnect();
  }, []);

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
    if (payload) {
      setSessions(payload.sessions);
      setPaused(payload.paused);
    }
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
      setLockedOut((current) => (current === id ? null : current));
      setConnectionError(null);
    } catch (cause) {
      if (cause instanceof Unauthorized) {
        onUnauthorized();
      } else if (cause instanceof Locked) {
        // Not a failure to report: the session asks for its password. A token
        // this tab held is dead (expired, revoked, or the server restarted),
        // so it is dropped rather than sent again. The prompt opens once — a
        // poll that lands while it is open must not stack another.
        api.forgetUnlock(id);
        tokensChanged();
        setGenerations([]);
        setLockedOut(id);
        setDialog((open) => open ?? { kind: "unlock", id, then: "open" });
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

  // Fetched once: the catalogue is a build-time fact of the server, not
  // session state. A failure is swallowed on purpose — see `upscalers`.
  useEffect(() => {
    void api
      .playgroundUpscalers()
      .then((listed) => setUpscalers(listed.upscalers))
      .catch(() => setUpscalers([]));
  }, []);

  // The URL is the only client-side state, and it holds an id, not a history.
  useEffect(() => {
    const wanted = new URLSearchParams(window.location.search).get("session");
    void refreshSessions().then((listed) => {
      if (wanted && listed?.some((entry) => entry.id === wanted)) setSelected(wanted);
    });
  }, [refreshSessions]);

  // `?view=plugin` — the studio alone, for an embedder that owns the controls.
  // Hermes' plugin opens this in its preview pane: it creates the session and
  // drives generation from the chat, so the sidebar (session switching) and the
  // composer (prompt entry) are not just unused here, they are two places to
  // change the same thing. Read once from the URL: which surface this is does
  // not change under the user the way a selected session does.
  const [embedded] = useState(
    () => new URLSearchParams(window.location.search).get("view") === "plugin",
  );
  useEffect(() => {
    const url = new URL(window.location.href);
    if (selected) url.searchParams.set("session", selected);
    else url.searchParams.delete("session");
    window.history.replaceState(null, "", url);
    void refreshDetail();
  }, [selected, refreshDetail]);

  // The opened project brings its own view with it. Keyed on the selection, not
  // written from the switcher's click, because this is the answer to "what was
  // this project left in" — and a project with no answer opens in the default,
  // which is also what a project whose preference could not be read gets.
  useEffect(() => {
    if (!selected) return;
    const asked = wantedMode.current;
    wantedMode.current = null;
    setMode(asked ?? rememberedMode(selected) ?? DEFAULT_MODE);
  }, [selected]);

  // The view in the URL, so a reload and a shared link land on it. Separate from
  // the effect above rather than folded into the `?session=` one: that effect
  // also refetches the project, and switching between two presentations of
  // images already in hand must not cost a request.
  //
  // Not written on the embedded surface: `?view=plugin` renders one view and has
  // no switcher, so a `?mode=` there would be a parameter nothing reads, written
  // into an embedder's URL.
  useEffect(() => {
    if (embedded) return;
    const url = new URL(window.location.href);
    url.searchParams.set(MODE_PARAM, mode);
    window.history.replaceState(null, "", url);
  }, [embedded, mode]);

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
        // Null until this lands, which is why the composer takes null rather
        // than a default: showing an Enhance control and then withdrawing it
        // is worse than showing it a moment late.
        setRewrite(caps.rewrite ?? null);
      }
    });
  }, [guarded]);

  useEffect(() => {
    const stop = api.subscribeProgress(setProgress, () => setProgress(IDLE));
    return stop;
  }, []);

  // Polling, and only while it buys something: a session with nothing in flight
  // changes only when this page changes it, and every mutation refetches.
  // `paused` is part of it: with the queue held and nothing in it, there is
  // nothing "in flight" to poll for, and a tab that stopped polling would never
  // learn that another tab had released the queue.
  const active =
    paused ||
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
    const detail = setInterval(() => {
      if (current.current !== lockedOut) void refreshDetail();
    }, DETAIL_POLL_MS);
    const list = setInterval(() => void refreshSessions(), LIST_POLL_MS);
    return () => {
      clearInterval(detail);
      clearInterval(list);
    };
  }, [active, lockedOut, refreshDetail, refreshSessions]);

  /**
   * The one submission path: create the session on demand, POST, refresh.
   *
   * Shared by the composer and by the per-image toolbar, so a refine or a
   * variation gets the same session-creation, refresh and error handling as a
   * typed prompt rather than a second, drifting copy of it.
   */
  async function post(sessionId: string | null, form: FormData) {
    await submitting(sessionId, (id) => api.playgroundGenerate(id, form));
  }

  /**
   * The submission path itself, parameterised by the call.
   *
   * Everything around a submission is the same whatever was submitted —
   * creating the session on demand, raising `sending`, refreshing both the
   * detail and the sidebar, and turning a 401 into a re-auth. Upscaling reuses
   * it rather than repeating it.
   */
  async function submitting(sessionId: string | null, call: (id: string) => Promise<unknown>) {
    setSending(true);
    setSubmitError(null);
    try {
      // Submitting with nothing selected still creates the project on demand.
      // "New project" is now the deliberate way to make one — named, before
      // anything is generated in it — but a prompt typed straight into an empty
      // studio must not be refused for want of a container, and this is the one
      // path where the server's auto-title from the first prompt is still what
      // names the result.
      const id = sessionId ?? (await api.playgroundSessionCreate()).id;
      await call(id);
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
    if (draft.negativePrompt) form.set("negative_prompt", draft.negativePrompt);
    form.set("model", draft.model);
    form.set("n", String(draft.n));
    if (draft.size) form.set("size", draft.size);
    if (draft.steps !== null) form.set("steps", String(draft.steps));
    if (draft.seed !== null) form.set("seed", String(draft.seed));
    if (draft.image) form.set("image", draft.image);
    // Only when asked. The server decides the rest -- whether it is configured,
    // whether the model can take it, and whether the prompt is already long
    // enough to be left alone -- and answers all of that at admission.
    if (draft.rewrite) form.set("rewrite", "true");
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
      // The image route wants the session's token like everything else; the
      // query form is what a bare `fetch` of a record URL lacks.
      const blob = await fetch(api.imageUrl(image.url, root.sessionId)).then((response) => {
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
   * The selected image, enlarged, joining its own entry.
   *
   * No bytes leave the browser, unlike `refine`: the source is a file the
   * server wrote and can attribute, so it is named rather than re-uploaded.
   */
  async function upscale(
    root: PlaygroundGeneration,
    image: { url: string; seed: number },
    choice: { model: string; scale: number },
  ) {
    await submitting(root.sessionId, (id) =>
      api.playgroundUpscale(id, {
        image: image.url.split("/").pop() ?? "",
        model: choice.model,
        scale: choice.scale,
        group: root.groupId,
      }),
    );
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
        const blob = await fetch(api.imageUrl(root.contextImage, root.sessionId)).then((response) => {
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
    const sessionId = selected;
    if (!sessionId) return;
    await guarded(() => api.playgroundImageDelete(sessionId, filename));
    await refreshDetail();
    // The server bumps the session's `updated_at`, which is both the sidebar's
    // sort key and its timestamp: an idle session polls nothing, so without this
    // the list would keep showing the pre-deletion order for ever.
    await refreshSessions();
  }

  async function deleteGroup(groupId: string) {
    const sessionId = selected;
    if (!sessionId) return;
    await guarded(() => api.playgroundGroupDelete(sessionId, groupId));
    await refreshDetail();
    // The server bumps the session's `updated_at`, which is the sidebar's sort
    // key: without this an idle session would keep its pre-deletion place.
    await refreshSessions();
  }

  /**
   * Hold or release the queue, for every session at once.
   *
   * Set here before the server answers, because the button must not sit inert
   * for a round trip on the one control whose whole point is to take effect now;
   * `refreshSessions` immediately after replaces the guess with the server's
   * answer, and a failure puts it back.
   */
  async function togglePause(next: boolean) {
    setPausing(true);
    setPaused(next);
    const result = await guarded(() => api.playgroundSetPaused(next));
    if (result === undefined) setPaused(!next);
    await refreshSessions();
    setPausing(false);
  }

  async function cancel(id: string) {
    const sessionId = generations.find((entry) => entry.id === id)?.sessionId ?? selected;
    if (!sessionId) return;
    setCancelling(id);
    await guarded(() => api.playgroundCancel(sessionId, id));
    await refreshDetail();
    await refreshSessions();
    setCancelling(null);
  }

  async function remove(id: string) {
    try {
      await api.playgroundSessionDelete(id);
    } catch (cause) {
      if (cause instanceof Unauthorized) onUnauthorized();
      else if (cause instanceof Locked) {
        // The password is the proof the server wants: ask, then delete.
        api.forgetUnlock(id);
        tokensChanged();
        setDialog({ kind: "unlock", id, then: "delete" });
      } else setConnectionError(messageOf(cause));
      return;
    }
    api.forgetUnlock(id);
    if (id === selected) setSelected(null);
    await refreshSessions();
  }

  /**
   * Create a project, named, and open it.
   *
   * Two calls, because the API has no "create with a title": `POST` makes the
   * record and `PATCH` names it. The order is forced — there is nothing to name
   * before the first call answers — and the window between them is the one thing
   * that had to be made safe, since a `PATCH` that fails would otherwise leave a
   * project the server holds and the user never asked for.
   *
   * `created` is what closes that window. The id of a project made but not yet
   * named is kept, the rail is refreshed either way, and a retry from the still
   * open dialog names *that* project instead of creating a second one. So the
   * worst case is a project called "Untitled project", visible in the rail and
   * deletable there — never a record only the server can see.
   *
   * The server's own auto-title (`add_generation` back-fills a NULL title from
   * the first prompt) is why the name is sent before the project can be
   * generated in: a project named here has a title already, so there is nothing
   * left for that back-fill to overwrite.
   */
  const created = useRef<string | null>(null);
  async function create(title: string) {
    const id = created.current ?? (await api.playgroundSessionCreate()).id;
    created.current = id;
    try {
      await api.playgroundSessionRename(id, title);
    } finally {
      await refreshSessions();
    }
    created.current = null;
    setDialog(null);
    setSelected(id);
  }

  // Written where the state changes rather than in an effect: the preference is
  // this click, and an effect would also write it on the first render, replacing
  // what another tab had just stored with what this tab happened to read.
  function toggleRail() {
    const next = !railCollapsed;
    setRailCollapsed(next);
    try {
      window.localStorage?.setItem(RAIL_KEY, next ? "1" : "0");
    } catch {
      // A preference that cannot be written is a preference that is not kept.
    }
  }

  /**
   * Switch view, and remember it for the project this is a view *of*.
   *
   * Written here rather than in an effect on `mode`, for the reason `toggleRail`
   * gives: an effect would also fire when the mode is *adopted* from the store,
   * writing back what was just read, and with nothing selected it would have no
   * project to key on. Nothing but presentation happens — no fetch, no record
   * touched — which is the whole of T1.
   */
  function chooseMode(next: ViewMode) {
    setMode(next);
    if (!selected) return;
    try {
      window.localStorage?.setItem(modeKey(selected), next);
    } catch {
      // A preference that cannot be written is a preference that is not kept.
    }
  }

  async function rename(id: string, title: string | null) {
    await api.playgroundSessionRename(id, title);
    setDialog(null);
    await refreshSessions();
  }

  async function unlock(id: string, password: string, then: "open" | "delete") {
    await api.playgroundSessionUnlock(id, password);
    tokensChanged();
    setDialog(null);
    setLockedOut((current) => (current === id ? null : current));
    if (then === "delete") {
      await remove(id);
      return;
    }
    if (id === selected) await refreshDetail();
    else setSelected(id);
    await refreshSessions();
  }

  async function lock(id: string) {
    await guarded(() => api.playgroundSessionLock(id));
    tokensChanged();
    if (id === selected) {
      setGenerations([]);
      setLockedOut(id);
    }
  }

  async function setPassword(id: string, password: string) {
    await api.playgroundSessionPasswordSet(id, password);
    tokensChanged();
    setDialog(null);
    await refreshSessions();
  }

  async function removePassword(id: string) {
    await api.playgroundSessionPasswordRemove(id);
    tokensChanged();
    setDialog(null);
    await refreshSessions();
  }

  const titleOf = (id: string) => sessions.find((entry) => entry.id === id)?.title ?? null;
  const isLocked = (id: string) => sessions.find((entry) => entry.id === id)?.locked ?? false;
  const srcOf = (url: string) => (selected ? api.imageUrl(url, selected) : url);
  // The tile the gallery scrolls, token and all: a locked project's thumbnails
  // are refused to exactly the caller its full images are refused to, so they
  // need the same `?t=` treatment.
  const thumbOf = (url: string) => (selected ? api.thumbnailUrl(url, selected) : url);

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
      {/* The rail is flush with the window's edge and runs its whole height, so
          the page is this row and the masthead belongs to the stage inside it —
          the identity and the queue's state moved into the rail's own head and
          foot with it (see `SessionList`). */}
      <div className={embedded ? "pg-layout pg-layout-embedded" : "pg-layout"}>
        {!embedded && (
          <SessionList
            sessions={sessions}
            selected={selected}
            collapsed={railCollapsed}
            paused={paused}
            unlocked={(id) => api.unlockToken(id) !== null}
            onSelect={setSelected}
            onNew={() => setDialog({ kind: "new" })}
            onToggleCollapsed={toggleRail}
            onRename={(id) => {
              if (isLocked(id) && api.unlockToken(id) === null)
                setDialog({ kind: "unlock", id, then: "open" });
              else setDialog({ kind: "rename", id });
            }}
            onPassword={(id) => {
              // Renaming and changing a password are proof-gated like everything
              // else on a locked project: unlock first, then the form.
              if (isLocked(id) && api.unlockToken(id) === null)
                setDialog({ kind: "unlock", id, then: "open" });
              else setDialog({ kind: "password", id });
            }}
            onLock={(id) => void lock(id)}
            onDelete={(id) => void remove(id)}
          />
        )}
        <div className="pg-stage">
          {/* The masthead is the stage's now, not the page's, and it holds what
              is about *this* project's view: the switcher and the queue control.
              Who this is and what the server is doing moved to the rail. */}
          <header>
            {/* The switcher, in the masthead rather than over the studio: this
                is the page's one control strip, the tab vocabulary (`.views` /
                `.view-tab`) is the dashboard's own, and a bar of its own above
                the images would spend vertical space the pictures want. Not
                rendered embedded — `?view=plugin` is the studio alone, and the
                pane's owner chose what it shows. */}
            {!embedded && (
              <nav className="views" role="tablist" aria-label="Project views">
                <button
                  role="tab"
                  aria-selected={mode === "prompts"}
                  className="view-tab"
                  onClick={() => chooseMode("prompts")}
                >
                  Prompts
                </button>
                <button
                  role="tab"
                  aria-selected={mode === "gallery"}
                  className="view-tab"
                  onClick={() => chooseMode("gallery")}
                >
                  Gallery
                </button>
                <button
                  role="tab"
                  aria-selected={mode === "table"}
                  className="view-tab"
                  onClick={() => chooseMode("table")}
                >
                  Light Table
                </button>
              </nav>
            )}
            {/* The queue's state, embedded only. It normally lives in the rail's
                footer, which is where the mockup puts it — but `?view=plugin`
                renders no rail, so moving it there took the indicator away from
                the one surface that cannot go looking for it: Hermes' pane has
                no sidebar to expand and no browser chrome to navigate with. The
                paused *notice* still appears below on both surfaces; this is the
                at-a-glance state, which is a different thing from a warning. */}
            {embedded &&
              (paused ? (
                <span className="pill pill-warn">Queue paused</span>
              ) : (
                <span className="pill pill-live">Running</span>
              ))}
            <button
              type="button"
              className="small pg-pause"
              disabled={pausing}
              aria-pressed={paused}
              onClick={() => void togglePause(!paused)}
            >
              {paused ? "Resume queue" : "Pause queue"}
            </button>
          </header>

          {connectionError && (
            <div className="notice notice-error" role="status">
              <strong>Could not reach the server.</strong> {connectionError}
            </div>
          )}

          {/* Every fact a held queue needs to state, and no more: what still
              happens, what does not, and the one way it can lose work. The last
              sentence is not a caveat — the queue lives in memory, so a restart
              fails what is waiting in it. */}
          {paused && (
            <div className="notice notice-warn" role="status">
              <strong>Queue paused.</strong> The image being generated will finish, then nothing
              more starts. Anything you send waits here. Direct API requests are unaffected, and
              a server restart discards what is waiting.
            </div>
          )}

          <section
            className="pg-studio"
            style={{ "--pg-dock-h": `${dockHeight}px` } as CSSProperties}
          >
            {selected && lockedOut === selected ? (
              <div className="pg-hero">
                <span className="pg-hero-icon" aria-hidden="true">
                  🔒
                </span>
                <h2 className="pg-hero-title">This project is locked</h2>
                <p className="pg-hero-tagline">Enter its password to see what it holds.</p>
                <div className="pg-hero-actions">
                  <button
                    className="primary"
                    onClick={() => setDialog({ kind: "unlock", id: selected, then: "open" })}
                  >
                    Unlock
                  </button>
                </div>
              </div>
            ) : generations.length === 0 ? (
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
            ) : mode === "gallery" ? (
              /* `progress` and cancel are here now, and the rule that kept them
                 out is reversed. It read: the gallery shows what exists, and a
                 run in flight has no image yet, so the feed is where a
                 generation is watched. That was extended from an earlier and
                 still-correct rule — no *metadata* on a tile — and the extension
                 was wrong. Nothing was ever lost by it: polling is driven by
                 `active`, which is computed from the generations and not from
                 the mode, so the picture landed in this view too. But a view
                 that gives no sign a submission was accepted reads as a
                 submission that failed, which is what the user reported. T8: in
                 every view, what the server accepted is visible until it
                 completes, fails or is cancelled. The tile is still a box with
                 no facts written on it. */
              <GalleryView
                generations={generations}
                progress={progress}
                paused={paused}
                cancelling={cancelling}
                onCancel={(id) => void cancel(id)}
                busy={sending}
                upscalers={upscalers}
                nameOf={nameOf}
                srcOf={srcOf}
                thumbOf={thumbOf}
                onRefine={(entry, image) => void refine(entry, image)}
                onVariation={(entry) => void variation(entry)}
                onUpscale={(entry, image, choice) => void upscale(entry, image, choice)}
                onDeleteImage={(url) => void deleteImage(url)}
              />
            ) : mode === "table" ? (
              /* `progress` and cancel are here for the same reason they are on
                 the gallery, and the same comment was reversed: "this view shows
                 a picture that exists" left the hero on an old picture with
                 nothing to say why, which is the reported defect at its worst —
                 a stage that looks answered while a request is still running.

                 Keyed on the project, which is the only prop of this kind on the
                 page. The light table holds a selected frame, and that selection
                 is an answer about *this* project's pictures — so it has to die
                 with the project rather than be carried into the next one. The
                 view resolves a selection it no longer recognises to its default
                 frame on its own, which covers a deleted image, a newly landed
                 one, and a run that has just finished; it cannot cover a project
                 switch while the view stays mounted, because both projects being
                 left in this view is exactly the case where nothing unmounts.
                 The key is that unmount, and it keeps the selection out of the
                 state up here. */
              <LightTableView
                key={selected ?? ""}
                generations={generations}
                progress={progress}
                paused={paused}
                cancelling={cancelling}
                onCancel={(id) => void cancel(id)}
                busy={sending}
                upscalers={upscalers}
                nameOf={nameOf}
                srcOf={srcOf}
                thumbOf={thumbOf}
                onRefine={(entry, image) => void refine(entry, image)}
                onVariation={(entry) => void variation(entry)}
                onUpscale={(entry, image, choice) => void upscale(entry, image, choice)}
                onDeleteImage={(url) => void deleteImage(url)}
              />
            ) : (
              <GenerationFeed
                generations={generations}
                progress={progress}
                onCancel={(id) => void cancel(id)}
                cancelling={cancelling}
                busy={sending}
                onRefine={(entry, image) => void refine(entry, image)}
                onVariation={(entry) => void variation(entry)}
                onUpscale={(entry, image, choice) => void upscale(entry, image, choice)}
                upscalers={upscalers}
                onDeleteImage={(url) => void deleteImage(url)}
                onDeleteGroup={(groupId) => void deleteGroup(groupId)}
                onUsePrompt={(text) =>
                  setPresetPrompt((current) => ({ text, nonce: (current?.nonce ?? 0) + 1 }))
                }
                paused={paused}
                nameOf={nameOf}
                srcOf={srcOf}
              />
            )}
            {!embedded && (
              <div className="pg-dock" ref={measureDock}>
                <Composer
                  models={models}
                  defaultModel={defaultModel}
                  maxN={maxN}
                  busy={sending}
                  error={submitError}
                  rewrite={rewrite}
                  presetPrompt={presetPrompt}
                  onSubmit={(draft) => void submit(draft)}
                />
              </div>
            )}
          </section>
        </div>
      </div>

      {dialog?.kind === "new" && (
        <NewProjectDialog
          onCancel={() => {
            setDialog(null);
            // Nothing to undo: `create` is what makes the record, so a cancel
            // before it ran leaves the server holding nothing. A cancel *after*
            // a failed rename leaves an "Untitled project" in the rail, which is
            // a project the user can see and delete.
          }}
          onCreate={create}
        />
      )}
      {dialog?.kind === "rename" && (
        <RenameDialog
          title={titleOf(dialog.id)}
          onCancel={() => setDialog(null)}
          onRename={(title) => rename(dialog.id, title)}
        />
      )}
      {dialog?.kind === "unlock" && (
        <UnlockDialog
          title={titleOf(dialog.id)}
          onCancel={() => {
            setDialog(null);
            // Cancelling on a locked selection leaves the locked hero, which
            // has its own Unlock button; a cancelled delete just stops.
          }}
          onUnlock={(password) => unlock(dialog.id, password, dialog.then)}
        />
      )}
      {dialog?.kind === "password" && (
        <PasswordDialog
          title={titleOf(dialog.id)}
          locked={isLocked(dialog.id)}
          onCancel={() => setDialog(null)}
          onSet={(password) => setPassword(dialog.id, password)}
          onRemove={() => removePassword(dialog.id)}
        />
      )}
    </main>
  );
}

import type { CSSProperties } from "react";

import appIcon from "../assets/app-icon.png";
import type { PlaygroundSession } from "../types";
import { Tool } from "./Tool";

/** "3m ago", coarsely. A sidebar row, not a timestamp anyone reads twice. */
function ago(seconds: number): string {
  const elapsed = Date.now() / 1000 - seconds;
  if (elapsed < 60) return "just now";
  if (elapsed < 3600) return `${Math.floor(elapsed / 60)}m ago`;
  if (elapsed < 86_400) return `${Math.floor(elapsed / 3600)}h ago`;
  return `${Math.floor(elapsed / 86_400)}d ago`;
}

/** What a project is called before anything names it. */
const UNTITLED = "Untitled project";

/**
 * The letter a collapsed rail shows instead of a picture, and the hue it shows
 * it in.
 *
 * The rail collapses to 56px, where no name fits, so each project needs a mark
 * that can be recognised rather than read. The hue is derived from the id, not
 * from the position in the list: the list reorders on every generation — it is
 * sorted by `updated_at` — and a colour that moved with the order would identify
 * the slot instead of the project.
 *
 * This is the *fallback* now. `/playground/api/sessions` carries a `cover` — the
 * project's most recent image, as a thumbnail — and a picture is a better
 * landmark than a letter. But it is `null` for a project with nothing in it, and
 * `null` for a **locked** project: that endpoint answers without an unlock token,
 * and a cover URL carries the `uuid4` filename that is itself the capability to
 * fetch the file, so the server withholds one. Both cases land here, and they
 * land on the same mark — nothing about this landmark says which of the two it
 * is standing in for, which is the point.
 */
function landmarkOf(session: PlaygroundSession): { letter: string; hue: number } {
  const words = (session.title ?? UNTITLED).trim().split(/\s+/);
  // The article is skipped, and this is not cosmetic. A project's title is its
  // first prompt unless the user named it, and prompts begin "A lone
  // lighthouse…", "An Icelandic cabin…", "The rooftop…" — measured on the live
  // store, fourteen of nineteen rows drew the same letter, which is no landmark
  // at all. The hue still separates them; the letter should too.
  const word = words.length > 1 && /^(a|an|the)$/i.test(words[0]!) ? words[1]! : words[0]!;
  let hue = 0;
  for (const character of session.id) hue = (hue * 31 + character.charCodeAt(0)) % 360;
  return { letter: [...word][0]?.toUpperCase() ?? "·", hue };
}

/**
 * The rail: every project this server holds, live ones included.
 *
 * **Creation.** "New project" opens a name dialog and the project is created
 * there and then, before anything has been generated in it. This reverses the
 * rule that stood here before, which was that `+` only cleared the selection and
 * the row was created by the first submission — deliberately, so that clicking
 * `+` repeatedly could not litter the rail with empty conversations.
 *
 * The rule changed because what the rail lists changed. A *session* was born
 * from its first message, so a session with no messages was an artefact of a
 * click and nothing else. A *project* is a container you name, open, and fill,
 * possibly over several sittings; an empty one is a project you have not started
 * yet, which is a legitimate thing to have. The cost of the reversal is accepted
 * rather than hidden: empty projects are permanent, and the way to be rid of one
 * is to delete it, which this rail offers on every row.
 *
 * **Collapse.** The rail collapses to a column of landmarks. Collapsed, a row is
 * a mark and its live/locked state — no name, no timestamp, no actions — because
 * 56px holds no more than that, and the state is remembered per browser by
 * `PlaygroundApp`. The embedded `?view=plugin` surface renders no rail at all, so
 * neither state applies there.
 *
 * **Chrome.** The rail also carries the page's wordmark and, at its foot, the
 * queue's state and the way to the server's own configuration. Those were a
 * masthead across the top of the page; the rail is flush and full height now,
 * so the page has no masthead row for them to sit in — and neither is about the
 * project the studio is showing, which is what the rest of this column is for.
 */
export function SessionList({
  sessions,
  selected,
  collapsed,
  paused,
  unlocked,
  onSelect,
  onNew,
  onToggleCollapsed,
  onRename,
  onPassword,
  onLock,
  onDelete,
}: {
  sessions: PlaygroundSession[];
  selected: string | null;
  /** Landmarks only, no names: the rail is at its narrow width. */
  collapsed: boolean;
  /**
   * The queue is held server-wide, for the footer's pill.
   *
   * Passed in rather than read here: `PlaygroundApp` owns the state, polls it
   * and is what the pause control writes to, so a second reader would be a
   * second answer to one question.
   */
  paused: boolean;
  /** Locked projects this tab holds a token for. */
  unlocked: (id: string) => boolean;
  onSelect: (id: string) => void;
  onNew: () => void;
  onToggleCollapsed: () => void;
  onRename: (id: string) => void;
  onPassword: (id: string) => void;
  onLock: (id: string) => void;
  onDelete: (id: string) => void;
}) {
  return (
    <aside className={collapsed ? "pg-sidebar pg-sidebar-collapsed" : "pg-sidebar"}>
      <div className="pg-sidebar-head">
        {/* The wordmark, in the rail rather than in a masthead over the page.
            The rail is flush with the window edge and runs its full height —
            the mockup's treatment — so the top of the rail is where the product
            names itself, and the page's one `h1` belongs here.

            "QDS", not "Quantum Diffusion Server": the rail is ~280px wide and
            carries a collapse button beside this, so the full name did not fit
            and wrapped or clipped. The accessible name keeps the whole thing —
            a screen reader has no width limit — so shortening the glyphs costs
            nothing there.

            The mark is the app's own icon rather than a coloured square, and it
            is `aria-hidden` for the same reason the square was: it repeats what
            the text already says. Imported rather than written as a path so the
            bundler fingerprints it and resolves the `/dashboard/` base — a bare
            `/app-icon.png` would 404 under the mount. */}
        <h1 className="pg-rail-wordmark">
          <img
            className="pg-rail-icon"
            src={appIcon}
            alt=""
            aria-hidden="true"
            width={22}
            height={22}
          />
          <span className="pg-rail-name" title="Quantum Diffusion Server">
            QDS
          </span>
          {/* The full product name stays the accessible one: a reader has no
              width limit, so the rail's does not have to cost it. */}
          <span className="pg-rail-fullname">Quantum Diffusion Server</span>
        </h1>
        <button
          type="button"
          className="small pg-rail-toggle"
          aria-label={collapsed ? "Expand the project rail" : "Collapse the project rail"}
          aria-expanded={!collapsed}
          title={collapsed ? "Expand the project rail" : "Collapse the project rail"}
          onClick={onToggleCollapsed}
        >
          <svg
            viewBox="0 0 24 24"
            width="15"
            height="15"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <path d={collapsed ? "M9 6l6 6-6 6" : "M15 6l-6 6 6 6"} />
          </svg>
        </button>
        <button
          className="primary small pg-rail-new"
          aria-label="New project"
          title="New project"
          onClick={onNew}
        >
          +
        </button>
      </div>
      {/* A label over the list, on its own line under the wordmark, as the
          mockup has it: the rail's one heading, and it says what the column is
          rather than competing with the project names under it. */}
      {!collapsed && <h2 className="pg-rail-label">Projects</h2>}
      {sessions.length === 0 ? (
        // Collapsed there is no room for a sentence, and an empty rail with a `+`
        // in it says the same thing.
        collapsed ? null : (
          <p className="note">No projects yet. Create one to start.</p>
        )
      ) : (
        <ul className="pg-sessions">
          {sessions.map((session) => {
            const name = session.title ?? UNTITLED;
            const mark = landmarkOf(session);
            return (
              <li key={session.id}>
                {collapsed ? (
                  <button
                    type="button"
                    className="pg-rail-mark"
                    style={{ "--pg-mark-hue": mark.hue } as CSSProperties}
                    aria-current={session.id === selected}
                    aria-label={name}
                    // Native, like the row actions below and for the same
                    // reason: the rail scrolls, and a scroll container clips
                    // anything drawn outside its box.
                    title={name}
                    onClick={() => onSelect(session.id)}
                  >
                    {session.cover ? (
                      // `alt=""` and not the project name: the button already
                      // carries that as its accessible name, and a screen reader
                      // reading it twice is worse than a decorative image.
                      <img className="pg-rail-mark-cover" src={session.cover} alt="" />
                    ) : (
                      <span aria-hidden="true">{mark.letter}</span>
                    )}
                    {session.locked && (
                      <span className="pg-rail-mark-lock" aria-hidden="true">
                        {unlocked(session.id) ? "🔓" : "🔒"}
                      </span>
                    )}
                    {session.generating && (
                      <span className="pg-dot pg-rail-mark-dot" aria-hidden="true" />
                    )}
                  </button>
                ) : (
                  <div className="pg-session-row" aria-selected={session.id === selected}>
                    <button
                      className="pg-session-open"
                      onClick={() => onSelect(session.id)}
                      aria-current={session.id === selected}
                    >
                      {/* The same tile the collapsed rail draws, at the head of
                          the row: the picture is the thing a project is
                          recognised by, and drawing it in one state only would
                          make collapsing the rail a change of vocabulary rather
                          than a change of width. Always present — a cover when
                          the payload has one, the letter-and-hue landmark
                          otherwise — so the row has one layout instead of two. */}
                      <span
                        className="pg-session-thumb"
                        style={{ "--pg-mark-hue": mark.hue } as CSSProperties}
                        aria-hidden="true"
                      >
                        {session.cover ? (
                          <img src={session.cover} alt="" />
                        ) : (
                          mark.letter
                        )}
                      </span>
                      <span className="pg-session-title">
                        {session.locked && (
                          <span
                            className="pg-session-lock"
                            role="img"
                            aria-label={unlocked(session.id) ? "Locked, open in this tab" : "Locked"}
                            title={unlocked(session.id) ? "Locked, open in this tab" : "Locked"}
                          >
                            {unlocked(session.id) ? "🔓" : "🔒"}
                          </span>
                        )}
                        {name}
                      </span>
                      <span className="pg-session-meta">
                        {session.generating && (
                          <span className="pg-dot" aria-label="Generating" title="Generating" />
                        )}
                        {ago(session.updatedAt)}
                      </span>
                    </button>
                    {/* Icons, not words, and drawn *over* the row rather than
                        beside it. Four text buttons wanted about 330px in a 280px
                        rail, and `visibility: hidden` does not free the space a
                        hidden element occupies — so every row's title was squeezed
                        to a few letters by buttons nobody could see, and the ones
                        you could see overhung the panel. Taking them out of flow is
                        what gives the name the whole row back.

                        `native` tooltips because `.pg-sidebar` scrolls, and a
                        scroll container clips anything drawn outside its box. */}
                    <div className="pg-session-actions" role="group" aria-label="Project actions">
                      <Tool
                        tip="Rename"
                        label={`Rename ${session.title ?? "project"}`}
                        native
                        onClick={() => onRename(session.id)}
                      >
                        <path d="M5 19h3.5L19 8.5 15.5 5 5 15.5z" />
                        <path d="M14 6.5 17.5 10" />
                      </Tool>
                      <Tool
                        tip={session.locked ? "Change password" : "Set a password"}
                        label={`${session.locked ? "Change password of" : "Set a password on"} ${session.title ?? "project"}`}
                        native
                        onClick={() => onPassword(session.id)}
                      >
                        <circle cx="8" cy="15" r="3.2" />
                        <path d="M10.4 12.9 19 4.5M15.8 7.7l2.2 2.2" />
                      </Tool>
                      {session.locked && unlocked(session.id) && (
                        <Tool
                          tip="Lock: ask for the password again in this tab"
                          label={`Lock ${session.title ?? "project"}`}
                          native
                          onClick={() => onLock(session.id)}
                        >
                          <rect x="5" y="10.5" width="14" height="9.5" rx="2" />
                          <path d="M8.5 10.5V7a3.5 3.5 0 0 1 7 0v3.5" />
                        </Tool>
                      )}
                      <Tool
                        tip="Delete"
                        label={`Delete ${session.title ?? "project"}`}
                        danger
                        native
                        onClick={() => {
                          // A generation in flight is stopped and its images deleted:
                          // worth a confirmation, and `window.confirm` is what the
                          // rest of this app uses for a destructive one-click action.
                          if (window.confirm("Delete this project and its images?")) {
                            onDelete(session.id);
                          }
                        }}
                      >
                        <path d="M4 7h16M9 7V5h6v2M6.5 7l.8 12h9.4l.8-12M10 11v5M14 11v5" />
                      </Tool>
                    </div>
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      )}
      {/* The footer strip: what the server is doing, and the way off this page.
          Both were in the page masthead, which the flush rail replaces — and
          both are about the server rather than about this project, which is what
          the foot of the rail is for. */}
      <div className="pg-rail-foot">
        {paused ? (
          <span className="pill pill-warn">Queue paused</span>
        ) : (
          <span className="pill pill-live">Running</span>
        )}
        {/* Dropped at 56px: it is a labelled link, and nothing that reads as
            words fits in that column. Expanding the rail is one click. */}
        {!collapsed && (
          <a className="shell-link pg-rail-config" href="/dashboard/?view=config" target="_blank">
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
        )}
      </div>
    </aside>
  );
}

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

/**
 * The history sidebar: every session this server holds, live ones included.
 *
 * "New session" only clears the selection. The session row is created by the
 * first submission, so clicking it repeatedly cannot litter the sidebar with
 * empty conversations.
 */
export function SessionList({
  sessions,
  selected,
  unlocked,
  onSelect,
  onNew,
  onRename,
  onPassword,
  onLock,
  onDelete,
}: {
  sessions: PlaygroundSession[];
  selected: string | null;
  /** Locked sessions this tab holds a token for. */
  unlocked: (id: string) => boolean;
  onSelect: (id: string) => void;
  onNew: () => void;
  onRename: (id: string) => void;
  onPassword: (id: string) => void;
  onLock: (id: string) => void;
  onDelete: (id: string) => void;
}) {
  return (
    <aside className="pg-sidebar">
      <div className="pg-sidebar-head">
        <h2>History</h2>
        <button className="primary small" onClick={onNew}>
          +
        </button>
      </div>
      {sessions.length === 0 ? (
        <p className="note">No sessions yet. Write a prompt to start one.</p>
      ) : (
        <ul className="pg-sessions">
          {sessions.map((session) => (
            <li key={session.id}>
              <div className="pg-session-row" aria-selected={session.id === selected}>
                <button
                  className="pg-session-open"
                  onClick={() => onSelect(session.id)}
                  aria-current={session.id === selected}
                >
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
                    {session.title ?? "New session"}
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
                    sidebar, and `visibility: hidden` does not free the space a
                    hidden element occupies — so every row's title was squeezed
                    to a few letters by buttons nobody could see, and the ones
                    you could see overhung the panel. Taking them out of flow is
                    what gives the name the whole row back.

                    `native` tooltips because `.pg-sidebar` scrolls, and a
                    scroll container clips anything drawn outside its box. */}
                <div className="pg-session-actions" role="group" aria-label="Session actions">
                  <Tool
                    tip="Rename"
                    label={`Rename ${session.title ?? "session"}`}
                    native
                    onClick={() => onRename(session.id)}
                  >
                    <path d="M5 19h3.5L19 8.5 15.5 5 5 15.5z" />
                    <path d="M14 6.5 17.5 10" />
                  </Tool>
                  <Tool
                    tip={session.locked ? "Change password" : "Set a password"}
                    label={`${session.locked ? "Change password of" : "Set a password on"} ${session.title ?? "session"}`}
                    native
                    onClick={() => onPassword(session.id)}
                  >
                    <circle cx="8" cy="15" r="3.2" />
                    <path d="M10.4 12.9 19 4.5M15.8 7.7l2.2 2.2" />
                  </Tool>
                  {session.locked && unlocked(session.id) && (
                    <Tool
                      tip="Lock: ask for the password again in this tab"
                      label={`Lock ${session.title ?? "session"}`}
                      native
                      onClick={() => onLock(session.id)}
                    >
                      <rect x="5" y="10.5" width="14" height="9.5" rx="2" />
                      <path d="M8.5 10.5V7a3.5 3.5 0 0 1 7 0v3.5" />
                    </Tool>
                  )}
                  <Tool
                    tip="Delete"
                    label={`Delete ${session.title ?? "session"}`}
                    danger
                    native
                    onClick={() => {
                      // A generation in flight is stopped and its images deleted:
                      // worth a confirmation, and `window.confirm` is what the
                      // rest of this app uses for a destructive one-click action.
                      if (window.confirm("Delete this session and its images?")) {
                        onDelete(session.id);
                      }
                    }}
                  >
                    <path d="M4 7h16M9 7V5h6v2M6.5 7l.8 12h9.4l.8-12M10 11v5M14 11v5" />
                  </Tool>
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}
    </aside>
  );
}

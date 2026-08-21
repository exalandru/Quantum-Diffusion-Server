import type { PlaygroundSession } from "../types";

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
                {/* Revealed with the row, like Delete was: the sidebar is a list
                    of names, not a toolbar per line. */}
                <div className="pg-session-actions" role="group" aria-label="Session actions">
                  <button
                    className="small"
                    aria-label={`Rename ${session.title ?? "session"}`}
                    title="Rename"
                    onClick={() => onRename(session.id)}
                  >
                    Rename
                  </button>
                  <button
                    className="small"
                    aria-label={`${session.locked ? "Change password of" : "Set a password on"} ${session.title ?? "session"}`}
                    title={session.locked ? "Change password" : "Set a password"}
                    onClick={() => onPassword(session.id)}
                  >
                    {session.locked ? "Password…" : "Protect"}
                  </button>
                  {session.locked && unlocked(session.id) && (
                    <button
                      className="small"
                      aria-label={`Lock ${session.title ?? "session"}`}
                      title="Lock: ask for the password again in this tab"
                      onClick={() => onLock(session.id)}
                    >
                      Lock
                    </button>
                  )}
                  <button
                    className="small danger pg-session-delete"
                    aria-label={`Delete ${session.title ?? "session"}`}
                    onClick={() => {
                      // A generation in flight is stopped and its images deleted:
                      // worth a confirmation, and `window.confirm` is what the
                      // rest of this app uses for a destructive one-click action.
                      if (window.confirm("Delete this session and its images?")) {
                        onDelete(session.id);
                      }
                    }}
                  >
                    Delete
                  </button>
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}
    </aside>
  );
}

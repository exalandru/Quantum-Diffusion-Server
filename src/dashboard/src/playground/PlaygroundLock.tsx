/**
 * The playground's password, asked for when the server demands it.
 *
 * A screen of its own rather than `LoginPrompt` with a flag, because the two
 * ask for two different secrets: the admin password opens the configuration
 * writer, the logs and the restart button, and sending someone here to type it
 * would be asking for more authority than generating an image needs. It also
 * cannot offer to *set* the first one — that is an act of administration, done
 * from the dashboard's Configuration screen — so the "no password yet" state is
 * a dead end here, and says so instead of pretending.
 *
 * `embedded` is `?view=plugin`: the same form, sized for a pane that is a few
 * hundred pixels wide and has no browser chrome around it. The compact variant
 * exists because that surface is where the lock is *most* likely to be met — the
 * plugin's pane is a fresh browsing context holding no cookie.
 */
import { useState } from "react";

import * as api from "../api";
import { messageOf } from "../api";

export function PlaygroundLock({
  status,
  embedded,
  onAuthenticated,
}: {
  status: api.PlaygroundSessionStatus;
  /** `?view=plugin`: no page header, tighter card. */
  embedded: boolean;
  onAuthenticated: () => void;
}) {
  const [value, setValue] = useState("");
  const [note, setNote] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    const password = value.trim();
    if (!password || busy) return;
    setBusy(true);
    setNote(null);
    try {
      await api.playgroundLogIn(password);
      onAuthenticated();
    } catch (cause) {
      // A throttle and a wrong password read the same way here on purpose: the
      // server's own message distinguishes them, and inventing a second
      // sentence for one of them is how the two drift apart.
      setNote(messageOf(cause));
    } finally {
      setBusy(false);
    }
  }

  const card = (
    <section className={embedded ? "card login-card login-card-compact" : "card login-card"}>
      <h2>Playground password</h2>
      {status.passwordSet ? (
        <>
          <p className="muted">
            This server asks for its playground password. It is not the admin password and not the
            API key: it opens the playground, and nothing else.
          </p>
          <form onSubmit={submit}>
            <div className="login-field">
              <label htmlFor="playground-password">Password</label>
              <input
                id="playground-password"
                type="password"
                autoComplete="current-password"
                autoFocus
                value={value}
                onChange={(event) => setValue(event.target.value)}
              />
            </div>
            <button type="submit" className="primary" disabled={!value.trim() || busy}>
              {busy ? "Working…" : "Unlock"}
            </button>
          </form>
        </>
      ) : (
        /* `playground_auth_scope` demands a password that no file holds. The
           server refuses to start in that state, so this is reachable only in a
           process that outlived the change — but a form with nothing to check
           against would be a dead end that looked like a working one. */
        <p className="muted">
          This server requires a playground password and none is set. Set one from the dashboard's
          Configuration screen, under Server, or put <code>playground_auth_scope</code> back to{" "}
          <code>network</code>.
        </p>
      )}

      {note && (
        <p className="setting-error" role="status">
          {note}
        </p>
      )}

      {status.passwordSet && (
        <p className="setting-help">
          Forgotten it? Whoever holds the admin password can set a new one from the dashboard's
          Configuration screen; an admin session opens the playground on its own.
        </p>
      )}
    </section>
  );

  if (embedded) return <main className="unreachable login login-embedded">{card}</main>;

  return (
    <main className="unreachable login">
      <header>
        <h1>Playground</h1>
        <span className="pill pill-down">Locked</span>
      </header>
      {card}
    </main>
  );
}

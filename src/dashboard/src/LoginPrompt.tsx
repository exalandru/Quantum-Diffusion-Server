/**
 * The admin password, asked for when the server demands it.
 *
 * Distinct from the API key, and the distinction is the point: the key is what
 * you hand to Open WebUI so it can generate images, and it deliberately no
 * longer opens the control plane. Handing an image generator the configuration
 * writer, the log buffer and the restart button was never what anyone meant by
 * "here is my key".
 *
 * Two modes, chosen by the server's answer rather than guessed: on a server
 * with no password yet, this offers to set one; otherwise it asks.
 */
import { useState } from "react";

import * as api from "./api";
import { TooManyAttempts, messageOf } from "./api";

export function LoginPrompt({
  status,
  onAuthenticated,
}: {
  status: api.SessionStatus;
  onAuthenticated: () => void;
}) {
  const [value, setValue] = useState("");
  const [note, setNote] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const first = !status.passwordSet;

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    const password = value.trim();
    if (!password || busy) return;
    setBusy(true);
    setNote(null);
    try {
      if (first) await api.setPassword(password);
      await api.logIn(password);
      onAuthenticated();
    } catch (cause) {
      // A throttle is not a wrong password, and saying so stops someone
      // retyping a password that was right.
      setNote(cause instanceof TooManyAttempts ? messageOf(cause) : messageOf(cause));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="unreachable">
      <header>
        <h1>Quantum Diffusion Server</h1>
        <span className="pill pill-down">{first ? "Unprotected" : "Locked"}</span>
      </header>

      <section className="card">
        <h2>{first ? "Choose an admin password" : "Admin password"}</h2>
        <p className="muted">
          {first ? (
            <>
              This server has no admin password. Setting one protects the configuration, the
              catalogue, the logs and the restart button. It is <strong>not</strong> the API key —
              that one stays for <code>/v1</code>, so anything you point at this server to generate
              images cannot also reconfigure it.
            </>
          ) : (
            <>
              Not the API key: that one is for <code>/v1</code>. This is the password that protects
              the control plane.
            </>
          )}
        </p>

        <form onSubmit={submit}>
          <label htmlFor="admin-password">{first ? "New password" : "Password"}</label>
          <input
            id="admin-password"
            type="password"
            autoComplete={first ? "new-password" : "current-password"}
            autoFocus
            value={value}
            onChange={(event) => setValue(event.target.value)}
          />
          <button type="submit" className="primary" disabled={!value.trim() || busy}>
            {busy ? "Working…" : first ? "Set password" : "Unlock"}
          </button>
        </form>

        {note && (
          <p className="setting-error" role="status">
            {note}
          </p>
        )}

        {first && (
          <p className="setting-help">
            At least 8 characters. Forgotten it? The menubar app and the command line use a token
            file on this machine instead, so they can always reach the server to reset it.
          </p>
        )}
      </section>
    </main>
  );
}

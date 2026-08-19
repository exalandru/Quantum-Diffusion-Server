/**
 * What this page shows when the server that served it has gone away.
 *
 * Which is a state it can genuinely be in: the Restart button below stops the
 * server for a moment, the menubar app can stop it outright, and a browser tab
 * left open overnight outlives any number of restarts.
 *
 * It says how to start one rather than only that it could not connect, because
 * there is nothing on this page that can start it — that authority belongs to
 * the menubar app, or to whoever typed `qds serve`. The shell keeps polling
 * behind this, so it disappears on its own when the server answers again.
 */
export function Unreachable({ reason }: { reason: string }) {
  return (
    <main className="unreachable">
      <header>
        <h1>Quantum Diffusion Server</h1>
        <span className="pill pill-down">Not responding</span>
      </header>

      <section className="card">
        <h2>The server is not answering.</h2>
        <p className="muted">{reason}</p>
        <p>
          Start it from the <strong>QDS</strong> menubar app, or run:
        </p>
        <pre>
          <code>qds serve</code>
        </pre>
        <p className="muted">
          This page reconnects on its own; there is nothing to click. If a restart was just
          requested, it should return within a few seconds.
        </p>
      </section>
    </main>
  );
}

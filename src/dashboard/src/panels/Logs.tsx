import { useEffect, useMemo, useRef, useState } from "react";

import type { LogEntry } from "../types";

const LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"] as const;
const RANK: Record<string, number> = { DEBUG: 0, INFO: 1, WARNING: 2, ERROR: 3, CRITICAL: 3 };

/**
 * Viewer for the server's own log tail.
 *
 * The desktop app read two channels here and had to tell them apart: stdout
 * carried JSON Lines and stderr human-readable text, because mflux renders its
 * denoising bar with tqdm as carriage-return fragments that made the JSON
 * unparsable when the two shared a stream. That parsing is gone — the server
 * keeps structured records now, so a level and an optional event arrive as
 * fields rather than as something to guess at.
 *
 * The pane fills what is left of the window rather than a fixed card, and
 * scrolls inside itself. That is a cascade arrangement, not a component one —
 * see the `main.view-logs` rule in `styles.css`, whose three declarations are
 * all load-bearing.
 */
export function Logs({
  entries,
  dropped,
  onClear,
}: {
  entries: LogEntry[];
  /** Lines that fell out of the server's ring buffer between polls. */
  dropped: number;
  onClear: () => void;
}) {
  const [minLevel, setMinLevel] = useState<(typeof LEVELS)[number]>("INFO");
  const [showRaw, setShowRaw] = useState(true);
  const [follow, setFollow] = useState(true);
  const container = useRef<HTMLDivElement>(null);

  const visible = useMemo(
    () =>
      entries.filter((entry) =>
        // `showRaw` now means "include the lines that carry no structured
        // event" — child output and library chatter — rather than "include the
        // other stream".
        entry.event ? (RANK[entry.level] ?? 1) >= RANK[minLevel]! : showRaw,
      ),
    [entries, minLevel, showRaw],
  );

  useEffect(() => {
    if (follow) container.current?.scrollTo({ top: container.current.scrollHeight });
  }, [visible.length, follow]);

  return (
    <section className="panel logs">
      <div className="log-toolbar">
        <h2 style={{ margin: 0 }}>Logs</h2>
        <span className="library-spec" style={{ marginLeft: 0 }}>
          {visible.length} of {entries.length}
          {dropped > 0 && <> · {dropped} dropped</>}
        </span>

        <div className="spacer" />

        <select
          aria-label="Minimum level"
          value={minLevel}
          onChange={(event) => setMinLevel(event.target.value as never)}
        >
          {LEVELS.map((level) => (
            <option key={level} value={level}>
              {level} and above
            </option>
          ))}
        </select>
        <label className="check">
          <input
            type="checkbox"
            checked={showRaw}
            onChange={(event) => setShowRaw(event.target.checked)}
          />
          <span>raw text</span>
        </label>
        <label className="check">
          <input
            type="checkbox"
            checked={follow}
            onChange={(event) => setFollow(event.target.checked)}
          />
          <span>follow</span>
        </label>
        <button className="small" onClick={onClear} disabled={entries.length === 0}>
          Clear
        </button>
      </div>

      {visible.length === 0 ? (
        <p className="empty">
          {entries.length === 0
            ? "Nothing yet. Server and job output appears here."
            : "Every line is filtered out by the controls above."}
        </p>
      ) : (
        <div className="console" ref={container}>
          {/* Keyed by `seq`, which the server assigns and never reuses. An
              array index would re-key every line each time the buffer trims. */}
          {visible.map((entry) => (
            <div className={`line lv-${entry.level}`} key={entry.seq}>
              <span className="ts">{entry.ts.slice(11)}</span>
              <span>
                {entry.message}
                {entry.event && <span className="ts"> ({entry.event})</span>}
              </span>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

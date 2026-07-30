import { useEffect, useMemo, useRef, useState } from "react";

import type { LogEvent, ServerLine } from "../types";

const LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"] as const;
const RANK: Record<string, number> = { DEBUG: 0, INFO: 1, WARNING: 2, ERROR: 3, CRITICAL: 3 };

type Entry =
  | { kind: "event"; event: LogEvent }
  /** Raw stderr text: tqdm bars, uvicorn startup, tracebacks. */
  | { kind: "raw"; line: string };

/**
 * Viewer for the server's two channels.
 *
 * stdout carries only JSON Lines — hence the level filtering — and stderr the
 * human-readable text. That separation is not cosmetic: mflux renders its
 * denoising bar with tqdm, as fragments terminated by a carriage return with no
 * newline, which made the JSON impossible to parse when the two shared stderr.
 */
export function Logs({ lines, onClear }: { lines: ServerLine[]; onClear: () => void }) {
  const [minLevel, setMinLevel] = useState<(typeof LEVELS)[number]>("INFO");
  const [showRaw, setShowRaw] = useState(true);
  const [follow, setFollow] = useState(true);
  const container = useRef<HTMLDivElement>(null);

  const entries = useMemo<Entry[]>(() => {
    const result: Entry[] = [];
    for (const { structured, line } of lines) {
      if (structured) {
        try {
          result.push({ kind: "event", event: JSON.parse(line) as LogEvent });
          continue;
        } catch {
          // A stdout line that fails to parse stays displayable as-is.
        }
      }
      result.push({ kind: "raw", line });
    }
    return result;
  }, [lines]);

  const visible = useMemo(
    () =>
      entries.filter((entry) =>
        entry.kind === "event"
          ? (RANK[entry.event.level] ?? 1) >= RANK[minLevel]!
          : showRaw,
      ),
    [entries, minLevel, showRaw],
  );

  useEffect(() => {
    if (follow) container.current?.scrollTo({ top: container.current.scrollHeight });
  }, [visible.length, follow]);

  return (
    <div className="card">
      <div className="row spread">
        <h2 style={{ margin: 0 }}>Logs</h2>
        <div className="row">
          <select value={minLevel} onChange={(event) => setMinLevel(event.target.value as never)}>
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
          <button onClick={onClear} disabled={lines.length === 0}>
            Clear
          </button>
        </div>
      </div>

      {visible.length === 0 ? (
        <p className="center-note">
          Nothing yet. Server and conversion output appears here.
        </p>
      ) : (
        <div className="console" ref={container} style={{ maxHeight: "calc(100vh - 230px)" }}>
          {visible.map((entry, index) =>
            entry.kind === "event" ? (
              <div className={`line lv-${entry.event.level}`} key={index}>
                <span className="ts">{entry.event.ts.slice(11)}</span>
                <span>
                  {entry.event.message}
                  {entry.event.event && <span className="ts"> ({entry.event.event})</span>}
                </span>
              </div>
            ) : (
              <div className="line raw" key={index}>
                <span className="ts">·</span>
                <span>{entry.line}</span>
              </div>
            ),
          )}
        </div>
      )}
    </div>
  );
}

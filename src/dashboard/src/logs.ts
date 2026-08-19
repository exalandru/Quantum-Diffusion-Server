/**
 * The server's log tail, polled with a cursor.
 *
 * The desktop app received every line as a Tauri event and kept the last two
 * thousand in React state. A web page has no such channel, so the buffer lives
 * in the server and this asks for whatever it has not seen — `after` the last
 * `seq` it holds.
 *
 * That shape is what makes the poll safe to miss: it is idempotent, a
 * backgrounded tab resumes exactly where it stopped, and when the server's ring
 * buffer has moved past us it says so with `dropped` rather than silently
 * handing back a gap.
 */
import { useCallback, useEffect, useRef, useState } from "react";

import * as api from "./api";
import type { LogEntry } from "./types";

/** Kept in the page. The server keeps its own bound; this one is the renderer's. */
export const LOG_LIMIT = 2000;

/** How often to ask. Slow enough to be cheap, fast enough to read along. */
export const POLL_MS = 2000;

export type LogsState = {
  entries: LogEntry[];
  /** Lines lost between polls, cumulative since the last clear. */
  dropped: number;
  clear: () => void;
};

export function useLogs(): LogsState {
  const [entries, setEntries] = useState<LogEntry[]>([]);
  const [dropped, setDropped] = useState(0);
  //  A ref, not state: the poller reads it every tick, and putting it in the
  //  dependency list would tear down and rebuild the interval on every page of
  //  logs — which is how a poll ends up firing far more often than its interval.
  const cursor = useRef(0);

  useEffect(() => {
    let cancelled = false;

    const poll = async () => {
      try {
        const page = await api.logsAfter(cursor.current);
        if (cancelled) return;
        cursor.current = page.lastSeq;
        if (page.dropped) setDropped((total) => total + page.dropped);
        if (page.entries.length) {
          setEntries((previous) => {
            const next = [...previous, ...page.entries];
            return next.length > LOG_LIMIT ? next.slice(-LOG_LIMIT) : next;
          });
        }
      } catch {
        // Deliberately silent. The shell already reports "cannot reach the
        // server" from its own poll; a second banner saying the same thing
        // teaches nothing, and a log view is not where an outage is diagnosed.
      }
    };

    void poll();
    const timer = setInterval(() => void poll(), POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  const clear = useCallback(() => {
    // Only what this page holds. The server's buffer is not ours to empty:
    // another tab, or the same one after a reload, would lose history it never
    // asked to discard.
    setEntries([]);
    setDropped(0);
  }, []);

  return { entries, dropped, clear };
}

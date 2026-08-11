/**
 * The Rust-owned long operation, polled once for the whole application.
 *
 * Rust is the authority (Slice 2): a download or a conversion belongs to the
 * supervisor, not to a React component, and survives the panel that started it.
 * The poll lives here — above the views — for one reason the panels cannot
 * provide themselves: switching to Logs and back used to unmount `Models`, stop
 * its poll, and re-enter with no operation on screen until the next tick. The
 * state a user walks away from has to be the state they walk back to.
 *
 * Nothing here decides anything. It reads a status and hands it on.
 */
import { useCallback, useEffect, useRef, useState } from "react";

import * as api from "./api";
import { messageOf } from "./api";
import type { JobStatus } from "./types";

/** How often we ask Rust what the long operation is doing. */
const POLL_MS = 1000;

export function isActive(job: JobStatus | null): boolean {
  return job?.state === "running" || job?.state === "cancelling";
}

/**
 * One line describing what the operation is doing, from its structured stream.
 *
 * The child's own event names and fields, verbatim — this maps them to English
 * and invents nothing. A percentage is deliberately absent: `mflux_save` writes
 * a model in one call and reports phases, so a fraction here would be a number
 * we made up.
 */
export function describeJob(job: JobStatus): string {
  const fields = job.fields ?? {};
  if (job.event === "prequantize_progress") {
    const { block, blocks } = fields as { block?: number; blocks?: number };
    if (block && blocks) return `quantizing block ${block} of ${blocks}`;
  }
  if (job.event === "prequantize_component_start") {
    const { component } = fields as { component?: string };
    if (component) return `reading ${component}`;
  }
  if (job.event === "prequantize_component_done") {
    const { component, written_gb } = fields as { component?: string; written_gb?: number };
    if (component) return `wrote ${component}${written_gb ? ` (${written_gb} GB)` : ""}`;
  }
  return job.message ?? "starting…";
}

export type JobView = {
  job: JobStatus | null;
  /** Rust could not be asked. Not the same as "no job is running". */
  error: string | null;
  active: boolean;
  /** Read it again now, rather than waiting for the next tick. */
  refresh: () => Promise<void>;
  /** Fires when a job reaches a terminal state, so views can reload what it changed. */
  onSettled: (listener: (job: JobStatus) => void) => () => void;
};

export function useJob(): JobView {
  const [job, setJob] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const previous = useRef<JobStatus | null>(null);
  const listeners = useRef(new Set<(job: JobStatus) => void>());

  const refresh = useCallback(async () => {
    try {
      const status = await api.jobStatus();
      setJob(status);
      setError(null);
      const before = previous.current;
      // The edge, not the level: a completed job stays completed until the next
      // one starts, and firing on every tick would reload the catalogue forever.
      if (before && isActive(before) && !isActive(status)) {
        for (const listener of listeners.current) listener(status);
      }
      previous.current = status;
    } catch (cause) {
      setError(messageOf(cause));
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = setInterval(() => void refresh(), POLL_MS);
    return () => clearInterval(timer);
  }, [refresh]);

  const onSettled = useCallback((listener: (job: JobStatus) => void) => {
    listeners.current.add(listener);
    return () => {
      listeners.current.delete(listener);
    };
  }, []);

  return { job, error, active: isActive(job), refresh, onSettled };
}

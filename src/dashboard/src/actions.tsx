/**
 * State for user-initiated actions, kept strictly apart from background polling.
 *
 * `App` used to own a single `error`, handed to every panel as `onError`, while the
 * four-second status poll cleared that same variable on success. Any message an
 * action produced — a refused start, a rejected save, a 401 on a gated download —
 * was therefore erased within four seconds, with no user action and no trace. Two
 * writers sharing one variable is the whole bug.
 *
 * So each action owns a slot, keyed by name, and only its own call writes to it.
 * A map in `useState` rather than a store: the requirement is separation, not
 * architecture, and the panels are the only readers.
 */
import { useCallback, useEffect, useRef, useState } from "react";

import { messageOf } from "./api";

export type ActionState =
  | { status: "idle" }
  | { status: "busy" }
  | { status: "ok"; message: string }
  | { status: "error"; message: string };

const IDLE: ActionState = { status: "idle" };

/**
 * How long a success message stays before it retires itself.
 *
 * Only success. A confirmation that is still on screen ten operations later has
 * stopped describing anything, and a column of them is what the eye learns to
 * skip — including the failure in the middle. An error is the opposite case: it
 * is the only record that something did not happen, so it stays until the same
 * action runs again or someone dismisses it. Nothing on a timer, and in
 * particular nothing a background poll does, may clear one.
 */
export const SUCCESS_NOTE_MS = 5000;

export function useActions() {
  const [states, setStates] = useState<Record<string, ActionState>>({});
  /** Pending retirement per key, so a new run supersedes the previous one's. */
  const timers = useRef(new Map<string, ReturnType<typeof setTimeout>>());
  /** Which run owns the key now: a late timer from an older run must not fire. */
  const tickets = useRef(new Map<string, number>());

  const clearTimer = useCallback((key: string) => {
    const timer = timers.current.get(key);
    if (timer !== undefined) {
      clearTimeout(timer);
      timers.current.delete(key);
    }
  }, []);

  // The timers outlive nothing: a panel unmounted mid-flight would otherwise
  // wake up to set state on a component that is gone.
  useEffect(() => {
    const pending = timers.current;
    return () => {
      for (const timer of pending.values()) clearTimeout(timer);
      pending.clear();
    };
  }, []);

  /**
   * Run one action and record its outcome under `key`.
   *
   * Every update is keyed, so starting a download cannot clear the result of the
   * conversion sitting above it — and one key holds exactly one outcome, so the
   * same button pressed twice replaces its result rather than stacking a second.
   * Returns whether it succeeded, for callers that need to chain.
   */
  const run = useCallback(
    async (key: string, action: () => Promise<unknown>, success?: string): Promise<boolean> => {
      clearTimer(key);
      const ticket = (tickets.current.get(key) ?? 0) + 1;
      tickets.current.set(key, ticket);
      setStates((previous) => ({ ...previous, [key]: { status: "busy" } }));
      try {
        await action();
        setStates((previous) => ({
          ...previous,
          [key]: success ? { status: "ok", message: success } : IDLE,
        }));
        if (success) {
          timers.current.set(
            key,
            setTimeout(() => {
              timers.current.delete(key);
              // A newer run owns the key: its outcome is not this one's to clear.
              if (tickets.current.get(key) !== ticket) return;
              setStates((previous) =>
                previous[key]?.status === "ok" ? { ...previous, [key]: IDLE } : previous,
              );
            }, SUCCESS_NOTE_MS),
          );
        }
        return true;
      } catch (cause) {
        setStates((previous) => ({
          ...previous,
          [key]: { status: "error", message: messageOf(cause) },
        }));
        return false;
      }
    },
    [clearTimer],
  );

  const dismiss = useCallback(
    (key: string) => {
      clearTimer(key);
      setStates((previous) => ({ ...previous, [key]: IDLE }));
    },
    [clearTimer],
  );

  const stateOf = useCallback((key: string): ActionState => states[key] ?? IDLE, [states]);
  const busy = useCallback((key: string) => stateOf(key).status === "busy", [stateOf]);
  /** True while any action in this panel is running: the panels serialize their buttons. */
  const anyBusy = Object.values(states).some((state) => state.status === "busy");

  return { run, dismiss, stateOf, busy, anyBusy };
}

/**
 * The outcome of one action, rendered where the action lives.
 *
 * Deliberately inline next to its button rather than in the shell's banner: that
 * banner is now the background channel, and the point of this slice is that a
 * reader can tell which of the two they are looking at.
 */
export function ActionNote({
  state,
  onDismiss,
}: {
  state: ActionState;
  onDismiss?: () => void;
}) {
  if (state.status !== "ok" && state.status !== "error") return null;
  return (
    <div className={`action-note ${state.status}`} role="status">
      <span>{state.message}</span>
      {onDismiss && (
        <button type="button" className="dismiss" onClick={onDismiss} aria-label="Dismiss">
          ×
        </button>
      )}
    </div>
  );
}

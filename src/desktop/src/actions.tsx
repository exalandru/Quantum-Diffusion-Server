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
import { useCallback, useState } from "react";

import { messageOf } from "./api";

export type ActionState =
  | { status: "idle" }
  | { status: "busy" }
  | { status: "ok"; message: string }
  | { status: "error"; message: string };

const IDLE: ActionState = { status: "idle" };

export function useActions() {
  const [states, setStates] = useState<Record<string, ActionState>>({});

  /**
   * Run one action and record its outcome under `key`.
   *
   * Every update is keyed, so starting a download cannot clear the result of the
   * token save sitting above it. Returns whether it succeeded, for callers that
   * need to chain.
   */
  const run = useCallback(
    async (key: string, action: () => Promise<unknown>, success?: string): Promise<boolean> => {
      setStates((previous) => ({ ...previous, [key]: { status: "busy" } }));
      try {
        await action();
        setStates((previous) => ({
          ...previous,
          [key]: success ? { status: "ok", message: success } : IDLE,
        }));
        return true;
      } catch (cause) {
        setStates((previous) => ({
          ...previous,
          [key]: { status: "error", message: messageOf(cause) },
        }));
        return false;
      }
    },
    [],
  );

  const dismiss = useCallback((key: string) => {
    setStates((previous) => ({ ...previous, [key]: IDLE }));
  }, []);

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

/**
 * How long an action's result is allowed to stay.
 *
 * Slice 1 separated action results from the background poll, which is what stops
 * a message being erased four seconds after it appears. This is the other end of
 * the same question: a confirmation that never leaves is a message nobody reads,
 * and a column of them is what teaches the eye to skip the one failure among
 * them. So success retires itself and failure does not — the asymmetry is the
 * point, and it is asserted in both directions.
 */
import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import { ActionNote, SUCCESS_NOTE_MS, useActions } from "./actions";

/** One panel with two buttons sharing a key, and one with its own. */
function Harness({
  first,
  second,
}: {
  first: () => Promise<unknown>;
  second?: () => Promise<unknown>;
}) {
  const { run, dismiss, stateOf } = useActions();
  return (
    <>
      <button onClick={() => void run("a", first, "First done.")}>a</button>
      <button onClick={() => void run("a", second ?? first, "Second done.")}>a again</button>
      <button onClick={() => void run("b", second ?? first, "Other done.")}>b</button>
      <ActionNote state={stateOf("a")} onDismiss={() => dismiss("a")} />
      <ActionNote state={stateOf("b")} onDismiss={() => dismiss("b")} />
    </>
  );
}

/** Click, and let the action's promise settle inside `act`. */
async function press(name: string) {
  await act(async () => {
    fireEvent.click(screen.getByRole("button", { name }));
    await vi.advanceTimersByTimeAsync(0);
  });
}

/** Let time pass with the renders it causes flushed. */
async function wait(ms: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

it("retires a success after about five seconds", async () => {
  render(<Harness first={async () => {}} />);

  await press("a");
  expect(screen.getByText("First done.")).toBeTruthy();

  await wait(SUCCESS_NOTE_MS - 1000);
  expect(screen.getByText("First done.")).toBeTruthy();

  await wait(1000);
  expect(screen.queryByText("First done.")).toBeNull();
});

it("keeps a failure until it is replaced or dismissed", async () => {
  render(
    <Harness
      first={async () => {
        throw new Error("it did not work");
      }}
    />,
  );

  await press("a");
  expect(screen.getByText("it did not work")).toBeTruthy();

  // Far longer than any success would have lasted.
  await wait(10 * SUCCESS_NOTE_MS);
  expect(screen.getByText("it did not work")).toBeTruthy();

  fireEvent.click(screen.getByRole("button", { name: "Dismiss" }));
  expect(screen.queryByText("it did not work")).toBeNull();
});

it("holds one result per operation rather than a second copy", async () => {
  render(<Harness first={async () => {}} />);

  await press("a");
  await press("a again");

  // Same key, so the later result replaced the earlier one.
  expect(screen.queryByText("First done.")).toBeNull();
  expect(screen.getByText("Second done.")).toBeTruthy();

  // A different operation is a different slot, and both are ephemeral.
  await press("b");
  expect(screen.getAllByRole("status")).toHaveLength(2);
  await wait(SUCCESS_NOTE_MS);
  expect(screen.queryAllByRole("status")).toHaveLength(0);
});

it("does not let a retired success take a newer result with it", async () => {
  // The race the ticket exists for: the first run's timer is still pending when
  // the second run writes its own outcome under the same key.
  render(<Harness first={async () => {}} />);

  await press("a");
  await wait(SUCCESS_NOTE_MS - 500);
  await press("a again");

  // The first run's timer fires here, and must not clear the second's message.
  await wait(600);
  expect(screen.getByText("Second done.")).toBeTruthy();

  await wait(SUCCESS_NOTE_MS);
  expect(screen.queryByText("Second done.")).toBeNull();
});

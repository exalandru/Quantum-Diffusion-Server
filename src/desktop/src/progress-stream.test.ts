/**
 * Reconnecting the progress stream.
 *
 * The limitation this closes: a dropped SSE stream froze the Dashboard's
 * progress until the user changed views and remounted the panel. Everything
 * asserted here is about the transport, because that is where the retry lives —
 * one loop owning one connection, so "no duplicate streams" is a property of the
 * structure rather than a rule the caller has to follow.
 *
 * Timers are faked throughout: the point of a backoff is the waiting, and a test
 * that really waits ten seconds is a test nobody runs.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { RECONNECT_BASE_MS, RECONNECT_MAX_MS, ServerClient, backoffMs } from "./api";
import type { Progress } from "./types";

const SNAPSHOT: Progress = {
  state: "idle",
  model: null,
  kind: null,
  seed: null,
  step: 0,
  total: 0,
  elapsed_s: null,
  loaded_model: null,
  memory: {},
};

/** One SSE frame, as the server writes it. */
function frame(progress: Partial<Progress> = {}): Uint8Array {
  return new TextEncoder().encode(`data: ${JSON.stringify({ ...SNAPSHOT, ...progress })}\n\n`);
}

/**
 * A stream whose lifetime the test controls: it stays open until `end` or
 * `fail` is called, so a "drop" is a thing that happens rather than a race.
 */
function openStream() {
  let push!: (chunk: Uint8Array) => void;
  let close!: () => void;
  let error!: (cause: unknown) => void;
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      push = (chunk) => controller.enqueue(chunk);
      close = () => controller.close();
      error = (cause) => controller.error(cause);
    },
  });
  return { body, push, close, error };
}

type Attempt = ReturnType<typeof openStream>;

/** Replaces `fetch`, handing each connection attempt back to the test. */
function stubFetch() {
  const attempts: Attempt[] = [];
  const signals: AbortSignal[] = [];
  const failures: Error[] = [];
  const stub = vi.fn(async (_url: string, init?: RequestInit) => {
    if (init?.signal) signals.push(init.signal);
    const failure = failures.shift();
    if (failure) throw failure;
    const attempt = openStream();
    attempts.push(attempt);
    return new Response(attempt.body, {
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
    });
  });
  vi.stubGlobal("fetch", stub);
  return { stub, attempts, signals, failures };
}

beforeEach(() => vi.useFakeTimers());
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

/** Let queued microtasks run without advancing the clock. */
const settle = () => vi.advanceTimersByTimeAsync(0);

describe("the backoff", () => {
  it("doubles and then stops growing", () => {
    expect(backoffMs(0)).toBe(RECONNECT_BASE_MS);
    expect(backoffMs(1)).toBe(1000);
    expect(backoffMs(2)).toBe(2000);
    expect(backoffMs(3)).toBe(4000);
    // Bounded: a server that is down for an hour is still retried every ten
    // seconds, not once an hour.
    expect(backoffMs(20)).toBe(RECONNECT_MAX_MS);
    expect(backoffMs(200)).toBe(RECONNECT_MAX_MS);
  });

  it("never returns a delay that would make a tight loop", () => {
    for (let attempt = 0; attempt < 50; attempt += 1) {
      expect(backoffMs(attempt)).toBeGreaterThanOrEqual(RECONNECT_BASE_MS);
      expect(backoffMs(attempt)).toBeLessThanOrEqual(RECONNECT_MAX_MS);
    }
  });
});

describe("reconnecting", () => {
  it("retries after the stream drops, without being asked", async () => {
    const { stub, attempts } = stubFetch();
    const progress = vi.fn();
    const errors: string[] = [];
    const client = new ServerClient(8765, null);

    const stop = client.subscribeProgress(progress, (message) => errors.push(message));
    await settle();
    expect(stub).toHaveBeenCalledTimes(1);

    attempts[0]!.push(frame({ step: 1 }));
    await settle();
    expect(progress).toHaveBeenCalledTimes(1);

    // The stream fails. Nothing remounts, nobody clicks anything.
    attempts[0]!.error(new Error("network error"));
    await settle();
    expect(errors).toHaveLength(1);
    expect(stub).toHaveBeenCalledTimes(1); // not yet: it waits

    await vi.advanceTimersByTimeAsync(RECONNECT_BASE_MS);
    expect(stub).toHaveBeenCalledTimes(2);

    // And the new connection delivers events.
    attempts[1]!.push(frame({ step: 2 }));
    await settle();
    expect(progress).toHaveBeenCalledTimes(2);
    expect(progress.mock.calls[1]![0]).toMatchObject({ step: 2 });

    stop();
  });

  it("treats a clean close as a drop too", async () => {
    // A server that ends the response politely used to leave the panel
    // subscribed to nothing, silently — no error, no events, no indication.
    const { stub, attempts } = stubFetch();
    const errors: string[] = [];
    const client = new ServerClient(8765, null);

    const stop = client.subscribeProgress(vi.fn(), (message) => errors.push(message));
    await settle();

    attempts[0]!.close();
    await settle();
    expect(errors).toEqual(["the server closed the progress stream"]);

    await vi.advanceTimersByTimeAsync(RECONNECT_BASE_MS);
    expect(stub).toHaveBeenCalledTimes(2);
    stop();
  });

  it("keeps exactly one connection open across a reconnect", async () => {
    const { stub, attempts, signals } = stubFetch();
    const client = new ServerClient(8765, null);
    const stop = client.subscribeProgress(vi.fn(), vi.fn());
    await settle();

    for (let round = 0; round < 3; round += 1) {
      attempts[round]!.error(new Error("drop"));
      await vi.advanceTimersByTimeAsync(RECONNECT_MAX_MS);
    }

    expect(stub).toHaveBeenCalledTimes(4);
    // Each attempt is awaited before the next begins, so at most one signal is
    // ever un-aborted: the three finished ones are done, the fourth is live.
    const live = signals.filter((signal) => !signal.aborted);
    expect(live).toHaveLength(1);

    stop();
    expect(signals.every((signal) => signal.aborted)).toBe(true);
  });

  it("backs off further on each consecutive failure, then holds at the maximum", async () => {
    const { stub, failures } = stubFetch();
    // Every attempt refuses: the server is not there at all.
    for (let i = 0; i < 10; i += 1) failures.push(new Error("connection refused"));
    const client = new ServerClient(8765, null);
    const stop = client.subscribeProgress(vi.fn(), vi.fn());
    await settle();
    expect(stub).toHaveBeenCalledTimes(1);

    for (const [index, delay] of [500, 1000, 2000, 4000, 8000, 10000, 10000].entries()) {
      // One millisecond short of the delay: still waiting.
      await vi.advanceTimersByTimeAsync(delay - 1);
      expect(stub).toHaveBeenCalledTimes(index + 1);
      await vi.advanceTimersByTimeAsync(1);
      expect(stub).toHaveBeenCalledTimes(index + 2);
    }
    stop();
  });

  it("starts the backoff over once a connection has opened", async () => {
    // An outage that has been fixed should not leave the next hiccup inheriting
    // a ten-second delay.
    const { stub, attempts, failures } = stubFetch();
    failures.push(new Error("refused"), new Error("refused"), new Error("refused"));
    const client = new ServerClient(8765, null);
    const stop = client.subscribeProgress(vi.fn(), vi.fn());
    await settle();

    await vi.advanceTimersByTimeAsync(500);
    await vi.advanceTimersByTimeAsync(1000);
    await vi.advanceTimersByTimeAsync(2000);
    expect(stub).toHaveBeenCalledTimes(4); // the fourth attempt connects
    expect(attempts).toHaveLength(1);

    attempts[0]!.error(new Error("drop"));
    await settle();
    // Back to the first delay, not the fourth.
    await vi.advanceTimersByTimeAsync(RECONNECT_BASE_MS);
    expect(stub).toHaveBeenCalledTimes(5);
    stop();
  });

  it("recovers when the server comes back", async () => {
    const { stub, attempts, failures } = stubFetch();
    const progress = vi.fn();
    const client = new ServerClient(8765, null);
    const stop = client.subscribeProgress(progress, vi.fn());
    await settle();

    // The server goes away mid-stream and refuses for a while…
    failures.push(new Error("refused"), new Error("refused"));
    attempts[0]!.error(new Error("connection reset"));
    await vi.advanceTimersByTimeAsync(500);
    await vi.advanceTimersByTimeAsync(1000);
    await vi.advanceTimersByTimeAsync(2000);

    // …then it is back, and progress resumes with no user action at all.
    expect(attempts).toHaveLength(2);
    attempts[1]!.push(frame({ step: 7, state: "generating" }));
    await settle();
    expect(progress).toHaveBeenCalledWith(expect.objectContaining({ step: 7 }));
    expect(stub).toHaveBeenCalledTimes(4);
    stop();
  });
});

describe("unsubscribing", () => {
  it("cancels a pending reconnect and stops retrying", async () => {
    const { stub, attempts } = stubFetch();
    const errors: string[] = [];
    const client = new ServerClient(8765, null);
    const stop = client.subscribeProgress(vi.fn(), (message) => errors.push(message));
    await settle();

    attempts[0]!.error(new Error("drop"));
    await settle();
    stop(); // while the backoff timer is pending

    await vi.advanceTimersByTimeAsync(RECONNECT_MAX_MS * 5);
    expect(stub).toHaveBeenCalledTimes(1);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("aborts the live connection and reports nothing for it", async () => {
    const { signals } = stubFetch();
    const errors: string[] = [];
    const client = new ServerClient(8765, null);
    const stop = client.subscribeProgress(vi.fn(), (message) => errors.push(message));
    await settle();

    stop();
    await settle();
    expect(signals[0]!.aborted).toBe(true);
    // An abort is us, not a fault: a panel on its way out must not be handed an
    // error about the teardown it asked for.
    expect(errors).toEqual([]);
  });

  it("delivers no events after unsubscribing", async () => {
    const { attempts } = stubFetch();
    const progress = vi.fn();
    const client = new ServerClient(8765, null);
    const stop = client.subscribeProgress(progress, vi.fn());
    await settle();

    attempts[0]!.push(frame({ step: 1 }));
    await settle();
    expect(progress).toHaveBeenCalledTimes(1);

    stop();
    await vi.advanceTimersByTimeAsync(RECONNECT_MAX_MS);
    expect(progress).toHaveBeenCalledTimes(1);
  });
});

import { formatBytes, describeEntry } from "./shared";
import type { DiskReport } from "../../types";

/**
 * How much disk this model is using, and what for.
 *
 * A `<details>` rather than a tooltip, for one reason: a tooltip is reachable
 * only with a pointer, and this is the only place the breakdown exists. The
 * summary is a real button to a keyboard and to a screen reader, and the
 * disclosure below it is the same information a hover would have shown.
 *
 * Nothing is estimated. A representation whose size nobody measured is left out
 * rather than derived from the source size and a bit depth, and a model whose
 * weights are not on this machine has no disk usage to report at all — its
 * catalogue size is what a download would cost, which is a different question
 * and is labelled as one.
 */
export function SizeBadge({
  disk,
  activeVariant,
}: {
  disk: DiskReport;
  activeVariant: number | null;
}) {
  const active = formatBytes(disk.active_bytes);
  if (!active) return null;
  const total = formatBytes(disk.total_bytes);
  // One representation and nothing else on disk: the breakdown would repeat the
  // badge back at the reader.
  const worthExpanding = disk.breakdown.length > 1;

  const label = activeVariant === null ? "active" : `active · ${activeVariant}-bit`;
  if (!worthExpanding) {
    return (
      <span className="pill" title={`${active} on disk`}>
        {active}
      </span>
    );
  }

  return (
    <details className="size-pop">
      <summary className="pill" title="What this model occupies on disk">
        {active} <span className="size-pop-hint">{label}</span>
      </summary>
      <div className="size-pop-body" role="group" aria-label="Disk usage">
        <dl className="size-breakdown">
          {disk.breakdown.map((entry) => (
            <div key={entry.path} className="size-line">
              <dt>
                {describeEntry(entry)}
                {entry.kind === "variant" && entry.bits === activeVariant && (
                  <span className="pill pill-accent">active</span>
                )}
              </dt>
              <dd>{formatBytes(entry.bytes) ?? "-"}</dd>
            </div>
          ))}
          <div className="size-line size-total">
            <dt>Total</dt>
            <dd>{total ?? "-"}</dd>
          </div>
        </dl>
      </div>
    </details>
  );
}

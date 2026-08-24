import { ModelRow } from "./ModelRow";
import type { RowProps } from "./shared";
import type { ModelStatus } from "../../types";

/**
 * One catalogue section.
 *
 * The grouping is what makes the list scannable, and it is drawn from the
 * backend's own `gated` flag rather than from a list kept here: a repository
 * that stops being gated moves group on the next catalogue read, with nothing
 * to update. An empty group renders nothing at all rather than a heading over
 * a hole.
 */
export function CatalogueGroup({
  heading,
  note,
  models,
  rowProps,
}: {
  heading: string;
  note: string;
  models: ModelStatus[];
  rowProps: (model: ModelStatus) => RowProps;
}) {
  // No early return for an empty group any more: it is now the *selected* tab's
  // panel, and a tab that renders nothing at all reads as a broken view rather
  // than as an empty one.
  return (
    <section className="catalogue-group">
      <h3 className="catalogue-heading">{heading}</h3>
      <p className="note" style={{ marginTop: 0 }}>
        {note}
      </p>
      {models.length === 0 ? (
        <p className="empty">Nothing in this half of the catalogue.</p>
      ) : (
        <ul className="models" aria-label={heading}>
          {models.map((model) => (
            <ModelRow key={model.key} {...rowProps(model)} />
          ))}
        </ul>
      )}
    </section>
  );
}

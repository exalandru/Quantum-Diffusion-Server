import { ModelRow } from "./ModelRow";
import { releaseSections } from "./shared";
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
 *
 * Inside it, the rows are cut again into the releases they belong to — Anima
 * Turbo and Anima Aesthetic under Anima, the three Stable Diffusion 3.5 rows
 * together — because a flat list of seventeen names is read one name at a time
 * while a list of nine releases is read by release. Both cuts are the backend's
 * (`gated`, `group_label`) and neither is recomputed here.
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
    // The tab's own accessible name lives here rather than on a list, because
    // there is no longer one list to put it on: the half is a set of releases.
    <section className="catalogue-group" aria-label={heading}>
      <h3 className="catalogue-heading">{heading}</h3>
      <p className="note" style={{ marginTop: 0 }}>
        {note}
      </p>
      {models.length === 0 ? (
        <p className="empty">Nothing in this half of the catalogue.</p>
      ) : (
        releaseSections(models).map((release) =>
          // Keyed by the release's first row rather than by its label: a label is
          // unique only while the backend keeps each release contiguous, and the
          // rendering above deliberately survives it not being.
          release.label === null ? (
            // No release declared — an older backend — so no box: the plain list
            // this was before, named by the tab it is in.
            <ul className="models" aria-label={heading} key={release.models[0]!.key}>
              {release.models.map((model) => (
                <ModelRow key={model.key} {...rowProps(model)} />
              ))}
            </ul>
          ) : (
            // The same fieldset the settings form groups by, and for the same
            // reason: a bordered box with its subject on the rim is what says
            // "these belong together" in this app already. A release of one
            // still gets one — a lone model outside a box would read as
            // differently-shaped rather than as alone.
            <fieldset className="settings-group catalogue-family" key={release.models[0]!.key}>
              <legend>{release.label}</legend>
              <ul className="models" aria-label={release.label}>
                {release.models.map((model) => (
                  <ModelRow key={model.key} {...rowProps(model)} />
                ))}
              </ul>
            </fieldset>
          ),
        )
      )}
    </section>
  );
}

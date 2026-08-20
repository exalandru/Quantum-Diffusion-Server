import { Modal } from "../modal";

/**
 * A generated image, over the page rather than in another tab.
 *
 * Built on the app's one dialog (`Modal`), so Escape, the backdrop click and the
 * focus round-trip are the same here as in the catalogue's dialogs rather than a
 * second implementation of the same three behaviours. The link to the file stays,
 * because "open the PNG itself" is still a thing you sometimes want — it is just
 * no longer what a click on the picture does.
 */
export function ImageViewer({
  url,
  caption,
  detail,
  onClose,
}: {
  url: string;
  /** The prompt that produced it, or what the image is. */
  caption: string;
  /** Seed and size, or nothing for a reference image. */
  detail?: string;
  onClose: () => void;
}) {
  return (
    <Modal title={caption} onClose={onClose} className="pg-viewer">
      <img className="pg-viewer-image" src={url} alt={caption} />
      {/* The stats sit with the file link, not beside the title: a prompt is as
          long as it likes, and the two were competing for one line. */}
      <div className="pg-viewer-foot">
        <a className="pg-viewer-link" href={url} target="_blank" rel="noreferrer">
          Open the file
        </a>
        {detail && <span className="pg-viewer-stats">{detail}</span>}
      </div>
    </Modal>
  );
}

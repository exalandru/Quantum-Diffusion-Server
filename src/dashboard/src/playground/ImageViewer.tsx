import { Modal } from "../modal";

/**
 * A generated image, over the page rather than in another tab.
 *
 * The picture and nothing else. It used to be the app's ordinary dialog with the
 * prompt as its title and a panel drawn round the image, which put a frame
 * around a frame and gave a paragraph-long prompt the top of the screen — while
 * what the user opened was the picture. So the chrome is the mockup's
 * (`.overlay-frame` / `.overlay-close` / `.overlay-bar`): a cross in a circle
 * over the top right corner, and one bar along the foot carrying the file link
 * and the run's stats.
 *
 * **`Modal` is still underneath, and deliberately.** Escape, the backdrop press
 * and the focus round-trip back to the thumbnail are three behaviours worth
 * having exactly once; reimplementing them here to remove a header would have
 * traded a visual defect for a real one. `bare` drops the head and the body
 * padding and nothing else — the prompt still names the dialog, through
 * `aria-label` rather than as visible text, so a screen reader is told which
 * image this is while the eye is given only the image.
 *
 * The close control is this component's own rather than the head's `Close`
 * button, because it is drawn *on* the picture; it calls the same `onClose`, so
 * the cross, Escape and the backdrop are one path out and not three.
 */
export function ImageViewer({
  url,
  caption,
  detail,
  onClose,
}: {
  url: string;
  /** The prompt that produced it, or what the image is. Names the dialog. */
  caption: string;
  /** Seed and size, or nothing for a reference image. */
  detail?: string;
  onClose: () => void;
}) {
  return (
    <Modal title={caption} bare onClose={onClose} className="pg-viewer">
      <img className="pg-viewer-image" src={url} alt={caption} />
      <button
        type="button"
        className="pg-viewer-close"
        aria-label="Close"
        title="Close"
        onClick={onClose}
      >
        <svg
          viewBox="0 0 24 24"
          width="17"
          height="17"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          aria-hidden="true"
        >
          <path d="M6 6l12 12M18 6 6 18" />
        </svg>
      </button>
      <div className="pg-viewer-bar">
        {/* A download, not a navigation: the URL carries the session unlock token
            as `?t=`, and opening it in a tab writes that token into the address
            bar and the browser's history. A download creates no history entry. */}
        <a className="pg-viewer-link" href={url} download rel="noreferrer">
          Download
        </a>
        {detail && <span className="pg-viewer-stats">{detail}</span>}
      </div>
    </Modal>
  );
}

/**
 * How blurred the partially-denoised preview should be, at a given step.
 *
 * The blur is not decoration. Early in a run the preview is mostly latent noise,
 * and a blur is what turns that into a readable *composition* — where the mass
 * is, roughly what colour — instead of a snowstorm. By the end the denoiser has
 * done that job itself, and any blur left is just hiding the picture.
 *
 * So it eases from `MAX_BLUR_PX` down to a floor. Two things decide it:
 *
 * * **progress**, as `step / total`. The exponent drops the blur faster than
 *   linearly at the start, where there is nothing to look at anyway, and eases
 *   out at the end where the detail is arriving.
 * * **how many steps the run has at all**, through the floor. A run's last step
 *   is not a fixed amount of "done": an 8-step schedule takes enormous jumps and
 *   its late previews are still coarse, while a 50-step one is nearly finished
 *   well before the end. `FLOOR_NUMERATOR / total` is what says that — one
 *   constant, monotone in `total`, and no table of models to keep in step with
 *   the backend.
 */

/** At the very first step, and whenever there is no step count to go on. */
export const MAX_BLUR_PX = 20;

/** `FLOOR_NUMERATOR / total`, clamped: the blur a run still carries at its end. */
export const FLOOR_NUMERATOR = 48;
export const MIN_FLOOR_PX = 0.5;
export const MAX_FLOOR_PX = 6;

/** Above 1 = fall away early, ease out late. */
export const EASE = 1.6;

/** The residual blur at the end of a run of `total` steps. */
export function floorFor(total: number): number {
  return Math.min(MAX_FLOOR_PX, Math.max(MIN_FLOOR_PX, FLOOR_NUMERATOR / total));
}

/**
 * Blur in pixels for `step` of `total`.
 *
 * A run with no step count yet — the weights are still loading, or the engine is
 * busy with somebody else's request — gets the full blur rather than a guess.
 */
export function previewBlurPx(step: number, total: number): number {
  if (!Number.isFinite(total) || total <= 0) return MAX_BLUR_PX;
  const done = Math.min(1, Math.max(0, step / total));
  const floor = floorFor(total);
  return floor + (MAX_BLUR_PX - floor) * (1 - done) ** EASE;
}

/**
 * How much to inflate the frame so its blurred edges stay outside the box.
 *
 * `.pg-preview` clips its content, and a CSS blur samples past the element's
 * edge — so an un-inflated blurred frame shows the container's background
 * bleeding in on all four sides. The inflation shrinks with the blur, so a
 * nearly-sharp late frame is cropped by about a percent.
 */
export function previewScale(blurPx: number): number {
  return 1 + blurPx / 200;
}

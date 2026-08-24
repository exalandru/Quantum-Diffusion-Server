

/**
 * A real switch.
 *
 * `role="switch"` with `aria-checked` rather than a styled checkbox or, as
 * before, an ON/OFF pill that looked interactive and was not: the state has to be
 * in the accessibility tree, not only in the pixels.
 */
export function Switch({
  checked,
  label,
  disabled,
  onChange,
}: {
  checked: boolean;
  label: string;
  disabled?: boolean;
  onChange: (next: boolean) => void;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      className="switch"
      disabled={disabled}
      onClick={() => onChange(!checked)}
    />
  );
}

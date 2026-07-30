/**
 * Direction encoding — the four channels, in one place.
 *
 * Red/green is the most common colour-vision deficiency pair. The measured
 * consequence for the obvious palette is severe (CVD ΔE 3.8, floor 6), and
 * even the validated pair this design ships sits at 12.3 — comfortably legal,
 * but not so wide that colour should be trusted alone.
 *
 * So direction is never colour-only. Every directional element carries:
 *
 *   1. colour    a CSS token, swappable for the CVD-safe or mono palette
 *   2. glyph     ▲ / ▼ / –
 *   3. label     "Up" / "Down" / "Flat"  (visible or screen-reader only)
 *   4. position  bars extend either side of a centre line; signs on values
 *
 * The acceptance test: set `data-palette="mono"` on the root and every chart
 * must still read. Channels 2-4 are what make that true.
 */

export type Direction = "up" | "down" | "flat";

/** Below this a move is rounding, and rendering it as directional invents a
 * signal the data does not contain. */
export const FLAT_EPSILON = 1e-9;

export function directionOf(value: number, epsilon = FLAT_EPSILON): Direction {
  if (!Number.isFinite(value)) return "flat";
  if (value > epsilon) return "up";
  if (value < -epsilon) return "down";
  return "flat";
}

export const GLYPH: Record<Direction, string> = {
  up: "▲",
  down: "▼",
  flat: "–",
};

export const LABEL: Record<Direction, string> = {
  up: "Up",
  down: "Down",
  flat: "Flat",
};

export const COLOR_VAR: Record<Direction, string> = {
  up: "var(--dir-up)",
  down: "var(--dir-down)",
  flat: "var(--dir-flat)",
};

export const WASH_VAR: Record<Direction, string> = {
  up: "var(--dir-up-wash)",
  down: "var(--dir-down-wash)",
  flat: "transparent",
};

/** Everything a directional element needs, so no call site can implement three
 * of the four channels and forget the fourth. */
export interface DirectionEncoding {
  direction: Direction;
  glyph: string;
  label: string;
  color: string;
  wash: string;
  /** Signed, for a bar that extends either side of a centre line. */
  extendsRight: boolean;
}

export function encode(value: number, epsilon = FLAT_EPSILON): DirectionEncoding {
  const direction = directionOf(value, epsilon);
  return {
    direction,
    glyph: GLYPH[direction],
    label: LABEL[direction],
    color: COLOR_VAR[direction],
    wash: WASH_VAR[direction],
    extendsRight: direction === "up",
  };
}

/** Format a number with an explicit sign — the fourth channel, in text.
 * "+18.2%" and "-7.5%" differ without any colour at all. */
export function signed(value: number, digits = 2, unit = "%"): string {
  if (!Number.isFinite(value)) return "n/a";
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}${unit}`;
}

/** Indian numbering: 12,34,567 rather than 1,234,567. The audience reads
 * lakhs and crores, and rendering NSE turnover in thousands separators is a
 * small, constant reminder that the product was built for somewhere else. */
export function formatINR(value: number): string {
  if (!Number.isFinite(value)) return "n/a";
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 2,
  }).format(value);
}

export function formatCompactINR(value: number): string {
  if (!Number.isFinite(value)) return "n/a";
  const abs = Math.abs(value);
  if (abs >= 1e7) return `₹${(value / 1e7).toFixed(2)} cr`;
  if (abs >= 1e5) return `₹${(value / 1e5).toFixed(2)} lakh`;
  return formatINR(value);
}

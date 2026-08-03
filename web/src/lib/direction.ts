/**
 * Direction encoding — four channels, one source.
 *
 * The measured reason colour is never alone: the obvious terminal green/red
 * scores CVD ΔE 5.1, and GitHub's own dark pair scores 3.5. The validated pair
 * this app ships is 9.7 — legal, but not so wide that hue should be trusted by
 * itself. So every directional element carries colour AND glyph AND label AND
 * position.
 *
 * Acceptance test: set data-palette="mono" and everything must still read.
 */

export type Direction = "up" | "down" | "flat";

export const FLAT_EPS = 1e-9;

export function directionOf(v: number, eps = FLAT_EPS): Direction {
  if (!Number.isFinite(v)) return "flat";
  if (v > eps) return "up";
  if (v < -eps) return "down";
  return "flat";
}

export const GLYPH: Record<Direction, string> = { up: "▲", down: "▼", flat: "–" };
export const LABEL: Record<Direction, string> = { up: "Up", down: "Down", flat: "Flat" };
export const COLOR: Record<Direction, string> = {
  up: "var(--dir-up)",
  down: "var(--dir-down)",
  flat: "var(--dir-flat)",
};
export const WASH: Record<Direction, string> = {
  up: "var(--dir-up-wash)",
  down: "var(--dir-down-wash)",
  flat: "transparent",
};

export interface Enc {
  direction: Direction;
  glyph: string;
  label: string;
  color: string;
  wash: string;
  extendsRight: boolean;
}

/** All four channels together, so no call site can implement three. */
export function encode(v: number, eps = FLAT_EPS): Enc {
  const d = directionOf(v, eps);
  return {
    direction: d,
    glyph: GLYPH[d],
    label: LABEL[d],
    color: COLOR[d],
    wash: WASH[d],
    extendsRight: d === "up",
  };
}

/** Explicit sign — the fourth channel, in text. */
export function signed(v: number, digits = 2, unit = "%"): string {
  if (!Number.isFinite(v)) return "n/a";
  return `${v >= 0 ? "+" : ""}${v.toFixed(digits)}${unit}`;
}

/** Indian grouping: 12,34,567 not 1,234,567. The audience reads lakhs and
 * crores, and thousands separators are a constant small signal that the
 * product was built for somewhere else. */
export function inr(v: number, digits = 2): string {
  if (!Number.isFinite(v)) return "n/a";
  return new Intl.NumberFormat("en-IN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(v);
}

export function compactINR(v: number): string {
  if (!Number.isFinite(v)) return "n/a";
  const a = Math.abs(v);
  if (a >= 1e7) return `₹${(v / 1e7).toFixed(2)}cr`;
  if (a >= 1e5) return `₹${(v / 1e5).toFixed(2)}L`;
  return `₹${inr(v, 0)}`;
}

export function compactVol(v: number): string {
  if (!Number.isFinite(v)) return "n/a";
  const a = Math.abs(v);
  if (a >= 1e7) return `${(v / 1e7).toFixed(2)}cr`;
  if (a >= 1e5) return `${(v / 1e5).toFixed(2)}L`;
  if (a >= 1e3) return `${(v / 1e3).toFixed(1)}k`;
  return String(Math.round(v));
}

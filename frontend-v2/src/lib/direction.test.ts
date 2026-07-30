/**
 * Direction-encoding tests.
 *
 * The colour-removal assertions are the design's stated acceptance criterion,
 * made executable: with every hue stripped, direction must still be
 * recoverable. If these pass, the four-channel claim is true; if they fail, the
 * product is unreadable for roughly 1 in 12 men and nobody would find out from
 * a screenshot.
 */

import { describe, expect, it } from "vitest";
import {
  COLOR_VAR,
  directionOf,
  encode,
  formatCompactINR,
  GLYPH,
  LABEL,
  signed,
  type Direction,
} from "./direction";

describe("directionOf", () => {
  it("classifies up, down and flat", () => {
    expect(directionOf(1)).toBe("up");
    expect(directionOf(-1)).toBe("down");
    expect(directionOf(0)).toBe("flat");
  });

  it("treats every non-finite value as flat, including Infinity", () => {
    // Infinity in a return series means a division by zero upstream — a data
    // error, not an infinite gain. Rendering it as a confident "up" would turn
    // a broken price into the strongest signal on the page, so it is grouped
    // with NaN as "not renderable" rather than given a direction.
    expect(directionOf(NaN)).toBe("flat");
    expect(directionOf(Infinity)).toBe("flat");
    expect(directionOf(-Infinity)).toBe("flat");
  });

  it("treats sub-epsilon moves as flat", () => {
    // Rendering rounding as directional invents a signal the data lacks.
    expect(directionOf(1e-12)).toBe("flat");
  });

  it("honours a caller-supplied epsilon", () => {
    expect(directionOf(0.003, 0.005)).toBe("flat");
    expect(directionOf(0.02, 0.005)).toBe("up");
  });
});

describe("the four channels", () => {
  const directions: Direction[] = ["up", "down", "flat"];

  it("gives every direction a distinct glyph", () => {
    const glyphs = new Set(directions.map((d) => GLYPH[d]));
    expect(glyphs.size).toBe(3);
  });

  it("gives every direction a distinct text label", () => {
    const labels = new Set(directions.map((d) => LABEL[d]));
    expect(labels.size).toBe(3);
  });

  it("ships a CSS token, never a hex value", () => {
    // Hex on the wire would freeze one palette and defeat the CVD and
    // monochrome alternates entirely.
    for (const d of directions) {
      expect(COLOR_VAR[d]).toMatch(/^var\(--dir-/);
      expect(COLOR_VAR[d]).not.toContain("#");
    }
  });

  it("encodes position independently of colour", () => {
    expect(encode(5).extendsRight).toBe(true);
    expect(encode(-5).extendsRight).toBe(false);
  });
});

describe("COLOUR-REMOVAL TEST — the design's acceptance criterion", () => {
  it("recovers direction from the glyph alone", () => {
    const up = encode(1);
    const down = encode(-1);
    const flat = encode(0);
    // Strip colour entirely; the glyphs must still disambiguate.
    expect(new Set([up.glyph, down.glyph, flat.glyph]).size).toBe(3);
  });

  it("recovers direction from the label alone", () => {
    expect(encode(1).label).not.toBe(encode(-1).label);
    expect(encode(1).label).not.toBe(encode(0).label);
  });

  it("recovers direction from position alone", () => {
    expect(encode(1).extendsRight).not.toBe(encode(-1).extendsRight);
  });

  it("recovers direction from the value's sign in text", () => {
    expect(signed(5)).toContain("+");
    expect(signed(-5)).toContain("-");
  });

  it("survives with THREE of four channels removed", () => {
    // The strongest form of the claim: keep only the glyph and the two
    // directions are still distinguishable.
    const channels = (v: number) => encode(v).glyph;
    expect(channels(1)).not.toBe(channels(-1));
  });
});

describe("signed", () => {
  it("always shows an explicit sign", () => {
    expect(signed(18.25)).toBe("+18.25%");
    expect(signed(-7.5)).toBe("-7.50%");
  });

  it("shows zero as positive-signed rather than bare", () => {
    expect(signed(0)).toBe("+0.00%");
  });

  it("reports non-finite values as n/a, not NaN%", () => {
    expect(signed(NaN)).toBe("n/a");
  });
});

describe("Indian number formatting", () => {
  it("uses crores above 1e7", () => {
    // The audience reads lakhs and crores. Rendering NSE turnover with
    // thousands separators is a constant small reminder that the product was
    // built for somewhere else.
    expect(formatCompactINR(25_000_000)).toBe("₹2.50 cr");
  });

  it("uses lakhs between 1e5 and 1e7", () => {
    expect(formatCompactINR(250_000)).toBe("₹2.50 lakh");
  });

  it("falls back to full currency below a lakh", () => {
    expect(formatCompactINR(5_000)).toContain("5,000");
  });

  it("reports non-finite values as n/a", () => {
    expect(formatCompactINR(NaN)).toBe("n/a");
  });
});

/**
 * The direction encoder is the only real logic in the frontend, and it is the
 * piece an accessibility claim rests on: "colour is never the sole carrier of
 * direction". These tests pin that claim so it stays falsifiable.
 */

import { describe, expect, it } from "vitest";
import {
  compactINR,
  compactVol,
  directionOf,
  encode,
  inr,
  signed,
} from "./direction";

describe("directionOf", () => {
  it("classifies sign around the epsilon", () => {
    expect(directionOf(0.5)).toBe("up");
    expect(directionOf(-0.5)).toBe("down");
    expect(directionOf(0)).toBe("flat");
  });

  it("treats a value inside the epsilon as flat, not as a tiny move", () => {
    // A 1e-12 drift is rounding, and rendering it as a green ▲ would claim a
    // direction the data does not support.
    expect(directionOf(1e-12)).toBe("flat");
    expect(directionOf(-1e-12)).toBe("flat");
  });

  it("treats non-finite values as flat rather than as up", () => {
    // NaN/Infinity reach here from a missing field. `NaN > eps` is false and
    // `NaN < -eps` is false, so this would fall through to flat anyway — but
    // Infinity would read as "up", asserting an infinite gain.
    expect(directionOf(NaN)).toBe("flat");
    expect(directionOf(Infinity)).toBe("flat");
    expect(directionOf(-Infinity)).toBe("flat");
  });
});

describe("encode", () => {
  it("carries direction on four independent channels", () => {
    const up = encode(1);
    const down = encode(-1);

    // Colour
    expect(up.color).not.toBe(down.color);
    // Glyph
    expect(up.glyph).not.toBe(down.glyph);
    // Label
    expect(up.label).not.toBe(down.label);
    // Position
    expect(up.extendsRight).toBe(true);
    expect(down.extendsRight).toBe(false);
  });

  it("remains distinguishable with colour removed", () => {
    // The monochrome acceptance test, in code: strip colour and the remaining
    // channels must still separate up from down.
    const strip = (v: number) => {
      const { color, wash, ...rest } = encode(v);
      return rest;
    };
    expect(strip(1)).not.toEqual(strip(-1));
    expect(strip(1)).not.toEqual(strip(0));
    expect(strip(-1)).not.toEqual(strip(0));
  });

  it("uses CSS variables so the palette switch reaches every element", () => {
    // Hard-coded hex here would silently opt an element out of the CVD and
    // monochrome palettes.
    expect(encode(1).color).toMatch(/^var\(--dir-/);
    expect(encode(-1).color).toMatch(/^var\(--dir-/);
  });
});

describe("signed", () => {
  it("always shows an explicit sign", () => {
    expect(signed(1.5, 1)).toBe("+1.5%");
    expect(signed(-1.5, 1)).toBe("-1.5%");
    expect(signed(0, 1)).toBe("+0.0%");
  });

  it("says n/a rather than printing NaN%", () => {
    expect(signed(NaN)).toBe("n/a");
  });
});

describe("Indian number formatting", () => {
  it("groups in lakhs, not thousands", () => {
    // 1234567 is 12,34,567 in Indian grouping — 1,234,567 would be the tell
    // that the locale was left at a default.
    expect(inr(1234567, 0)).toBe("12,34,567");
  });

  it("shortens to lakh and crore", () => {
    expect(compactINR(15_000_000)).toBe("₹1.50cr");
    expect(compactINR(250_000)).toBe("₹2.50L");
    expect(compactINR(754)).toBe("₹754");
  });

  it("formats volumes without a currency mark", () => {
    expect(compactVol(12_160_000)).toBe("1.22cr");
    expect(compactVol(5_400)).toBe("5.4k");
  });

  it("says n/a for non-finite money and volume", () => {
    expect(compactINR(NaN)).toBe("n/a");
    expect(compactVol(Infinity)).toBe("n/a");
  });
});

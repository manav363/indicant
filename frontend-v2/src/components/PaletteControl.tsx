/**
 * Theme and palette control.
 *
 * The palette options are deliberately NOT hidden in a settings page. The
 * measured reason: the obvious green/red palette sits at CVD ΔE 3.8, and even
 * the validated pair this product ships is 12.3 — legal, not luxurious. A
 * reader who finds green/red harder to read should not have to hunt, and should
 * not have to justify the preference to anyone.
 *
 * "Monochrome" is also the design's own acceptance test, shipped as a setting:
 * with it on, every chart must still be readable. Making it user-reachable
 * keeps that claim continuously falsifiable rather than true only in CI.
 */

import { useEffect } from "react";
import { applyPrefs, type Palette, useUiPrefs } from "../hooks/useUiPrefs";
import "./PaletteControl.css";

const PALETTES: { value: Palette; label: string; hint: string }[] = [
  { value: "default", label: "Green / red", hint: "the market convention" },
  { value: "cvd", label: "Blue / orange", hint: "wider colour separation" },
  { value: "mono", label: "Monochrome", hint: "no colour at all" },
];

export function PaletteControl() {
  const { theme, palette, setTheme, setPalette } = useUiPrefs();

  useEffect(() => {
    applyPrefs(theme, palette);
  }, [theme, palette]);

  return (
    <div className="prefs">
      <fieldset className="prefs__group">
        <legend className="prefs__legend">Direction colours</legend>
        {PALETTES.map((p) => (
          <label key={p.value} className="prefs__option">
            <input
              type="radio"
              name="palette"
              value={p.value}
              checked={palette === p.value}
              onChange={() => setPalette(p.value)}
            />
            <span className="prefs__label">{p.label}</span>
            <span className="prefs__hint">{p.hint}</span>
          </label>
        ))}
      </fieldset>

      <button
        type="button"
        className="prefs__theme"
        onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
        aria-pressed={theme === "dark"}
      >
        {theme === "dark" ? "Light" : "Dark"} mode
      </button>
    </div>
  );
}

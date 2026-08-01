/**
 * UI-only client state.
 *
 * v1 put server data in Zustand, which duplicated it and let the two copies
 * drift. This store holds ONLY things the server has no opinion about: theme,
 * palette, and whether the table view is expanded. Server state lives in
 * TanStack Query; shareable state lives in the URL.
 *
 * The palette choice persists because a reader who needs the colour-vision-safe
 * or monochrome variant needs it on every visit, and making them re-select it
 * each time is its own small accessibility failure.
 */

import { create } from "zustand";
import { persist } from "zustand/middleware";

export type Theme = "light" | "dark" | "system";
export type Palette = "default" | "cvd" | "mono";

interface UiPrefs {
  theme: Theme;
  palette: Palette;
  setTheme: (t: Theme) => void;
  setPalette: (p: Palette) => void;
}

export const useUiPrefs = create<UiPrefs>()(
  persist(
    (set) => ({
      theme: "system",
      palette: "default",
      setTheme: (theme) => set({ theme }),
      setPalette: (palette) => set({ palette }),
    }),
    { name: "indicant-ui-prefs" },
  ),
);

/** Stamp the root element so CSS can respond.
 *
 * `theme="system"` removes the attribute entirely rather than writing a value,
 * so the `prefers-color-scheme` media query is what applies — writing
 * `data-theme="system"` would match neither the light nor dark selector and
 * silently strand the user in the light default.
 */
export function applyPrefs(theme: Theme, palette: Palette): void {
  const root = document.documentElement;
  if (theme === "system") {
    root.removeAttribute("data-theme");
  } else {
    root.setAttribute("data-theme", theme);
  }
  if (palette === "default") {
    root.removeAttribute("data-palette");
  } else {
    root.setAttribute("data-palette", palette);
  }
}

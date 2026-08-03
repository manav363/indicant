import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      // lightweight-charts is ~52kb gz. Split so it loads with the chart, not
      // with the shell.
      output: { manualChunks: { charts: ["lightweight-charts"] } },
    },
  },
  server: { proxy: { "/api": "http://localhost:8000" } },
  test: { environment: "jsdom", globals: true, setupFiles: ["./src/test-setup.ts"] },
});

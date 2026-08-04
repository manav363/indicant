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
  // Proxy to nginx (8080), not the gateway directly. Compose stopped
  // publishing the gateway when nginx became the only ingress, so :8000 has
  // been a dead address since — and this way dev traverses the same path as
  // production, including the CSP and the /internal/ 404.
  server: {
    proxy: {
      "/api": process.env.INDICANT_API_ORIGIN || "http://localhost:8080",
    },
  },
  test: { environment: "jsdom", globals: true, setupFiles: ["./src/test-setup.ts"] },
});

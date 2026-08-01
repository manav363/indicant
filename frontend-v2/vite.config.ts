import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        // lightweight-charts is ~45kb gz and only the stock page needs it.
        // Splitting it keeps the market-pulse and model pages off that cost.
        manualChunks: { charts: ["lightweight-charts"] },
      },
    },
  },
  server: { proxy: { "/api": "http://localhost:8000" } },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test-setup.ts"],
  },
});

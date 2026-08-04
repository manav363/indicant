import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { App } from "./App";
// Self-hosted so infra/nginx.conf keeps `font-src 'self'` — a CDN would mean
// widening the CSP for the one service that faces the internet.
import "@fontsource-variable/archivo";
import "@fontsource-variable/inter";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import "@fontsource/ibm-plex-mono/600.css";
import "./styles/global.css";

const qc = new QueryClient({
  defaultOptions: {
    queries: {
      // Predictions change once a day when new prices land; refetching on
      // every focus would spend a panel scan for the same answer.
      refetchOnWindowFocus: false,
      staleTime: 300_000,
    },
  },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={qc}><App /></QueryClientProvider>
  </StrictMode>,
);

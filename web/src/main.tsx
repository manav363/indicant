import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { App } from "./App";
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

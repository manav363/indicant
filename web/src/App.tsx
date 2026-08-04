/**
 * Application shell.
 *
 * The chain rail is the spine: SOURCE → GATE → UNIVERSE → MODEL → CALL. It is
 * persistent, shows each stage's live state, and switching screens moves along
 * it. That is the whole architectural idea — the endpoint audit found the UI
 * reached 4 of 24 routes, and the 13 with no UI were the evidence tier that
 * makes this unlike a stock-tip app.
 *
 * Screen and symbol both live in the URL so a view is shareable and the back
 * button works. A terminal you cannot send someone a link to is a worse tool.
 */

import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "./lib/api";
import { ChainRail, type Screen } from "./components/ChainRail";
import { TickerSearch } from "./components/TickerSearch";
import { CallScreen } from "./screens/Call";
import { GateScreen, UniverseScreen, ModelScreen } from "./screens/Evidence";
import "./App.css";

type Palette = "default" | "cvd" | "mono";
const HORIZONS = [1, 3, 6, 12] as const;
const SCREENS: Screen[] = ["call", "gate", "universe", "model"];

function readUrl() {
  const p = new URLSearchParams(location.search);
  const h = Number(p.get("h"));
  const s = p.get("screen") as Screen | null;
  return {
    symbol: (p.get("s") || "RELIANCE").toUpperCase(),
    horizon: HORIZONS.includes(h as (typeof HORIZONS)[number]) ? h : 6,
    screen: s && SCREENS.includes(s) ? s : ("call" as Screen),
  };
}

export function App() {
  const [view, setView] = useState(readUrl);
  const { symbol, horizon, screen } = view;

  const [palette, setPalette] = useState<Palette>(
    () => (localStorage.getItem("indicant-palette") as Palette) || "default",
  );

  useEffect(() => {
    if (palette === "default") document.documentElement.removeAttribute("data-palette");
    else document.documentElement.setAttribute("data-palette", palette);
    localStorage.setItem("indicant-palette", palette);
  }, [palette]);

  useEffect(() => {
    const p = new URLSearchParams();
    p.set("screen", screen);
    if (screen === "call") { p.set("s", symbol); p.set("h", String(horizon)); }
    history.replaceState(null, "", `?${p}`);
  }, [symbol, horizon, screen]);

  useEffect(() => {
    const onPop = () => setView(readUrl());
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const chain = useQuery({ queryKey: ["chain"], queryFn: api.chain, staleTime: 300_000 });
  const stock = useQuery({
    queryKey: ["stock", symbol, horizon],
    queryFn: () => api.stock(symbol, horizon),
    staleTime: 300_000,
    enabled: screen === "call",
  });

  const p = stock.data?.prediction;
  const callLabel = p
    ? `${p.signal} ${p.probability_up.toFixed(3)}`
    : stock.isLoading
    ? "…"
    : "—";

  const sig = chain.data?.model.isSignificant;

  return (
    <div className="app">
      <a className="skip" href="#main">Skip to content</a>

      <ChainRail
        chain={chain.data}
        screen={screen}
        symbol={symbol}
        call={callLabel}
        onNavigate={(s) => setView((v) => ({ ...v, screen: s }))}
      />

      <div className="toolbar">
        <TickerSearch
          onPick={(s) => setView((v) => ({ ...v, symbol: s, screen: "call" }))}
        />

        {screen === "call" && (
          <nav className="horizons" aria-label="Prediction horizon">
            {HORIZONS.map((h) => (
              <button
                key={h} type="button" className="horizons__btn"
                aria-current={h === horizon ? "true" : undefined}
                onClick={() => setView((v) => ({ ...v, horizon: h }))}
              >
                {h}M
              </button>
            ))}
          </nav>
        )}

        <div className="toolbar__right">
          <label className="sr-only" htmlFor="pal">Direction colours</label>
          <select
            id="pal" className="toolbar__select" value={palette}
            onChange={(e) => setPalette(e.target.value as Palette)}
          >
            <option value="default">green / red</option>
            <option value="cvd">blue / orange</option>
            <option value="mono">monochrome</option>
          </select>
        </div>
      </div>

      <main id="main">
        {screen === "call" && (
          <CallScreen
            symbol={symbol} horizon={horizon} palette={palette}
            onPick={(s) => setView((v) => ({ ...v, symbol: s }))}
          />
        )}
        {screen === "gate" && <GateScreen />}
        {screen === "universe" && <UniverseScreen />}
        {screen === "model" && <ModelScreen />}
      </main>

      <footer className="footer">
        <span>
          Research output, not investment advice.
          {sig === false && " This model's edge is not distinguishable from chance."}
        </span>
        <span className="footer__enc">direction = colour + glyph + label + position</span>
      </footer>
    </div>
  );
}

/**
 * Terminal shell.
 *
 * Layout: status bar / search / chart + prediction panel / screener.
 * URL carries the symbol and horizon so a view is shareable and the back
 * button works — a terminal where you cannot send someone a link to what you
 * are looking at is a worse tool.
 */

import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, ApiError } from "./lib/api";
import { encode, signed, compactINR } from "./lib/direction";
import { Chart } from "./components/Chart";
import { TickerSearch } from "./components/TickerSearch";
import { VerdictPanel } from "./components/VerdictPanel";
import "./App.css";

type Palette = "default" | "cvd" | "mono";
const HORIZONS = [1, 3, 6, 12] as const;

function readUrl() {
  const p = new URLSearchParams(location.search);
  const h = Number(p.get("h"));
  return {
    symbol: (p.get("s") || "RELIANCE").toUpperCase(),
    horizon: HORIZONS.includes(h as (typeof HORIZONS)[number]) ? h : 6,
  };
}

export function App() {
  const [{ symbol, horizon }, setView] = useState(readUrl);
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
    p.set("s", symbol); p.set("h", String(horizon));
    history.replaceState(null, "", `?${p}`);
  }, [symbol, horizon]);

  useEffect(() => {
    const onPop = () => setView(readUrl());
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const market = useQuery({ queryKey: ["market"], queryFn: api.market, staleTime: 300_000 });
  const stock = useQuery({
    queryKey: ["stock", symbol, horizon],
    queryFn: () => api.stock(symbol, horizon),
    staleTime: 300_000,
    // Retrying a scope refusal or an untrained model just repeats a correct
    // answer. Only transient failures deserve a second attempt.
    retry: (n, e) => !(e instanceof ApiError && (e.isScope || e.isUntrained)) && n < 2,
  });
  const screen = useQuery({
    queryKey: ["screen", horizon],
    queryFn: () => api.screen(horizon, 12),
    staleTime: 300_000,
  });

  const m = market.data;
  const sig = m?.model.isSignificant;

  // "Not loaded yet" is NOT "no lake" and NOT "untrained".
  //
  // These read from `m?.…` with a falsy fallback, which meant that for the
  // couple of seconds /api/market was in flight the status bar stated, as
  // fact, that there was no data and no trained model. That is the one kind
  // of wrong this project cannot ship: an unknown rendered as a negative
  // finding. Pending gets its own visible state.
  const pending = market.isLoading;

  return (
    <div className="term">
      <header className="statusbar">
        <span className="statusbar__brand">INDICANT</span>
        <span className="statusbar__sep" aria-hidden="true">│</span>
        <span className="statusbar__item">
          <em>DATA</em>{" "}
          {pending
            ? <span className="statusbar__pending">checking…</span>
            : m?.lake.hasData
            ? <span className="num">{m.lake.lastDate} · {m.lake.tradingDays.toLocaleString("en-IN")}d</span>
            : <span className="statusbar__warn">no lake</span>}
        </span>
        <span className="statusbar__sep" aria-hidden="true">│</span>
        <span className="statusbar__item">
          <em>REGIME</em>{" "}
          {pending ? (
            <span className="statusbar__pending">…</span>
          ) : (
            <span data-regime={m?.regime?.majority_regime ?? "unknown"}>
              {m?.regime?.majority_regime ?? "—"}
            </span>
          )}
        </span>
        <span className="statusbar__sep" aria-hidden="true">│</span>
        {/* The model's honest standing lives in the status bar, permanently.
            A terminal that shows predictions without showing whether they are
            distinguishable from chance is hiding the most important number. */}
        <span className="statusbar__item">
          <em>MODEL</em>{" "}
          {pending ? (
            <span className="statusbar__pending">…</span>
          ) : !m?.model.trained ? (
            <span className="statusbar__warn">untrained</span>
          ) : sig === null ? (
            <span className="statusbar__warn">untested</span>
          ) : sig ? (
            <span className="num">p={m.model.pValue?.toFixed(4)} significant</span>
          ) : (
            <span className="statusbar__warn num">
              p={m.model.pValue?.toFixed(4)} NOT significant
            </span>
          )}
        </span>

        <div className="statusbar__right">
          <label className="sr-only" htmlFor="pal">Direction colours</label>
          <select
            id="pal" className="statusbar__select" value={palette}
            onChange={(e) => setPalette(e.target.value as Palette)}
          >
            <option value="default">green / red</option>
            <option value="cvd">blue / orange</option>
            <option value="mono">monochrome</option>
          </select>
        </div>
      </header>

      <div className="toolbar">
        <TickerSearch onPick={(s) => setView((v) => ({ ...v, symbol: s }))} />
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
        <span className="toolbar__symbol num">{symbol}</span>
      </div>

      <main className="grid" id="main">
        <section className="grid__chart" aria-label={`${symbol} price chart`}>
          {stock.isLoading && <div className="chart chart--empty">loading {symbol}…</div>}
          {stock.error && (
            <div className="chart chart--empty" data-kind={
              stock.error instanceof ApiError && stock.error.isScope ? "scope" : "fail"
            }>
              {stock.error instanceof ApiError ? stock.error.userMessage : "Failed to load."}
            </div>
          )}
          {stock.data && (
            <Chart
              candles={stock.data.candles}
              volume={stock.data.volume}
              symbol={symbol}
              palette={palette}
            />
          )}
        </section>

        <VerdictPanel
          prediction={stock.data?.prediction ?? null}
          unavailable={stock.data?.predictionUnavailable}
        />

        <section className="grid__screen" aria-labelledby="screen-h">
          <h2 className="panel__title" id="screen-h">Screener · {horizon}M</h2>
          {/* Ranking the universe means one prediction per symbol, so this is
              the slowest call on the page. Without a pending state the section
              was a bare heading over empty space, which reads as "nothing to
              show" rather than "still working". */}
          {screen.isLoading && (
            <p className="screen__none">ranking the universe…</p>
          )}
          {screen.error && (
            <p className="screen__none">
              {screen.error instanceof ApiError
                ? screen.error.userMessage
                : "The screener failed to load."}
            </p>
          )}
          {screen.data?.unavailable && (
            <p className="screen__none">{screen.data.unavailable.user_message}</p>
          )}
          {screen.data?.rows && screen.data.rows.length > 0 && (
            <table className="screen">
              <thead>
                <tr>
                  <th scope="col">Symbol</th>
                  <th scope="col">Sig</th>
                  <th scope="col">P(up)</th>
                  <th scope="col">Edge</th>
                  <th scope="col">Price</th>
                </tr>
              </thead>
              <tbody>
                {screen.data.rows.map((r) => {
                  const edgePct = (r.probability_up - 0.5) * 200;
                  const e = encode(edgePct, 0.5);
                  return (
                    <tr
                      key={r.symbol}
                      className={r.symbol === symbol ? "is-current" : undefined}
                      onClick={() => setView((v) => ({ ...v, symbol: r.symbol }))}
                    >
                      <th scope="row" className="num">{r.symbol}</th>
                      <td style={{ color: e.color }}>
                        <span aria-hidden="true">{e.glyph}</span> {r.signal}
                        <span className="sr-only"> ({e.label})</span>
                      </td>
                      <td className="num">{r.probability_up.toFixed(3)}</td>
                      <td className="screen__bar-cell">
                        {/* Position encodes direction independently of colour —
                            bars extend either side of a centre line. */}
                        <span className="screen__track">
                          <span className="screen__centre" aria-hidden="true" />
                          <span
                            className="screen__bar"
                            style={{
                              width: `${Math.min(50, Math.abs(edgePct) * 2)}%`,
                              [e.extendsRight ? "left" : "right"]: "50%",
                              background: e.color,
                            }}
                          />
                        </span>
                        <span className="num screen__edge">{signed(edgePct, 1)}</span>
                      </td>
                      <td className="num">{compactINR(r.current_price)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </section>
      </main>

      <footer className="footer">
        <span>
          Research output, not investment advice.
          {sig === false && " This model's edge is not distinguishable from chance."}
        </span>
        <span className="footer__enc">
          direction = colour + glyph + label + position
        </span>
      </footer>
    </div>
  );
}
